"""
Plate verification API routes.

Provides endpoints for verifying vehicle plates
from camera images.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.api.deps import ApiKeyAuth, Idempotency, RateLimited, VerificationUseCase
from app.core.logging import get_logger
from app.domain.models import Decision, VerificationStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/plates", tags=["plates"])


class PlateVerificationResponse(BaseModel):
    """Response model for plate verification."""
    
    plate_number: str | None = Field(
        description="Detected and normalized plate number",
        examples=["MH12AB1234"],
    )
    status: VerificationStatus = Field(
        description="Verification status",
        examples=["authorized"],
    )
    action: Decision = Field(
        description="Action to take",
        examples=["allow"],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined confidence score",
        examples=[0.92],
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "plate_number": "MH12AB1234",
                    "status": "authorized",
                    "action": "allow",
                    "confidence": 0.92,
                }
            ]
        }
    }


class PlateVerificationDetailedResponse(PlateVerificationResponse):
    """Detailed response with additional debug info."""
    
    raw_ocr_text: str | None = Field(
        default=None,
        description="Raw text from OCR before normalization",
    )
    detection_confidence: float | None = Field(
        default=None,
        description="Confidence from plate detector",
    )
    ocr_confidence: float | None = Field(
        default=None,
        description="Confidence from OCR engine",
    )


@router.post(
    "/verify",
    response_model=PlateVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify vehicle plate",
    description="Upload a camera image to detect and verify the vehicle plate against the whitelist.",
    responses={
        200: {"description": "Verification completed successfully"},
        400: {"description": "Invalid image or request"},
        401: {"description": "Invalid API key"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def verify_plate(
    image: Annotated[UploadFile, File(description="Camera image containing vehicle plate")],
    camera_id: Annotated[str, Form(description="Camera identifier", examples=["CAM001"])],
    use_case: VerificationUseCase,
    idempotency: Idempotency,
    _: ApiKeyAuth,
    __: RateLimited,
    timestamp: Annotated[datetime | None, Form(description="Capture timestamp")] = None,
    include_details: Annotated[bool, Form(description="Include detailed response")] = False,
) -> PlateVerificationResponse | PlateVerificationDetailedResponse:
    """
    Verify a vehicle plate from camera image.
    
    This endpoint accepts a camera image, detects the number plate,
    performs OCR, validates against the whitelist, and returns
    the verification result with an action decision.
    
    **Authentication**: Requires X-API-Key header.
    
    **Rate Limiting**: Limited to 100 requests per minute per IP.
    """
    # Validate image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image.",
        )
    
    # Read image bytes
    try:
        image_bytes = await image.read()
    except Exception as e:
        logger.error("image_read_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read image file",
        )
    
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image file",
        )
    
    # Check for duplicate request
    idempotency_key = idempotency.compute_key(image_bytes, camera_id)
    if idempotency.is_duplicate(idempotency_key):
        cached = idempotency.get_cached_response(idempotency_key)
        if cached:
            logger.info("returning_cached_response", camera_id=camera_id)
            return cached
    
    # Run verification
    result = await use_case.verify(
        image_bytes=image_bytes,
        camera_id=camera_id,
        timestamp=timestamp,
    )
    
    # Build response
    if include_details:
        response = PlateVerificationDetailedResponse(
            plate_number=result.plate_number,
            status=result.status,
            action=result.action,
            confidence=result.confidence,
            raw_ocr_text=result.raw_ocr_text,
            detection_confidence=result.detection_confidence,
            ocr_confidence=result.ocr_confidence,
        )
    else:
        response = PlateVerificationResponse(
            plate_number=result.plate_number,
            status=result.status,
            action=result.action,
            confidence=result.confidence,
        )
    
    # Cache response for idempotency
    idempotency.mark_seen(idempotency_key, response)
    
    return response
