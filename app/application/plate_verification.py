"""
Plate verification use case.

Orchestrates the complete plate verification pipeline:
1. Image preprocessing
2. Plate detection
3. Region cropping  
4. OCR extraction
5. Text normalization & validation
6. Whitelist lookup
7. Decision generation
8. Access log creation
"""

from datetime import datetime

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.models import (
    AccessLog,
    Decision,
    OCRResult,
    PlateDetectionResult,
    PlateVerificationResult,
    VerificationStatus,
)
from app.domain.services import (
    ConfidenceThresholdEvaluator,
    DecisionMaker,
    PlateTextNormalizer,
    PlateValidator,
)
from app.infrastructure.db.repository import AccessLogRepository, WhitelistRepository
from app.infrastructure.ml.detector import YOLOPlateDetector, crop_plate_region
from app.infrastructure.ml.ocr import OCREngine, get_ocr_engine
from app.infrastructure.storage.storage import ImageStorage

logger = get_logger(__name__)


class PlateVerificationUseCase:
    """
    Use case for verifying vehicle plates against whitelist.
    
    Implements the complete verification pipeline with proper
    error handling and fail-safe behavior.
    
    Example:
        use_case = PlateVerificationUseCase(session)
        result = await use_case.verify(image_bytes, "CAM001")
    """
    
    def __init__(
        self,
        session: AsyncSession,
        detector: YOLOPlateDetector | None = None,
        ocr_engine: OCREngine | None = None,
        storage: ImageStorage | None = None,
    ):
        """
        Initialize verification use case.
        
        Args:
            session: Database session for repository access.
            detector: Optional custom detector.
            ocr_engine: Optional custom OCR engine.
            storage: Optional custom storage service.
        """
        self._session = session
        self._whitelist_repo = WhitelistRepository(session)
        self._access_log_repo = AccessLogRepository(session)
        
        settings = get_settings()
        
        # ML components
        self._detector = detector or YOLOPlateDetector(
            confidence_threshold=settings.detector_confidence_threshold
        )
        self._ocr_engine = ocr_engine or get_ocr_engine()
        self._storage = storage or ImageStorage()
        
        # Domain services
        self._normalizer = PlateTextNormalizer()
        self._validator = PlateValidator()
        self._confidence_evaluator = ConfidenceThresholdEvaluator(
            threshold=settings.ocr_confidence_threshold
        )
        self._decision_maker = DecisionMaker()
    
    async def verify(
        self,
        image_bytes: bytes,
        camera_id: str,
        timestamp: datetime | None = None,
    ) -> PlateVerificationResult:
        """
        Verify a plate from camera image.
        
        This is the main entry point for plate verification.
        All ML operations run outside the async event loop.
        
        Args:
            image_bytes: Raw image data from camera.
            camera_id: Identifier of the source camera.
            timestamp: Optional capture timestamp.
        
        Returns:
            PlateVerificationResult: Verification result with decision.
        """
        ts = timestamp or datetime.utcnow()
        
        logger.info(
            "verification_started",
            camera_id=camera_id,
            image_size=len(image_bytes),
        )
        
        # Decode image
        image = await run_in_threadpool(self._decode_image, image_bytes)
        if image is None:
            return await self._handle_failure(
                VerificationStatus.DETECTION_FAILED,
                camera_id,
                ts,
                None,
                "Failed to decode image",
            )
        
        # Run detection outside event loop
        try:
            detections = await run_in_threadpool(self._detector.detect, image)
        except Exception as e:
            logger.error("detection_error", error=str(e))
            return await self._handle_failure(
                VerificationStatus.DETECTION_FAILED,
                camera_id,
                ts,
                image,
                str(e),
            )
        
        if not detections:
            logger.warning("no_plate_detected", camera_id=camera_id)
            return await self._handle_failure(
                VerificationStatus.DETECTION_FAILED,
                camera_id,
                ts,
                image,
                "No plate detected",
            )
        
        # Use highest confidence detection
        best_detection = max(detections, key=lambda d: d.confidence)
        
        # Crop plate region
        plate_image = await run_in_threadpool(
            crop_plate_region, image, best_detection
        )
        
        # Run OCR outside event loop
        try:
            ocr_result = await run_in_threadpool(
                self._ocr_engine.extract_text, plate_image
            )
        except Exception as e:
            logger.error("ocr_error", error=str(e))
            return await self._handle_failure(
                VerificationStatus.OCR_FAILED,
                camera_id,
                ts,
                image,
                str(e),
                detection_confidence=best_detection.confidence,
            )
        
        # Normalize and validate text
        normalized_plate = self._normalizer.normalize(ocr_result.raw_text)
        
        logger.debug(
            "plate_text_processing",
            raw_text=ocr_result.raw_text,
            normalized=normalized_plate,
        )
        
        # Check confidence threshold
        final_confidence = self._confidence_evaluator.compute_final_confidence(
            best_detection.confidence,
            ocr_result.confidence,
        )
        
        is_confident = self._confidence_evaluator.is_confident(
            best_detection.confidence,
            ocr_result.confidence,
        )
        
        # Validate plate format
        is_valid_format = self._validator.is_valid(normalized_plate)
        
        logger.debug(
            "plate_validation",
            normalized=normalized_plate,
            is_valid_format=is_valid_format,
            is_confident=is_confident,
            final_confidence=final_confidence,
        )
        
        if not is_confident or not is_valid_format:
            logger.warning(
                "verification_uncertain",
                plate=normalized_plate,
                confidence=final_confidence,
                is_valid_format=is_valid_format,
            )
            return await self._create_result(
                plate_number=normalized_plate if is_valid_format else None,
                status=VerificationStatus.UNKNOWN,
                confidence=final_confidence,
                camera_id=camera_id,
                timestamp=ts,
                image=image,
                raw_ocr_text=ocr_result.raw_text,
                detection_confidence=best_detection.confidence,
                ocr_confidence=ocr_result.confidence,
            )
        
        # Check whitelist
        is_whitelisted = await self._whitelist_repo.is_whitelisted(normalized_plate)
        
        status = (
            VerificationStatus.AUTHORIZED if is_whitelisted
            else VerificationStatus.UNAUTHORIZED
        )
        
        logger.info(
            "verification_complete",
            plate=normalized_plate,
            status=status.value,
            confidence=final_confidence,
        )
        
        return await self._create_result(
            plate_number=normalized_plate,
            status=status,
            confidence=final_confidence,
            camera_id=camera_id,
            timestamp=ts,
            image=image,
            raw_ocr_text=ocr_result.raw_text,
            detection_confidence=best_detection.confidence,
            ocr_confidence=ocr_result.confidence,
        )
    
    def _decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        """Decode image bytes to numpy array."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error("image_decode_failed", error=str(e))
            return None
    
    async def _handle_failure(
        self,
        status: VerificationStatus,
        camera_id: str,
        timestamp: datetime,
        image: np.ndarray | None,
        error_message: str,
        detection_confidence: float | None = None,
    ) -> PlateVerificationResult:
        """Handle detection or OCR failure."""
        logger.warning(
            "verification_failed",
            status=status.value,
            error=error_message,
        )
        
        return await self._create_result(
            plate_number=None,
            status=status,
            confidence=0.0,
            camera_id=camera_id,
            timestamp=timestamp,
            image=image,
            raw_ocr_text=None,
            detection_confidence=detection_confidence,
            ocr_confidence=None,
        )
    
    async def _create_result(
        self,
        plate_number: str | None,
        status: VerificationStatus,
        confidence: float,
        camera_id: str,
        timestamp: datetime,
        image: np.ndarray | None,
        raw_ocr_text: str | None,
        detection_confidence: float | None,
        ocr_confidence: float | None,
    ) -> PlateVerificationResult:
        """Create verification result and log to database."""
        decision = self._decision_maker.make_decision(status)
        
        # Save image if available
        image_path = None
        if image is not None:
            try:
                image_path = await run_in_threadpool(
                    self._storage.save,
                    image,
                    plate_number,
                    camera_id,
                    timestamp,
                )
            except Exception as e:
                logger.error("image_save_failed", error=str(e))
        
        # Create access log
        access_log = AccessLog(
            plate_number=plate_number,
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=confidence,
            status=status,
            decision=decision,
            image_path=image_path,
        )
        
        saved_log = await self._access_log_repo.create(access_log)
        
        # Create alert for non-authorized access
        if status != VerificationStatus.AUTHORIZED and saved_log.id is not None:
            from app.application.alert_service import AlertService
            alert_service = AlertService(self._session)
            await alert_service.create_from_access_log(saved_log)
        
        return PlateVerificationResult(
            plate_number=plate_number,
            status=status,
            action=decision,
            confidence=confidence,
            raw_ocr_text=raw_ocr_text,
            detection_confidence=detection_confidence,
            ocr_confidence=ocr_confidence,
        )

