"""
Domain models for Smart Parking Access Control System.

These are pure domain objects with no infrastructure dependencies.
They represent the core business concepts and rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import NamedTuple


class VerificationStatus(str, Enum):
    """
    Status of plate verification result.
    
    Includes both successful and failure states for proper
    error handling and analytics.
    """
    
    AUTHORIZED = "authorized"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"
    OCR_FAILED = "ocr_failed"
    DETECTION_FAILED = "detection_failed"


class Decision(str, Enum):
    """
    Action to take based on verification result.
    
    ALLOW: Grant access to the vehicle.
    ALERT: Notify guard and deny access.
    """
    
    ALLOW = "allow"
    ALERT = "alert"


class BoundingBox(NamedTuple):
    """
    Bounding box coordinates for detected plate region.
    
    Coordinates are in pixels, origin at top-left.
    """
    
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Area of the bounding box in pixels."""
        return self.width * self.height
    
    @property
    def center(self) -> tuple[int, int]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass(frozen=True)
class PlateDetectionResult:
    """
    Result from plate detection model.
    
    Attributes:
        bounding_box: Detected plate region coordinates.
        confidence: Model confidence score (0.0 to 1.0).
        image_width: Original image width for normalization.
        image_height: Original image height for normalization.
    """
    
    bounding_box: BoundingBox
    confidence: float
    image_width: int
    image_height: int
    
    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass(frozen=True)
class OCRResult:
    """
    Result from OCR extraction on plate region.
    
    Attributes:
        raw_text: Raw text extracted by OCR.
        confidence: OCR confidence score (0.0 to 1.0).
    """
    
    raw_text: str
    confidence: float
    
    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass(frozen=True)
class PlateVerificationResult:
    """
    Final verification result for a plate check request.
    
    This is the primary output of the verification use case,
    containing all information needed for the API response.
    
    Attributes:
        plate_number: Normalized plate number (or None if detection/OCR failed).
        status: Verification status (authorized, unauthorized, or error state).
        action: Action to take (allow or alert).
        confidence: Combined confidence score from detector and OCR.
        raw_ocr_text: Original text before normalization (for debugging).
        detection_confidence: Confidence from plate detector.
        ocr_confidence: Confidence from OCR engine.
    """
    
    plate_number: str | None
    status: VerificationStatus
    action: Decision
    confidence: float
    raw_ocr_text: str | None = None
    detection_confidence: float | None = None
    ocr_confidence: float | None = None


@dataclass
class WhitelistEntry:
    """
    Domain model for a whitelisted vehicle.
    
    Attributes:
        plate_number: Unique normalized plate number.
        owner_name: Name of the vehicle owner.
        vehicle_type: Type of vehicle (car, bike, truck, etc.).
        is_active: Soft delete flag.
        created_at: When the entry was created.
        updated_at: When the entry was last updated.
    """
    
    plate_number: str
    owner_name: str
    vehicle_type: str = "car"
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessLog:
    """
    Domain model for an access log entry.
    
    Records every plate verification attempt for auditing
    and analytics purposes.
    
    Attributes:
        id: Unique log identifier (set by database).
        plate_number: Detected plate number (may be None if detection failed).
        camera_id: Identifier of the camera that captured the image.
        timestamp: When the verification occurred.
        confidence: Combined confidence score.
        status: Verification status.
        decision: Action taken.
        image_path: Path to stored image (if saved).
    """
    
    plate_number: str | None
    camera_id: str
    timestamp: datetime
    confidence: float
    status: VerificationStatus
    decision: Decision
    image_path: str | None = None
    id: int | None = None


@dataclass
class Alert:
    """
    Domain model for an unauthorized access alert.
    
    Created when an unauthorized or unknown plate is detected.
    
    Attributes:
        id: Unique alert identifier (set by database).
        access_log_id: Reference to the access log entry.
        plate_number: Detected plate number.
        camera_id: Camera that detected the plate.
        timestamp: When the alert was created.
        is_acknowledged: Whether a guard has acknowledged the alert.
        acknowledged_by: Username of acknowledging guard.
        acknowledged_at: When the alert was acknowledged.
        image_path: Path to stored image.
    """
    
    access_log_id: int
    plate_number: str | None
    camera_id: str
    timestamp: datetime
    image_path: str | None = None
    is_acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    id: int | None = None
