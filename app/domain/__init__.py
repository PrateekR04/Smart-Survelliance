"""Domain layer package - business rules and core models."""

from app.domain.models import (
    AccessLog,
    Alert,
    BoundingBox,
    Decision,
    OCRResult,
    PlateDetectionResult,
    PlateVerificationResult,
    VerificationStatus,
    WhitelistEntry,
)
from app.domain.services import (
    ConfidenceThresholdEvaluator,
    DecisionMaker,
    PlateTextNormalizer,
    PlateValidator,
)

__all__ = [
    # Models
    "AccessLog",
    "Alert",
    "BoundingBox",
    "Decision",
    "OCRResult",
    "PlateDetectionResult",
    "PlateVerificationResult",
    "VerificationStatus",
    "WhitelistEntry",
    # Services
    "ConfidenceThresholdEvaluator",
    "DecisionMaker",
    "PlateTextNormalizer",
    "PlateValidator",
]
