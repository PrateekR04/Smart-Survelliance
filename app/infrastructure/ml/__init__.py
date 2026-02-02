"""ML infrastructure package."""

from app.infrastructure.ml.detector import (
    DetectionError,
    PlateDetector,
    YOLOPlateDetector,
    crop_plate_region,
)
from app.infrastructure.ml.model_loader import ModelLoader, ModelLoadError
from app.infrastructure.ml.ocr import (
    EasyOCREngine,
    MockOCREngine,
    OCREngine,
    OCRError,
    get_ocr_engine,
)

__all__ = [
    # Model Loader
    "ModelLoader",
    "ModelLoadError",
    # Detector
    "PlateDetector",
    "YOLOPlateDetector",
    "DetectionError",
    "crop_plate_region",
    # OCR
    "OCREngine",
    "EasyOCREngine",
    "MockOCREngine",
    "OCRError",
    "get_ocr_engine",
]
