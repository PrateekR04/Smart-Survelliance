"""Infrastructure layer package."""

from app.infrastructure.db import (
    AccessLogRepository,
    AlertRepository,
    WhitelistRepository,
    close_db,
    get_session,
    init_db,
)
from app.infrastructure.ml import (
    DetectionError,
    ModelLoadError,
    ModelLoader,
    OCRError,
    PlateDetector,
    YOLOPlateDetector,
    get_ocr_engine,
)
from app.infrastructure.storage import ImageStorage, StorageError

__all__ = [
    # Database
    "get_session",
    "init_db",
    "close_db",
    "WhitelistRepository",
    "AccessLogRepository",
    "AlertRepository",
    # ML
    "ModelLoader",
    "ModelLoadError",
    "PlateDetector",
    "YOLOPlateDetector",
    "DetectionError",
    "OCRError",
    "get_ocr_engine",
    # Storage
    "ImageStorage",
    "StorageError",
]
