"""
Plate detector interface and implementations.

Provides abstraction over plate detection models for easy swapping
and testing. All inference runs outside async event loop via threadpool.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.core.logging import get_logger
from app.domain.models import BoundingBox, PlateDetectionResult
from app.infrastructure.ml.model_loader import ModelLoader

logger = get_logger(__name__)


class DetectionError(Exception):
    """Raised when plate detection fails."""
    
    pass


class PlateDetector(ABC):
    """
    Abstract base class for plate detectors.
    
    Implementations must provide the detect method that returns
    bounding boxes and confidence scores.
    """
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[PlateDetectionResult]:
        """
        Detect number plates in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
        
        Returns:
            list: List of PlateDetectionResult with bounding boxes.
        
        Raises:
            DetectionError: If detection fails.
        """
        pass


class YOLOPlateDetector(PlateDetector):
    """
    Plate detector using YOLO model.
    
    Supports ultralytics YOLO format and generic PyTorch models.
    Uses the singleton ModelLoader for model access.
    
    Example:
        detector = YOLOPlateDetector(confidence_threshold=0.5)
        results = detector.detect(image)
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize detector.
        
        Args:
            confidence_threshold: Minimum confidence to accept detection.
        """
        self.confidence_threshold = confidence_threshold
        self._loader = ModelLoader.get_instance()
    
    def detect(self, image: np.ndarray) -> list[PlateDetectionResult]:
        """
        Detect plates using YOLO model.
        
        Args:
            image: Input image (BGR format).
        
        Returns:
            list: Detected plates with bounding boxes and confidence.
        
        Raises:
            DetectionError: If model not loaded or inference fails.
        """
        if not self._loader.is_loaded():
            raise DetectionError("Model not loaded")
        
        model = self._loader.get_model()
        height, width = image.shape[:2]
        
        try:
            results = self._run_inference(model, image)
            return self._parse_results(results, width, height)
            
        except Exception as e:
            logger.error("detection_failed", error=str(e))
            raise DetectionError(f"Detection failed: {e}") from e
    
    def _run_inference(self, model: Any, image: np.ndarray) -> Any:
        """
        Run model inference.
        
        Args:
            model: Loaded detection model.
            image: Input image.
        
        Returns:
            Raw model output.
        """
        if hasattr(model, "predict"):
            # Ultralytics YOLO
            return model.predict(image, conf=self.confidence_threshold, verbose=False)
        elif hasattr(model, "__call__"):
            # Mock or generic callable
            return model(image)
        else:
            raise DetectionError("Model doesn't have predict or __call__ method")
    
    def _parse_results(
        self,
        results: Any,
        image_width: int,
        image_height: int,
    ) -> list[PlateDetectionResult]:
        """
        Parse model output to PlateDetectionResult.
        
        Args:
            results: Raw model output.
            image_width: Original image width.
            image_height: Original image height.
        
        Returns:
            list: Parsed detection results.
        """
        detections = []
        
        # Handle ultralytics YOLO output
        if hasattr(results, "__iter__") and not isinstance(results, (dict, list)):
            for result in results:
                if hasattr(result, "boxes"):
                    for box_data in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                    ):
                        bbox, conf = box_data
                        if conf >= self.confidence_threshold:
                            detections.append(
                                PlateDetectionResult(
                                    bounding_box=BoundingBox(
                                        x1=int(bbox[0]),
                                        y1=int(bbox[1]),
                                        x2=int(bbox[2]),
                                        y2=int(bbox[3]),
                                    ),
                                    confidence=float(conf),
                                    image_width=image_width,
                                    image_height=image_height,
                                )
                            )
        
        # Handle list of dicts (mock detector format)
        elif isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    box = result.get("box") or result.get("bbox")
                    conf = result.get("confidence", 0.0)
                    
                    if box and conf >= self.confidence_threshold:
                        detections.append(
                            PlateDetectionResult(
                                bounding_box=BoundingBox(
                                    x1=int(box[0]),
                                    y1=int(box[1]),
                                    x2=int(box[2]),
                                    y2=int(box[3]),
                                ),
                                confidence=float(conf),
                                image_width=image_width,
                                image_height=image_height,
                            )
                        )
        
        logger.debug(
            "detection_complete",
            num_detections=len(detections),
        )
        
        return detections


def crop_plate_region(
    image: np.ndarray,
    detection: PlateDetectionResult,
    padding: int = 5,
) -> np.ndarray:
    """
    Crop the plate region from an image.
    
    Args:
        image: Full input image.
        detection: Detection result with bounding box.
        padding: Extra pixels to include around the box.
    
    Returns:
        np.ndarray: Cropped plate region.
    """
    bbox = detection.bounding_box
    height, width = image.shape[:2]
    
    # Apply padding with bounds checking
    x1 = max(0, bbox.x1 - padding)
    y1 = max(0, bbox.y1 - padding)
    x2 = min(width, bbox.x2 + padding)
    y2 = min(height, bbox.y2 + padding)
    
    return image[y1:y2, x1:x2]
