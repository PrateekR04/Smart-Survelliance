"""
Thread-safe model loader for number plate detection.

Handles loading pre-trained YOLO model weights with:
- Singleton pattern for single load at startup
- Thread-safe initialization
- Model warm-up to prevent slow first inference
- Fail-fast on invalid weights
"""

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when model fails to load."""
    
    pass


class ModelLoader:
    """
    Thread-safe singleton loader for plate detection model.
    
    The model is loaded once at application startup and reused
    for all subsequent inference requests.
    
    Example:
        loader = ModelLoader.get_instance()
        model = loader.get_model()
        results = model(image)
    """
    
    _instance: "ModelLoader | None" = None
    _lock: threading.Lock = threading.Lock()
    _model: Any = None
    _initialized: bool = False
    
    def __new__(cls) -> "ModelLoader":
        """Ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """
        Get the singleton model loader instance.
        
        Returns:
            ModelLoader: The singleton instance.
        """
        return cls()
    
    def load_model(self, model_path: str | None = None) -> None:
        """
        Load the plate detection model from disk.
        
        Should be called once during application startup.
        Performs warm-up inference to ensure fast first request.
        
        Args:
            model_path: Optional override for model path.
        
        Raises:
            ModelLoadError: If model fails to load or weights are invalid.
        """
        if self._initialized:
            logger.warning("model_already_loaded", msg="Model loader called twice")
            return
        
        with self._lock:
            if self._initialized:
                return
            
            settings = get_settings()
            path = Path(model_path or settings.ml_model_path)
            
            logger.info("model_loading", path=str(path))
            
            try:
                # Check if weights file exists
                if not path.exists():
                    logger.warning(
                        "model_weights_not_found",
                        path=str(path),
                        msg="Using mock detector for development",
                    )
                    self._model = self._create_mock_model()
                else:
                    # Load YOLO model (or custom model)
                    self._model = self._load_yolo_model(path)
                
                # Warm-up inference
                self._warmup()
                
                self._initialized = True
                logger.info("model_loaded", path=str(path))
                
            except Exception as e:
                logger.error("model_load_failed", error=str(e))
                raise ModelLoadError(f"Failed to load model from {path}: {e}") from e
    
    def _load_yolo_model(self, path: Path) -> Any:
        """
        Load YOLO model from weights file.
        
        Args:
            path: Path to .pt weights file.
        
        Returns:
            Loaded model ready for inference.
        """
        try:
            # Try ultralytics YOLO (YOLOv8)
            from ultralytics import YOLO
            
            model = YOLO(str(path))
            return model
        except ImportError:
            logger.warning(
                "ultralytics_not_installed",
                msg="Falling back to torch.load",
            )
            # Fallback to raw PyTorch load
            model = torch.load(str(path), map_location="cpu")
            if hasattr(model, "eval"):
                model.eval()
            return model
    
    def _create_mock_model(self) -> "MockDetector":
        """
        Create a mock detector for development/testing.
        
        Returns:
            MockDetector: Fake detector that returns random boxes.
        """
        return MockDetector()
    
    def _warmup(self) -> None:
        """
        Perform warm-up inference to optimize first request.
        
        This prevents slow cold-start on first real inference.
        """
        logger.info("model_warmup_start")
        
        # Create dummy input (640x640 is standard YOLO input)
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        
        try:
            if hasattr(self._model, "predict"):
                # Ultralytics YOLO
                self._model.predict(dummy_input, verbose=False)
            elif hasattr(self._model, "__call__"):
                # Generic callable
                if isinstance(self._model, MockDetector):
                    self._model(dummy_input)
                else:
                    # PyTorch model
                    tensor_input = torch.zeros((1, 3, 640, 640))
                    self._model(tensor_input)
        except Exception as e:
            logger.warning("model_warmup_failed", error=str(e))
        
        logger.info("model_warmup_complete")
    
    def get_model(self) -> Any:
        """
        Get the loaded model for inference.
        
        Returns:
            The loaded detection model.
        
        Raises:
            ModelLoadError: If model has not been loaded.
        """
        if not self._initialized:
            raise ModelLoadError("Model not loaded. Call load_model() first.")
        return self._model
    
    def is_loaded(self) -> bool:
        """
        Check if model has been loaded.
        
        Returns:
            bool: True if model is ready for inference.
        """
        return self._initialized
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the loader (for testing purposes only).
        
        Warning: Do not use in production.
        """
        with cls._lock:
            cls._instance = None
            cls._model = None
            cls._initialized = False


class MockDetector:
    """
    Mock detector for development when weights are unavailable.
    
    Returns synthetic detection results for testing purposes.
    """
    
    def __call__(self, image: np.ndarray) -> list[dict]:
        """
        Simulate plate detection.
        
        Args:
            image: Input image as numpy array.
        
        Returns:
            list: Synthetic detection results.
        """
        h, w = image.shape[:2] if len(image.shape) >= 2 else (640, 640)
        
        return [
            {
                "box": [int(w * 0.2), int(h * 0.6), int(w * 0.8), int(h * 0.8)],
                "confidence": 0.95,
            }
        ]
    
    def predict(self, image: np.ndarray, **kwargs) -> "MockResults":
        """
        Simulate YOLO-style predict method.
        
        Args:
            image: Input image.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            MockResults: Simulated YOLO results.
        """
        return MockResults(image)


class MockResults:
    """Mock YOLO results for development."""
    
    def __init__(self, image: np.ndarray):
        """Initialize with image dimensions."""
        self.shape = image.shape if hasattr(image, "shape") else (640, 640, 3)
        h, w = self.shape[:2]
        self.boxes = MockBoxes(w, h)


class MockBoxes:
    """Mock YOLO bounding boxes."""
    
    def __init__(self, width: int, height: int):
        """Create mock box data."""
        import torch
        
        # Single detection in center of image
        self.xyxy = torch.tensor([
            [width * 0.2, height * 0.6, width * 0.8, height * 0.8]
        ])
        self.conf = torch.tensor([0.95])
        self.cls = torch.tensor([0])  # Class 0 = plate
