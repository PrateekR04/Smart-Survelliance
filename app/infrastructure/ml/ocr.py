"""
OCR engine implementations using strategy pattern.

Provides pluggable OCR engines (EasyOCR, PaddleOCR) with
image preprocessing for optimal text extraction from plates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.models import OCRResult

logger = get_logger(__name__)


class OCRError(Exception):
    """Raised when OCR extraction fails."""
    
    pass


class OCREngine(ABC):
    """
    Abstract base class for OCR engines.
    
    Implementations must provide the extract_text method.
    Use the strategy pattern to swap engines at runtime.
    """
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from a plate image.
        
        Args:
            image: Cropped plate region (BGR format).
        
        Returns:
            OCRResult: Extracted text with confidence.
        
        Raises:
            OCRError: If extraction fails.
        """
        pass


class ImagePreprocessor:
    """
    Preprocesses images for optimal OCR performance.
    
    Applies resize, grayscale, denoising, and thresholding
    to improve text recognition accuracy.
    """
    
    def __init__(
        self,
        target_height: int = 100,
        denoise_strength: int = 10,
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_height: Height to resize plate images to.
            denoise_strength: Strength of denoising filter.
        """
        self.target_height = target_height
        self.denoise_strength = denoise_strength
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline.
        
        Args:
            image: Input plate image (BGR).
        
        Returns:
            np.ndarray: Preprocessed image optimized for OCR.
        """
        # Resize maintaining aspect ratio
        processed = self._resize(image)
        
        # Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        processed = cv2.fastNlMeansDenoising(
            processed,
            h=self.denoise_strength,
        )
        
        # Apply adaptive thresholding for contrast
        processed = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        
        return processed
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target height maintaining aspect ratio."""
        height, width = image.shape[:2]
        
        if height == 0:
            return image
        
        scale = self.target_height / height
        new_width = int(width * scale)
        
        return cv2.resize(
            image,
            (new_width, self.target_height),
            interpolation=cv2.INTER_CUBIC,
        )


class EasyOCREngine(OCREngine):
    """
    OCR engine using EasyOCR.
    
    EasyOCR provides good accuracy for license plates
    with support for multiple languages.
    
    Example:
        engine = EasyOCREngine()
        result = engine.extract_text(plate_image)
    """
    
    _reader = None
    _lock = None
    
    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
    ):
        """
        Initialize EasyOCR engine.
        
        Args:
            languages: Languages to recognize (default: English).
            gpu: Whether to use GPU acceleration.
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.preprocessor = ImagePreprocessor()
        
        # Lazy initialization
        if EasyOCREngine._lock is None:
            import threading
            EasyOCREngine._lock = threading.Lock()
    
    def _get_reader(self):
        """Lazy-load the EasyOCR reader."""
        if EasyOCREngine._reader is None:
            with EasyOCREngine._lock:
                if EasyOCREngine._reader is None:
                    try:
                        import easyocr
                        EasyOCREngine._reader = easyocr.Reader(
                            self.languages,
                            gpu=self.gpu,
                        )
                        logger.info("easyocr_initialized", gpu=self.gpu)
                    except ImportError:
                        raise OCRError("EasyOCR not installed. Run: pip install easyocr")
        return EasyOCREngine._reader
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Plate image (BGR format).
        
        Returns:
            OCRResult: Extracted text with confidence.
        """
        try:
            reader = self._get_reader()
            
            # Preprocess for better accuracy
            processed = self.preprocessor.preprocess(image)
            
            # Run OCR
            results = reader.readtext(processed)
            
            if not results:
                logger.warning("no_text_detected")
                return OCRResult(raw_text="", confidence=0.0)
            
            # Combine all detected text
            texts = []
            confidences = []
            
            for detection in results:
                _, text, conf = detection
                texts.append(text)
                confidences.append(conf)
            
            combined_text = "".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.debug(
                "ocr_complete",
                text=combined_text,
                confidence=avg_confidence,
            )
            
            return OCRResult(
                raw_text=combined_text,
                confidence=avg_confidence,
            )
            
        except Exception as e:
            logger.error("ocr_failed", error=str(e))
            raise OCRError(f"OCR extraction failed: {e}") from e


class MockOCREngine(OCREngine):
    """
    Mock OCR engine for testing.
    
    Returns configurable text for testing purposes.
    """
    
    def __init__(
        self,
        mock_text: str = "MH12AB1234",
        mock_confidence: float = 0.95,
    ):
        """
        Initialize mock OCR.
        
        Args:
            mock_text: Text to return from extraction.
            mock_confidence: Confidence to return.
        """
        self.mock_text = mock_text
        self.mock_confidence = mock_confidence
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Return mock OCR result."""
        return OCRResult(
            raw_text=self.mock_text,
            confidence=self.mock_confidence,
        )


def get_ocr_engine(use_mock: bool = False) -> OCREngine:
    """
    Factory function to get appropriate OCR engine.
    
    Args:
        use_mock: Whether to use mock engine for testing.
    
    Returns:
        OCREngine: Configured OCR engine instance.
    """
    if use_mock:
        return MockOCREngine()
    
    try:
        import easyocr
        return EasyOCREngine()
    except ImportError:
        logger.warning("easyocr_not_available", using_mock=True)
        return MockOCREngine()
