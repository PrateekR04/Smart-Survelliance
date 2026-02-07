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
            
            logger.debug(
                "ocr_input",
                image_shape=image.shape,
            )
            
            # Try on raw image first (often works better)
            results = reader.readtext(image)
            
            logger.debug(
                "ocr_raw_results",
                num_results=len(results),
                results=[(r[1], r[2]) for r in results] if results else [],
            )
            
            # If raw doesn't work well, try preprocessed
            if not results or (results and max(r[2] for r in results) < 0.5):
                processed = self.preprocessor.preprocess(image)
                processed_results = reader.readtext(processed)
                
                logger.debug(
                    "ocr_preprocessed_results",
                    num_results=len(processed_results),
                    results=[(r[1], r[2]) for r in processed_results] if processed_results else [],
                )
                
                # Use whichever has better confidence
                if processed_results:
                    raw_max_conf = max((r[2] for r in results), default=0)
                    processed_max_conf = max((r[2] for r in processed_results), default=0)
                    if processed_max_conf > raw_max_conf:
                        results = processed_results
            
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


class PaddleOCREngine(OCREngine):
    """
    OCR engine using PaddleOCR with PP-OCRv5.
    
    PaddleOCR provides excellent accuracy for license plates
    with support for 100+ languages.
    
    Example:
        engine = PaddleOCREngine()
        result = engine.extract_text(plate_image)
    """
    
    _ocr = None
    _lock = None
    
    def __init__(self, use_angle_cls: bool = True, lang: str = "en"):
        """
        Initialize PaddleOCR engine.
        
        Args:
            use_angle_cls: Enable angle classification for rotated text.
            lang: Language for OCR (default: English).
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.preprocessor = ImagePreprocessor()
        
        # Lazy initialization
        if PaddleOCREngine._lock is None:
            import threading
            PaddleOCREngine._lock = threading.Lock()
    
    def _get_ocr(self):
        """Lazy-load the PaddleOCR instance."""
        if PaddleOCREngine._ocr is None:
            with PaddleOCREngine._lock:
                if PaddleOCREngine._ocr is None:
                    try:
                        from paddleocr import PaddleOCR
                        PaddleOCREngine._ocr = PaddleOCR(
                            use_angle_cls=self.use_angle_cls,
                            lang=self.lang,
                        )
                        logger.info("paddleocr_initialized", lang=self.lang)
                    except ImportError:
                        raise OCRError(
                            "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
                        )
        return PaddleOCREngine._ocr
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text using PaddleOCR.
        
        Args:
            image: Plate image (BGR format).
        
        Returns:
            OCRResult: Extracted text with confidence.
        """
        try:
            ocr = self._get_ocr()
            
            logger.debug(
                "paddleocr_input",
                image_shape=image.shape,
            )
            
            # PaddleOCR predict() takes image directly
            result = ocr.predict(image)
            
            logger.debug(
                "paddleocr_raw_result",
                result_type=type(result).__name__,
            )
            
            # Extract text from result
            texts = []
            confidences = []
            
            # Handle the result structure from PaddleOCR
            for res in result:
                # Access rec_texts and rec_scores attributes
                if hasattr(res, 'rec_texts') and res.rec_texts:
                    texts.extend(res.rec_texts)
                    if hasattr(res, 'rec_scores') and res.rec_scores:
                        confidences.extend(res.rec_scores)
                # Fallback: check for dict-like access
                elif hasattr(res, 'get'):
                    if 'rec_texts' in res:
                        texts.extend(res['rec_texts'])
                    if 'rec_scores' in res:
                        confidences.extend(res['rec_scores'])
            
            logger.debug(
                "paddleocr_results",
                num_texts=len(texts),
                texts=texts,
                confidences=confidences,
            )
            
            if not texts:
                logger.warning("no_text_detected_paddle")
                return OCRResult(raw_text="", confidence=0.0)
            
            # Combine all detected text
            combined_text = "".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
            
            logger.debug(
                "paddleocr_complete",
                text=combined_text,
                confidence=avg_confidence,
            )
            
            return OCRResult(
                raw_text=combined_text,
                confidence=avg_confidence,
            )
            
        except Exception as e:
            logger.error("paddleocr_failed", error=str(e))
            raise OCRError(f"PaddleOCR extraction failed: {e}") from e


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


def get_ocr_engine(
    use_mock: bool = False,
    engine_type: str = "vlm",
) -> OCREngine:
    """
    Factory function to get appropriate OCR engine.
    
    Args:
        use_mock: Whether to use mock engine for testing.
        engine_type: Engine type - "vlm" (PaddleOCR-VL) or "easy" (EasyOCR).
    
    Returns:
        OCREngine: Configured OCR engine instance.
    """
    if use_mock:
        return MockOCREngine()
    
    # Try PaddleOCR first (default)
    if engine_type == "vlm" or engine_type == "paddle":
        try:
            from paddleocr import PaddleOCR
            return PaddleOCREngine()
        except ImportError:
            logger.warning("paddleocr_not_available", falling_back_to="easyocr")
            engine_type = "easy"  # Fall back to EasyOCR
    
    # EasyOCR fallback
    if engine_type == "easy":
        try:
            import easyocr
            return EasyOCREngine()
        except ImportError:
            logger.warning("easyocr_not_available", using_mock=True)
            return MockOCREngine()
    
    # Unknown engine type, default to PaddleOCR
    logger.warning("unknown_engine_type", engine_type=engine_type, using="paddleocr")
    return PaddleOCREngine()


