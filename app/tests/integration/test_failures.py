"""
Failure scenario tests.

Tests system behavior under failure conditions:
- OCR failures
- Model load failures  
- Database unavailable
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import numpy as np

from app.domain.models import VerificationStatus, Decision
from app.application.plate_verification import PlateVerificationUseCase
from app.infrastructure.ml.detector import DetectionError
from app.infrastructure.ml.ocr import OCRError


class TestDetectionFailures:
    """Tests for plate detection failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_no_plate_detected(self, db_session, mock_ocr_engine, mock_storage):
        """Test handling when no plate is detected in image."""
        # Mock detector that returns empty list
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr_engine,
            storage=mock_storage,
        )
        
        # Create a test image
        image_bytes = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
        
        with patch.object(use_case, '_decode_image', return_value=np.zeros((480, 640, 3))):
            result = await use_case.verify(image_bytes, "CAM001")
        
        assert result.status == VerificationStatus.DETECTION_FAILED
        assert result.action == Decision.ALERT
        assert result.plate_number is None
    
    @pytest.mark.asyncio
    async def test_detector_exception(self, db_session, mock_ocr_engine, mock_storage):
        """Test handling when detector raises exception."""
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = DetectionError("Model failed")
        
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr_engine,
            storage=mock_storage,
        )
        
        image_bytes = b"fake image bytes"
        
        with patch.object(use_case, '_decode_image', return_value=np.zeros((480, 640, 3))):
            result = await use_case.verify(image_bytes, "CAM001")
        
        assert result.status == VerificationStatus.DETECTION_FAILED
        assert result.action == Decision.ALERT


class TestOCRFailures:
    """Tests for OCR failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_ocr_exception(self, db_session, mock_detector, mock_storage):
        """Test handling when OCR raises exception."""
        mock_ocr = MagicMock()
        mock_ocr.extract_text.side_effect = OCRError("OCR failed")
        
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr,
            storage=mock_storage,
        )
        
        image_bytes = b"fake image bytes"
        
        with patch.object(use_case, '_decode_image', return_value=np.zeros((480, 640, 3))):
            result = await use_case.verify(image_bytes, "CAM001")
        
        assert result.status == VerificationStatus.OCR_FAILED
        assert result.action == Decision.ALERT
    
    @pytest.mark.asyncio
    async def test_low_confidence_ocr(self, db_session, mock_detector, mock_storage):
        """Test handling when OCR confidence is below threshold."""
        from app.domain.models import OCRResult
        
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = OCRResult(
            raw_text="MH12AB1234",
            confidence=0.40,  # Below default 0.70 threshold
        )
        
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr,
            storage=mock_storage,
        )
        
        image_bytes = b"fake image bytes"
        
        with patch.object(use_case, '_decode_image', return_value=np.zeros((480, 640, 3))):
            result = await use_case.verify(image_bytes, "CAM001")
        
        assert result.status == VerificationStatus.UNKNOWN
        assert result.action == Decision.ALERT


class TestImageDecodeFailures:
    """Tests for image decode failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_image_bytes(self, db_session, mock_detector, mock_ocr_engine, mock_storage):
        """Test handling when image cannot be decoded."""
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr_engine,
            storage=mock_storage,
        )
        
        # Invalid image bytes
        image_bytes = b"not a valid image"
        
        result = await use_case.verify(image_bytes, "CAM001")
        
        assert result.status == VerificationStatus.DETECTION_FAILED
        assert result.action == Decision.ALERT


class TestModelLoadFailures:
    """Tests for model loading failure scenarios."""
    
    def test_model_not_loaded_error(self):
        """Test that accessing unloaded model raises error."""
        from app.infrastructure.ml.model_loader import ModelLoader, ModelLoadError
        
        # Reset the loader
        ModelLoader.reset()
        
        loader = ModelLoader.get_instance()
        
        with pytest.raises(ModelLoadError, match="not loaded"):
            loader.get_model()
