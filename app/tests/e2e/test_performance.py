"""
Performance tests.

Verifies response time requirements with mocked ML.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from app.domain.models import OCRResult, PlateDetectionResult, BoundingBox
from app.application.plate_verification import PlateVerificationUseCase


class TestPerformance:
    """Performance tests for verification pipeline."""
    
    @pytest.mark.asyncio
    async def test_verification_under_2_seconds(self, db_session, mock_detector, mock_ocr_engine, mock_storage):
        """Test that verification completes in under 2 seconds with mocked ML."""
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr_engine,
            storage=mock_storage,
        )
        
        image_bytes = b"test image bytes"
        
        with patch.object(use_case, '_decode_image', return_value=np.zeros((480, 640, 3))):
            start_time = time.time()
            result = await use_case.verify(image_bytes, "CAM001")
            elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"Verification took {elapsed:.2f}s, expected < 2s"
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_verifications(self, db_session, mock_detector, mock_ocr_engine, mock_storage):
        """Test handling multiple concurrent verification requests."""
        use_case = PlateVerificationUseCase(
            session=db_session,
            detector=mock_detector,
            ocr_engine=mock_ocr_engine,
            storage=mock_storage,
        )
        
        image_bytes = b"test image bytes"
        
        async def verify_once():
            with patch.object(use_case, '_decode_image', return_value=np.zeros((480, 640, 3))):
                return await use_case.verify(image_bytes, "CAM001")
        
        # Run 5 concurrent verifications
        start_time = time.time()
        tasks = [verify_once() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        assert len(results) == 5
        assert all(r is not None for r in results)
        # Should complete in reasonable time even with 5 concurrent requests
        assert elapsed < 5.0, f"5 concurrent verifications took {elapsed:.2f}s"
