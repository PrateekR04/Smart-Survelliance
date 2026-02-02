"""
Pytest configuration and fixtures.

Provides shared fixtures for testing including:
- Mock database sessions
- Mock ML components
- Test client
"""

import asyncio
from datetime import datetime
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.domain.models import OCRResult, PlateDetectionResult, BoundingBox
from app.infrastructure.db.models import Base


# Use SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def db_session() -> AsyncIterator[AsyncSession]:
    """Create a test database session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
def mock_detector() -> MagicMock:
    """Create mock plate detector."""
    detector = MagicMock()
    detector.detect.return_value = [
        PlateDetectionResult(
            bounding_box=BoundingBox(100, 150, 300, 200),
            confidence=0.95,
            image_width=640,
            image_height=480,
        )
    ]
    return detector


@pytest.fixture
def mock_ocr_engine() -> MagicMock:
    """Create mock OCR engine."""
    engine = MagicMock()
    engine.extract_text.return_value = OCRResult(
        raw_text="MH 12 AB 1234",
        confidence=0.92,
    )
    return engine


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create mock image storage."""
    storage = MagicMock()
    storage.save.return_value = "2026-02-01/test_image.jpg"
    return storage


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    # Create a simple test image
    import cv2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white rectangle to simulate a plate
    cv2.rectangle(img, (100, 150), (300, 200), (255, 255, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def test_client() -> TestClient:
    """Create test client with mocked dependencies."""
    # Patch model loader before importing app
    with patch("app.infrastructure.ml.model_loader.ModelLoader") as mock_loader:
        mock_instance = MagicMock()
        mock_instance.is_loaded.return_value = True
        mock_instance.get_model.return_value = MagicMock()
        mock_loader.get_instance.return_value = mock_instance
        
        from app.main import app
        
        with TestClient(app) as client:
            yield client


@pytest.fixture
async def async_client(test_client) -> AsyncIterator[AsyncClient]:
    """Create async test client."""
    from app.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
