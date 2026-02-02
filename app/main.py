"""
FastAPI application entry point.

Configures the application with:
- Lifespan handlers for model loading and database setup
- CORS middleware
- Correlation ID middleware
- Health and readiness probes
- API routes
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, set_correlation_id, setup_logging
from app.infrastructure.db.session import close_db, init_db
from app.infrastructure.ml.model_loader import ModelLoader

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str = "1.0.0"


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    
    status: str
    model_loaded: bool
    database_connected: bool
    ocr_ready: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Startup: Initialize DB, load ML models
    - Shutdown: Clean up connections
    """
    logger.info("application_starting")
    
    # Startup
    try:
        # Initialize database
        await init_db()
        logger.info("database_initialized")
        
        # Load ML models (runs warm-up)
        from fastapi.concurrency import run_in_threadpool
        
        loader = ModelLoader.get_instance()
        await run_in_threadpool(loader.load_model)
        logger.info("ml_models_loaded")
        
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise
    
    logger.info("application_started")
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    await close_db()
    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance.
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Smart Parking Access Control",
        description="ML-based vehicle plate detection and verification system",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Correlation ID middleware
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """Add correlation ID to each request."""
        correlation_id = request.headers.get("X-Correlation-ID")
        set_correlation_id(correlation_id)
        
        response = await call_next(request)
        
        # Add correlation ID to response headers
        from app.core.logging import get_correlation_id
        response.headers["X-Correlation-ID"] = get_correlation_id()
        
        return response
    
    # Health check endpoints
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
    )
    async def health_check() -> HealthResponse:
        """
        Basic liveness probe.
        
        Returns 200 if the application is running.
        """
        return HealthResponse(status="healthy")
    
    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        tags=["health"],
    )
    async def readiness_check() -> ReadinessResponse:
        """
        Readiness probe for Kubernetes/load balancers.
        
        Returns 200 only if all components are ready:
        - ML model loaded
        - Database connected
        - OCR engine ready
        """
        loader = ModelLoader.get_instance()
        model_loaded = loader.is_loaded()
        
        # Check database connection
        database_connected = True
        try:
            from app.infrastructure.db.session import get_session_factory
            session_factory = get_session_factory()
            # Ping database
            async with session_factory() as session:
                await session.execute("SELECT 1")
        except Exception:
            database_connected = False
        
        # Check OCR (EasyOCR lazily loads, so just check import)
        ocr_ready = True
        try:
            import easyocr
        except ImportError:
            ocr_ready = False
        
        all_ready = model_loaded and database_connected and ocr_ready
        
        response = ReadinessResponse(
            status="ready" if all_ready else "not_ready",
            model_loaded=model_loaded,
            database_connected=database_connected,
            ocr_ready=ocr_ready,
        )
        
        if not all_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(),
            )
        
        return response
    
    # Include API routes
    app.include_router(api_router)
    
    return app


# Create app instance
app = create_app()
