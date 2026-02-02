"""API routes package."""

from fastapi import APIRouter

from app.api.routes import alerts, plates

# Main API router
api_router = APIRouter(prefix="/api/v1")

# Include route modules
api_router.include_router(plates.router)
api_router.include_router(alerts.router)
