"""
FastAPI dependencies for dependency injection.

Provides database sessions, use case instances, and
authentication dependencies for route handlers.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.alert_service import AlertService
from app.application.idempotency import IdempotencyService, get_idempotency_service
from app.application.plate_verification import PlateVerificationUseCase
from app.core.security import User, check_rate_limit, verify_api_key, verify_basic_auth
from app.infrastructure.db.session import get_session


# Type aliases for cleaner route signatures
Session = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(verify_basic_auth)]
ApiKeyAuth = Annotated[None, Depends(verify_api_key)]
RateLimited = Annotated[None, Depends(check_rate_limit)]


async def get_verification_use_case(
    session: Session,
) -> PlateVerificationUseCase:
    """
    Dependency to get plate verification use case.
    
    Args:
        session: Database session.
    
    Returns:
        PlateVerificationUseCase: Configured use case instance.
    """
    return PlateVerificationUseCase(session)


async def get_alert_service(session: Session) -> AlertService:
    """
    Dependency to get alert service.
    
    Args:
        session: Database session.
    
    Returns:
        AlertService: Configured service instance.
    """
    return AlertService(session)


def get_idempotency() -> IdempotencyService:
    """
    Dependency to get idempotency service.
    
    Returns:
        IdempotencyService: Global idempotency service.
    """
    return get_idempotency_service()


# Type aliases for use case dependencies
VerificationUseCase = Annotated[PlateVerificationUseCase, Depends(get_verification_use_case)]
AlertSvc = Annotated[AlertService, Depends(get_alert_service)]
Idempotency = Annotated[IdempotencyService, Depends(get_idempotency)]
