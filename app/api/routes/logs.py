"""
Access logs API routes.

Provides read-only endpoints for viewing access logs.
Admin-only access.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ApiKeyAuth, RateLimited
from app.core.logging import get_logger
from app.infrastructure.db.session import get_session
from app.infrastructure.db.models import AccessLogDB

logger = get_logger(__name__)

router = APIRouter(prefix="/logs", tags=["logs"])


class AccessLogEntry(BaseModel):
    """Response model for an access log entry."""
    
    id: int
    plate_number: str | None
    camera_id: str
    timestamp: datetime
    confidence: float
    status: str
    decision: str
    image_path: str | None
    created_at: datetime


class AccessLogListResponse(BaseModel):
    """Response for listing access logs."""
    
    logs: list[AccessLogEntry]
    count: int
    total: int


@router.get(
    "",
    response_model=AccessLogListResponse,
    summary="List access logs",
    description="Get access logs with filtering. Admin only.",
)
async def list_logs(
    _: ApiKeyAuth,
    __: RateLimited,
    db: AsyncSession = Depends(get_session),
    status_filter: str | None = Query(None, alias="status"),
    camera_id: str | None = None,
    plate_number: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
) -> AccessLogListResponse:
    """List access logs with optional filtering."""
    query = select(AccessLogDB)
    count_query = select(func.count(AccessLogDB.id))
    
    if status_filter:
        query = query.where(AccessLogDB.status == status_filter)
        count_query = count_query.where(AccessLogDB.status == status_filter)
    
    if camera_id:
        query = query.where(AccessLogDB.camera_id == camera_id)
        count_query = count_query.where(AccessLogDB.camera_id == camera_id)
    
    if plate_number:
        query = query.where(AccessLogDB.plate_number.ilike(f"%{plate_number}%"))
        count_query = count_query.where(AccessLogDB.plate_number.ilike(f"%{plate_number}%"))
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Get paginated results
    query = query.order_by(AccessLogDB.timestamp.desc())
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return AccessLogListResponse(
        logs=[
            AccessLogEntry(
                id=log.id,
                plate_number=log.plate_number,
                camera_id=log.camera_id,
                timestamp=log.timestamp,
                confidence=log.confidence,
                status=log.status,
                decision=log.decision,
                image_path=log.image_path,
                created_at=log.created_at,
            )
            for log in logs
        ],
        count=len(logs),
        total=total,
    )


@router.get(
    "/{log_id}",
    response_model=AccessLogEntry,
    summary="Get log details",
    description="Get details of a specific access log. Admin only.",
)
async def get_log(
    log_id: Annotated[int, Path(description="Access log ID")],
    _: ApiKeyAuth,
    db: AsyncSession = Depends(get_session),
) -> AccessLogEntry:
    """Get details of a specific access log entry."""
    result = await db.execute(
        select(AccessLogDB).where(AccessLogDB.id == log_id)
    )
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access log not found",
        )
    
    return AccessLogEntry(
        id=log.id,
        plate_number=log.plate_number,
        camera_id=log.camera_id,
        timestamp=log.timestamp,
        confidence=log.confidence,
        status=log.status,
        decision=log.decision,
        image_path=log.image_path,
        created_at=log.created_at,
    )
