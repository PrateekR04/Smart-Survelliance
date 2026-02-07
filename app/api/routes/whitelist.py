"""
Whitelist management API routes.

Provides CRUD endpoints for managing authorized vehicle plates.
Admin-only access.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ApiKeyAuth, RateLimited
from app.core.logging import get_logger
from app.infrastructure.db.session import get_session
from app.infrastructure.db.models import WhitelistEntryDB

logger = get_logger(__name__)

router = APIRouter(prefix="/whitelist", tags=["whitelist"])


class WhitelistEntry(BaseModel):
    """Response model for a whitelist entry."""
    
    id: int
    plate_number: str
    owner_name: str
    vehicle_type: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class WhitelistListResponse(BaseModel):
    """Response for listing whitelist entries."""
    
    entries: list[WhitelistEntry]
    count: int


class WhitelistCreateRequest(BaseModel):
    """Request to add a new whitelist entry."""
    
    plate_number: str = Field(..., min_length=6, max_length=20)
    owner_name: str = Field(..., min_length=1, max_length=100)
    vehicle_type: str = Field(default="car", max_length=50)


class WhitelistUpdateRequest(BaseModel):
    """Request to update a whitelist entry."""
    
    owner_name: str | None = Field(None, max_length=100)
    vehicle_type: str | None = Field(None, max_length=50)
    is_active: bool | None = None


@router.get(
    "",
    response_model=WhitelistListResponse,
    summary="List whitelist entries",
    description="Get all whitelist entries. Admin only.",
)
async def list_whitelist(
    _: ApiKeyAuth,
    __: RateLimited,
    db: AsyncSession = Depends(get_session),
    is_active: bool | None = None,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
) -> WhitelistListResponse:
    """List all whitelist entries with optional filtering."""
    query = select(WhitelistEntryDB)
    
    if is_active is not None:
        query = query.where(WhitelistEntryDB.is_active == is_active)
    
    query = query.order_by(WhitelistEntryDB.created_at.desc())
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    entries = result.scalars().all()
    
    return WhitelistListResponse(
        entries=[
            WhitelistEntry(
                id=e.id,
                plate_number=e.plate_number,
                owner_name=e.owner_name,
                vehicle_type=e.vehicle_type,
                is_active=e.is_active,
                created_at=e.created_at,
                updated_at=e.updated_at,
            )
            for e in entries
        ],
        count=len(entries),
    )


@router.post(
    "",
    response_model=WhitelistEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Add to whitelist",
    description="Add a new plate to the whitelist. Admin only.",
)
async def create_whitelist_entry(
    request: WhitelistCreateRequest,
    _: ApiKeyAuth,
    db: AsyncSession = Depends(get_session),
) -> WhitelistEntry:
    """Add a new plate number to the whitelist."""
    # Check for duplicate
    existing = await db.execute(
        select(WhitelistEntryDB).where(
            WhitelistEntryDB.plate_number == request.plate_number.upper()
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Plate number already exists in whitelist",
        )
    
    # Create entry
    entry = WhitelistEntryDB(
        plate_number=request.plate_number.upper(),
        owner_name=request.owner_name,
        vehicle_type=request.vehicle_type,
    )
    db.add(entry)
    await db.flush()
    await db.refresh(entry)
    
    logger.info(
        "whitelist_entry_created",
        plate=entry.plate_number,
        owner=entry.owner_name,
    )
    
    return WhitelistEntry(
        id=entry.id,
        plate_number=entry.plate_number,
        owner_name=entry.owner_name,
        vehicle_type=entry.vehicle_type,
        is_active=entry.is_active,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.put(
    "/{entry_id}",
    response_model=WhitelistEntry,
    summary="Update whitelist entry",
    description="Update an existing whitelist entry. Admin only.",
)
async def update_whitelist_entry(
    entry_id: Annotated[int, Path(description="Whitelist entry ID")],
    request: WhitelistUpdateRequest,
    _: ApiKeyAuth,
    db: AsyncSession = Depends(get_session),
) -> WhitelistEntry:
    """Update an existing whitelist entry."""
    result = await db.execute(
        select(WhitelistEntryDB).where(WhitelistEntryDB.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Whitelist entry not found",
        )
    
    if request.owner_name is not None:
        entry.owner_name = request.owner_name
    if request.vehicle_type is not None:
        entry.vehicle_type = request.vehicle_type
    if request.is_active is not None:
        entry.is_active = request.is_active
    
    await db.flush()
    await db.refresh(entry)
    
    logger.info(
        "whitelist_entry_updated",
        entry_id=entry_id,
    )
    
    return WhitelistEntry(
        id=entry.id,
        plate_number=entry.plate_number,
        owner_name=entry.owner_name,
        vehicle_type=entry.vehicle_type,
        is_active=entry.is_active,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.delete(
    "/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete whitelist entry",
    description="Remove a plate from the whitelist. Admin only.",
)
async def delete_whitelist_entry(
    entry_id: Annotated[int, Path(description="Whitelist entry ID")],
    _: ApiKeyAuth,
    db: AsyncSession = Depends(get_session),
) -> None:
    """Delete a whitelist entry."""
    result = await db.execute(
        select(WhitelistEntryDB).where(WhitelistEntryDB.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Whitelist entry not found",
        )
    
    await db.delete(entry)
    
    logger.info(
        "whitelist_entry_deleted",
        entry_id=entry_id,
        plate=entry.plate_number,
    )
