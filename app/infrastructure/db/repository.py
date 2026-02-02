"""
Repository pattern implementations for data access.

Repositories abstract database operations and provide
a clean interface for the application layer.
"""

from datetime import datetime
from typing import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import (
    AccessLog,
    Alert,
    Decision,
    VerificationStatus,
    WhitelistEntry,
)
from app.infrastructure.db.models import AccessLogDB, AlertDB, WhitelistEntryDB


class WhitelistRepository:
    """
    Repository for whitelist operations.
    
    Provides methods for looking up and managing whitelisted plates.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session
    
    async def is_whitelisted(self, plate_number: str) -> bool:
        """
        Check if a plate number is whitelisted and active.
        
        Args:
            plate_number: Normalized plate number to check.
        
        Returns:
            bool: True if plate is in active whitelist.
        """
        stmt = select(WhitelistEntryDB).where(
            WhitelistEntryDB.plate_number == plate_number,
            WhitelistEntryDB.is_active == True,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None
    
    async def get_by_plate(self, plate_number: str) -> WhitelistEntry | None:
        """
        Get whitelist entry by plate number.
        
        Args:
            plate_number: Plate number to look up.
        
        Returns:
            WhitelistEntry: Domain model if found, None otherwise.
        """
        stmt = select(WhitelistEntryDB).where(
            WhitelistEntryDB.plate_number == plate_number,
        )
        result = await self._session.execute(stmt)
        db_entry = result.scalar_one_or_none()
        
        if db_entry is None:
            return None
        
        return self._to_domain(db_entry)
    
    async def create(self, entry: WhitelistEntry) -> WhitelistEntry:
        """
        Create a new whitelist entry.
        
        Args:
            entry: Domain model to persist.
        
        Returns:
            WhitelistEntry: Created entry with ID populated.
        """
        db_entry = WhitelistEntryDB(
            plate_number=entry.plate_number,
            owner_name=entry.owner_name,
            vehicle_type=entry.vehicle_type,
            is_active=entry.is_active,
        )
        self._session.add(db_entry)
        await self._session.flush()
        
        return self._to_domain(db_entry)
    
    async def soft_delete(self, plate_number: str) -> bool:
        """
        Soft delete a whitelist entry by marking as inactive.
        
        Args:
            plate_number: Plate to deactivate.
        
        Returns:
            bool: True if entry was found and deactivated.
        """
        stmt = (
            update(WhitelistEntryDB)
            .where(WhitelistEntryDB.plate_number == plate_number)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0
    
    async def list_active(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[WhitelistEntry]:
        """
        List all active whitelist entries.
        
        Args:
            limit: Maximum entries to return.
            offset: Pagination offset.
        
        Returns:
            list: Active whitelist entries.
        """
        stmt = (
            select(WhitelistEntryDB)
            .where(WhitelistEntryDB.is_active == True)
            .order_by(WhitelistEntryDB.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [self._to_domain(entry) for entry in result.scalars().all()]
    
    def _to_domain(self, db_entry: WhitelistEntryDB) -> WhitelistEntry:
        """Convert database model to domain model."""
        return WhitelistEntry(
            plate_number=db_entry.plate_number,
            owner_name=db_entry.owner_name,
            vehicle_type=db_entry.vehicle_type,
            is_active=db_entry.is_active,
            created_at=db_entry.created_at,
            updated_at=db_entry.updated_at,
        )


class AccessLogRepository:
    """
    Repository for access log operations.
    
    Provides methods for creating and querying access logs.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session
    
    async def create(self, log: AccessLog) -> AccessLog:
        """
        Create a new access log entry.
        
        Args:
            log: Domain model to persist.
        
        Returns:
            AccessLog: Created entry with ID populated.
        """
        db_log = AccessLogDB(
            plate_number=log.plate_number,
            camera_id=log.camera_id,
            timestamp=log.timestamp,
            confidence=log.confidence,
            status=log.status.value,
            decision=log.decision.value,
            image_path=log.image_path,
        )
        self._session.add(db_log)
        await self._session.flush()
        
        log.id = db_log.id
        return log
    
    async def get_by_id(self, log_id: int) -> AccessLog | None:
        """
        Get access log by ID.
        
        Args:
            log_id: Log entry ID.
        
        Returns:
            AccessLog: Domain model if found, None otherwise.
        """
        stmt = select(AccessLogDB).where(AccessLogDB.id == log_id)
        result = await self._session.execute(stmt)
        db_log = result.scalar_one_or_none()
        
        if db_log is None:
            return None
        
        return self._to_domain(db_log)
    
    async def list_recent(
        self,
        limit: int = 50,
        plate_number: str | None = None,
        camera_id: str | None = None,
    ) -> Sequence[AccessLog]:
        """
        List recent access logs with optional filters.
        
        Args:
            limit: Maximum entries to return.
            plate_number: Optional plate filter.
            camera_id: Optional camera filter.
        
        Returns:
            list: Recent access log entries.
        """
        stmt = select(AccessLogDB).order_by(AccessLogDB.timestamp.desc()).limit(limit)
        
        if plate_number:
            stmt = stmt.where(AccessLogDB.plate_number == plate_number)
        if camera_id:
            stmt = stmt.where(AccessLogDB.camera_id == camera_id)
        
        result = await self._session.execute(stmt)
        return [self._to_domain(log) for log in result.scalars().all()]
    
    def _to_domain(self, db_log: AccessLogDB) -> AccessLog:
        """Convert database model to domain model."""
        return AccessLog(
            id=db_log.id,
            plate_number=db_log.plate_number,
            camera_id=db_log.camera_id,
            timestamp=db_log.timestamp,
            confidence=db_log.confidence,
            status=VerificationStatus(db_log.status),
            decision=Decision(db_log.decision),
            image_path=db_log.image_path,
        )


class AlertRepository:
    """
    Repository for alert operations.
    
    Provides methods for creating, acknowledging, and querying alerts.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session
    
    async def create(self, alert: Alert) -> Alert:
        """
        Create a new alert.
        
        Args:
            alert: Domain model to persist.
        
        Returns:
            Alert: Created alert with ID populated.
        """
        db_alert = AlertDB(
            access_log_id=alert.access_log_id,
            plate_number=alert.plate_number,
            camera_id=alert.camera_id,
            timestamp=alert.timestamp,
            image_path=alert.image_path,
            is_acknowledged=alert.is_acknowledged,
        )
        self._session.add(db_alert)
        await self._session.flush()
        
        alert.id = db_alert.id
        return alert
    
    async def acknowledge(
        self,
        alert_id: int,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge.
            acknowledged_by: Username of the guard.
        
        Returns:
            bool: True if alert was found and acknowledged.
        """
        stmt = (
            update(AlertDB)
            .where(AlertDB.id == alert_id, AlertDB.is_acknowledged == False)
            .values(
                is_acknowledged=True,
                acknowledged_by=acknowledged_by,
                acknowledged_at=datetime.utcnow(),
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0
    
    async def get_pending(
        self,
        limit: int = 50,
        camera_id: str | None = None,
    ) -> Sequence[Alert]:
        """
        Get pending (unacknowledged) alerts.
        
        Args:
            limit: Maximum alerts to return.
            camera_id: Optional camera filter.
        
        Returns:
            list: Pending alerts ordered by newest first.
        """
        stmt = (
            select(AlertDB)
            .where(AlertDB.is_acknowledged == False)
            .order_by(AlertDB.timestamp.desc())
            .limit(limit)
        )
        
        if camera_id:
            stmt = stmt.where(AlertDB.camera_id == camera_id)
        
        result = await self._session.execute(stmt)
        return [self._to_domain(alert) for alert in result.scalars().all()]
    
    async def get_by_id(self, alert_id: int) -> Alert | None:
        """
        Get alert by ID.
        
        Args:
            alert_id: Alert ID.
        
        Returns:
            Alert: Domain model if found, None otherwise.
        """
        stmt = select(AlertDB).where(AlertDB.id == alert_id)
        result = await self._session.execute(stmt)
        db_alert = result.scalar_one_or_none()
        
        if db_alert is None:
            return None
        
        return self._to_domain(db_alert)
    
    def _to_domain(self, db_alert: AlertDB) -> Alert:
        """Convert database model to domain model."""
        return Alert(
            id=db_alert.id,
            access_log_id=db_alert.access_log_id,
            plate_number=db_alert.plate_number,
            camera_id=db_alert.camera_id,
            timestamp=db_alert.timestamp,
            image_path=db_alert.image_path,
            is_acknowledged=db_alert.is_acknowledged,
            acknowledged_by=db_alert.acknowledged_by,
            acknowledged_at=db_alert.acknowledged_at,
        )
