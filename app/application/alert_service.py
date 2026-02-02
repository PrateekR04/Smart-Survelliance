"""
Alert service for managing unauthorized access alerts.

Handles alert creation, acknowledgment, and retrieval
for the guard web application.
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.domain.models import AccessLog, Alert, Decision, VerificationStatus
from app.infrastructure.db.repository import AlertRepository

logger = get_logger(__name__)


class AlertService:
    """
    Service for managing access alerts.
    
    Creates alerts for unauthorized or unknown plates
    and handles guard acknowledgment workflow.
    
    Example:
        service = AlertService(session)
        await service.create_from_access_log(log)
        alerts = await service.get_pending_alerts()
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize alert service.
        
        Args:
            session: Database session.
        """
        self._session = session
        self._alert_repo = AlertRepository(session)
    
    async def create_from_access_log(self, access_log: AccessLog) -> Alert | None:
        """
        Create an alert from an access log if needed.
        
        Only creates alerts for non-authorized access attempts.
        
        Args:
            access_log: The access log entry.
        
        Returns:
            Alert: Created alert, or None if not needed.
        """
        # Only create alerts for non-authorized access
        if access_log.status == VerificationStatus.AUTHORIZED:
            return None
        
        if access_log.id is None:
            logger.error("cannot_create_alert", reason="access_log_id_missing")
            return None
        
        alert = Alert(
            access_log_id=access_log.id,
            plate_number=access_log.plate_number,
            camera_id=access_log.camera_id,
            timestamp=access_log.timestamp,
            image_path=access_log.image_path,
        )
        
        created_alert = await self._alert_repo.create(alert)
        
        logger.info(
            "alert_created",
            alert_id=created_alert.id,
            plate=access_log.plate_number,
            camera=access_log.camera_id,
        )
        
        return created_alert
    
    async def acknowledge(
        self,
        alert_id: int,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge.
            acknowledged_by: Username of acknowledging guard.
        
        Returns:
            bool: True if successfully acknowledged.
        """
        success = await self._alert_repo.acknowledge(alert_id, acknowledged_by)
        
        if success:
            logger.info(
                "alert_acknowledged",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by,
            )
        else:
            logger.warning(
                "alert_acknowledge_failed",
                alert_id=alert_id,
                reason="not_found_or_already_acknowledged",
            )
        
        return success
    
    async def get_pending_alerts(
        self,
        limit: int = 50,
        camera_id: str | None = None,
    ) -> list[Alert]:
        """
        Get all pending (unacknowledged) alerts.
        
        Args:
            limit: Maximum alerts to return.
            camera_id: Optional filter by camera.
        
        Returns:
            list: Pending alerts, newest first.
        """
        alerts = await self._alert_repo.get_pending(limit, camera_id)
        return list(alerts)
    
    async def get_alert(self, alert_id: int) -> Alert | None:
        """
        Get a specific alert by ID.
        
        Args:
            alert_id: Alert ID.
        
        Returns:
            Alert: The alert if found.
        """
        return await self._alert_repo.get_by_id(alert_id)
