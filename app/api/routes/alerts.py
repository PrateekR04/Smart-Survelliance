"""
Alert management API routes.

Provides endpoints for guards to view and acknowledge
security alerts from unauthorized access attempts.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, status
from pydantic import BaseModel, Field

from app.api.deps import AlertSvc, CurrentUser, RateLimited
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertResponse(BaseModel):
    """Response model for a single alert."""
    
    id: int = Field(description="Alert ID")
    plate_number: str | None = Field(description="Detected plate number")
    camera_id: str = Field(description="Camera that detected the plate")
    timestamp: datetime = Field(description="When the alert was created")
    image_path: str | None = Field(description="Path to captured image")
    is_acknowledged: bool = Field(description="Whether alert has been acknowledged")
    acknowledged_by: str | None = Field(description="Guard who acknowledged")
    acknowledged_at: datetime | None = Field(description="When alert was acknowledged")


class AlertListResponse(BaseModel):
    """Response model for alert list."""
    
    alerts: list[AlertResponse] = Field(description="List of alerts")
    count: int = Field(description="Number of alerts returned")


class AcknowledgeRequest(BaseModel):
    """Request body for acknowledging an alert."""
    
    # No body required - user comes from auth


class AcknowledgeResponse(BaseModel):
    """Response for acknowledge action."""
    
    success: bool = Field(description="Whether acknowledgment succeeded")
    message: str = Field(description="Status message")


@router.get(
    "",
    response_model=AlertListResponse,
    summary="List pending alerts",
    description="Get all unacknowledged alerts for the guard dashboard.",
)
async def list_alerts(
    alert_service: AlertSvc,
    current_user: CurrentUser,
    _: RateLimited,
    camera_id: str | None = None,
    limit: int = 50,
) -> AlertListResponse:
    """
    List pending (unacknowledged) alerts.
    
    **Authentication**: Requires HTTP Basic Auth (guard credentials).
    
    Returns alerts ordered by timestamp, newest first.
    """
    logger.info(
        "alerts_requested",
        user=current_user.username,
        camera_id=camera_id,
    )
    
    alerts = await alert_service.get_pending_alerts(limit=limit, camera_id=camera_id)
    
    alert_responses = [
        AlertResponse(
            id=alert.id,
            plate_number=alert.plate_number,
            camera_id=alert.camera_id,
            timestamp=alert.timestamp,
            image_path=alert.image_path,
            is_acknowledged=alert.is_acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
        )
        for alert in alerts
        if alert.id is not None
    ]
    
    return AlertListResponse(
        alerts=alert_responses,
        count=len(alert_responses),
    )


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get alert details",
    description="Get details of a specific alert.",
)
async def get_alert(
    alert_id: Annotated[int, Path(description="Alert ID")],
    alert_service: AlertSvc,
    current_user: CurrentUser,
) -> AlertResponse:
    """
    Get details of a specific alert.
    
    **Authentication**: Requires HTTP Basic Auth.
    """
    alert = await alert_service.get_alert(alert_id)
    
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )
    
    return AlertResponse(
        id=alert.id,
        plate_number=alert.plate_number,
        camera_id=alert.camera_id,
        timestamp=alert.timestamp,
        image_path=alert.image_path,
        is_acknowledged=alert.is_acknowledged,
        acknowledged_by=alert.acknowledged_by,
        acknowledged_at=alert.acknowledged_at,
    )


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AcknowledgeResponse,
    summary="Acknowledge alert",
    description="Mark an alert as acknowledged by a guard.",
)
async def acknowledge_alert(
    alert_id: Annotated[int, Path(description="Alert ID to acknowledge")],
    alert_service: AlertSvc,
    current_user: CurrentUser,
) -> AcknowledgeResponse:
    """
    Acknowledge a security alert.
    
    **Authentication**: Requires HTTP Basic Auth.
    
    Guards acknowledge alerts after reviewing them on the dashboard.
    """
    success = await alert_service.acknowledge(
        alert_id=alert_id,
        acknowledged_by=current_user.username,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found or already acknowledged",
        )
    
    logger.info(
        "alert_acknowledged_via_api",
        alert_id=alert_id,
        user=current_user.username,
    )
    
    return AcknowledgeResponse(
        success=True,
        message=f"Alert {alert_id} acknowledged by {current_user.username}",
    )
