"""
SQLAlchemy ORM models for database tables.

These models define the database schema and provide
persistence for domain entities.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain.models import Decision, VerificationStatus


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    
    pass


class WhitelistEntryDB(Base):
    """
    Database model for whitelisted vehicles.
    
    Stores authorized plate numbers with owner information.
    Supports soft delete via is_active flag.
    """
    
    __tablename__ = "whitelist_entries"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_number: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
        index=True,
    )
    owner_name: Mapped[str] = mapped_column(String(100), nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String(50), default="car")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Composite index for active plate lookups
    __table_args__ = (
        Index("ix_whitelist_active_plate", "plate_number", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<WhitelistEntry(plate={self.plate_number}, owner={self.owner_name})>"


class AccessLogDB(Base):
    """
    Database model for access log entries.
    
    Records every plate verification attempt for auditing.
    """
    
    __tablename__ = "access_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_number: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    decision: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationship to alerts
    alerts: Mapped[list["AlertDB"]] = relationship(
        "AlertDB",
        back_populates="access_log",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        Index("ix_access_logs_time_status", "timestamp", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<AccessLog(plate={self.plate_number}, status={self.status})>"


class AlertDB(Base):
    """
    Database model for unauthorized access alerts.
    
    Created when a plate is unauthorized, unknown, or detection fails.
    Guards acknowledge alerts via the web app.
    """
    
    __tablename__ = "alerts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    access_log_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("access_logs.id", ondelete="CASCADE"),
        nullable=False,
    )
    plate_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationship to access log
    access_log: Mapped["AccessLogDB"] = relationship(
        "AccessLogDB",
        back_populates="alerts",
    )
    
    __table_args__ = (
        Index("ix_alerts_pending", "is_acknowledged", "timestamp"),
    )
    
    def __repr__(self) -> str:
        return f"<Alert(plate={self.plate_number}, ack={self.is_acknowledged})>"
