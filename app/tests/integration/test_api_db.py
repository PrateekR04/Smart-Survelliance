"""
Integration tests for API → Database flow.

Tests the full request flow from API to database.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from app.domain.models import Decision, VerificationStatus
from app.infrastructure.db.repository import (
    AccessLogRepository,
    AlertRepository,
    WhitelistRepository,
)
from app.infrastructure.db.models import WhitelistEntryDB, AccessLogDB


class TestWhitelistRepository:
    """Tests for WhitelistRepository."""
    
    @pytest.mark.asyncio
    async def test_is_whitelisted_true(self, db_session):
        """Test checking a whitelisted plate."""
        # Add a whitelist entry
        entry = WhitelistEntryDB(
            plate_number="MH12AB1234",
            owner_name="Test Owner",
            vehicle_type="car",
            is_active=True,
        )
        db_session.add(entry)
        await db_session.flush()
        
        repo = WhitelistRepository(db_session)
        result = await repo.is_whitelisted("MH12AB1234")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_whitelisted_false(self, db_session):
        """Test checking a non-whitelisted plate."""
        repo = WhitelistRepository(db_session)
        result = await repo.is_whitelisted("XX99ZZ9999")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_whitelisted_inactive(self, db_session):
        """Test that inactive entries are not whitelisted."""
        entry = WhitelistEntryDB(
            plate_number="MH12AB1234",
            owner_name="Test Owner",
            vehicle_type="car",
            is_active=False,  # Soft deleted
        )
        db_session.add(entry)
        await db_session.flush()
        
        repo = WhitelistRepository(db_session)
        result = await repo.is_whitelisted("MH12AB1234")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_soft_delete(self, db_session):
        """Test soft delete functionality."""
        entry = WhitelistEntryDB(
            plate_number="MH12AB1234",
            owner_name="Test Owner",
            is_active=True,
        )
        db_session.add(entry)
        await db_session.flush()
        
        repo = WhitelistRepository(db_session)
        
        # Should be whitelisted initially
        assert await repo.is_whitelisted("MH12AB1234") is True
        
        # Soft delete
        await repo.soft_delete("MH12AB1234")
        await db_session.flush()
        
        # Should no longer be whitelisted
        assert await repo.is_whitelisted("MH12AB1234") is False


class TestAccessLogRepository:
    """Tests for AccessLogRepository."""
    
    @pytest.mark.asyncio
    async def test_create_access_log(self, db_session):
        """Test creating an access log entry."""
        from app.domain.models import AccessLog
        
        repo = AccessLogRepository(db_session)
        
        log = AccessLog(
            plate_number="MH12AB1234",
            camera_id="CAM001",
            timestamp=datetime.utcnow(),
            confidence=0.92,
            status=VerificationStatus.AUTHORIZED,
            decision=Decision.ALLOW,
        )
        
        created = await repo.create(log)
        
        assert created.id is not None
        assert created.plate_number == "MH12AB1234"
    
    @pytest.mark.asyncio
    async def test_list_recent_logs(self, db_session):
        """Test listing recent access logs."""
        repo = AccessLogRepository(db_session)
        
        # Create some logs
        for i in range(3):
            log_entry = AccessLogDB(
                plate_number=f"MH12AB{1234 + i}",
                camera_id="CAM001",
                timestamp=datetime.utcnow(),
                confidence=0.90,
                status="authorized",
                decision="allow",
            )
            db_session.add(log_entry)
        
        await db_session.flush()
        
        logs = await repo.list_recent(limit=10)
        
        assert len(logs) == 3


class TestAlertRepository:
    """Tests for AlertRepository."""
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, db_session):
        """Test acknowledging an alert."""
        from app.infrastructure.db.models import AlertDB, AccessLogDB
        
        # Create access log first
        log = AccessLogDB(
            plate_number="UNKNOWN123",
            camera_id="CAM001",
            timestamp=datetime.utcnow(),
            confidence=0.60,
            status="unknown",
            decision="alert",
        )
        db_session.add(log)
        await db_session.flush()
        
        # Create alert
        alert = AlertDB(
            access_log_id=log.id,
            plate_number="UNKNOWN123",
            camera_id="CAM001",
            timestamp=datetime.utcnow(),
            is_acknowledged=False,
        )
        db_session.add(alert)
        await db_session.flush()
        
        repo = AlertRepository(db_session)
        
        # Acknowledge
        success = await repo.acknowledge(alert.id, "guard_user")
        await db_session.flush()
        
        assert success is True
        
        # Verify acknowledged
        fetched = await repo.get_by_id(alert.id)
        assert fetched.is_acknowledged is True
        assert fetched.acknowledged_by == "guard_user"
