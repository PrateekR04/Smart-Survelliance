"""
Async SQLAlchemy database session configuration.

Provides async engine and session factory for MySQL using aiomysql.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

# Create async engine with connection pooling
_engine = None
_session_factory = None


def get_engine():
    """
    Get or create the async database engine.
    
    Uses connection pooling for production and NullPool for testing.
    
    Returns:
        AsyncEngine: SQLAlchemy async engine instance.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session factory.
    
    Returns:
        async_sessionmaker: Factory for creating async sessions.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session.
    
    Usage with FastAPI:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    
    Yields:
        AsyncSession: Database session that auto-closes on exit.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database by creating all tables.
    
    Should be called during application startup.
    """
    from app.infrastructure.db.models import Base
    
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """
    Close database connections.
    
    Should be called during application shutdown.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def create_test_engine(database_url: str):
    """
    Create a test engine with NullPool for isolated testing.
    
    Args:
        database_url: Test database connection string.
    
    Returns:
        AsyncEngine: Test engine instance.
    """
    return create_async_engine(
        database_url,
        echo=True,
        poolclass=NullPool,
    )
