"""
Database connection management for ARGO data.

This module provides async database connections, session management,
and connection pooling optimized for high-performance LLM queries.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncEngine, 
    AsyncSession, 
    async_sessionmaker
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from floatchat.core.config import settings
from floatchat.core.logging import get_logger

logger = get_logger(__name__)

# Global engine instance
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def get_database_url() -> str:
    """
    Build database URL from configuration.
    
    Returns:
        Database URL with proper async driver
    """
    if settings.testing:
        # Use in-memory SQLite for testing
        return "sqlite+aiosqlite:///:memory:"
    
    # Build PostgreSQL URL for production/development
    user = settings.database.postgres_user
    password = settings.database.postgres_password
    server = settings.database.postgres_server
    port = settings.database.postgres_port
    database = settings.database.postgres_db
    
    return f"postgresql+asyncpg://{user}:{password}@{server}:{port}/{database}"


def create_engine(database_url: Optional[str] = None) -> AsyncEngine:
    """
    Create async database engine with optimized settings.
    
    Args:
        database_url: Override default database URL
        
    Returns:
        Configured async SQLAlchemy engine
    """
    url = database_url or get_database_url()
    
    # Engine configuration
    engine_kwargs = {
        "echo": settings.debug and not settings.testing,
        "future": True,
    }
    
    if "postgresql" in url:
        # PostgreSQL-specific optimizations
        engine_kwargs.update({
            "pool_size": settings.database.database_pool_size,
            "max_overflow": settings.database.database_max_overflow,
            "pool_timeout": 30,
            "pool_recycle": 3600,  # Recycle connections every hour
            "pool_pre_ping": True,  # Validate connections before use
        })
    elif "sqlite" in url:
        # SQLite-specific settings for testing
        engine_kwargs.update({
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 20,
            },
        })
    
    engine = create_async_engine(url, **engine_kwargs)
    
    # Add connection event listeners
    if "postgresql" in url:
        @event.listens_for(engine.sync_engine, "connect")
        def configure_postgres_connection(dbapi_connection, connection_record):
            """Configure PostgreSQL connection settings."""
            with dbapi_connection.cursor() as cursor:
                # Set timezone to UTC
                cursor.execute("SET timezone TO 'UTC'")
                # Optimize for read-heavy workloads
                cursor.execute("SET default_transaction_isolation TO 'read committed'")
                # Enable JIT for complex queries
                cursor.execute("SET jit TO on")
    
    logger.info(f"Created database engine with URL: {url.split('@')[0]}@***")
    return engine


async def init_database() -> None:
    """
    Initialize database connection and create tables if needed.
    """
    global _engine, _session_factory
    
    _engine = create_engine()
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Test connection
    async with _engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    
    logger.info("Database connection initialized successfully")


async def close_database() -> None:
    """
    Close database connections gracefully.
    """
    global _engine
    
    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed")


def get_engine() -> AsyncEngine:
    """
    Get the global database engine.
    
    Returns:
        Database engine instance
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session with automatic cleanup.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    
    Yields:
        Async database session
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class DatabaseManager:
    """
    Database manager for handling connections and migrations.
    """
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
    
    async def initialize(self, database_url: Optional[str] = None) -> None:
        """Initialize database connections."""
        self.engine = create_engine(database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with self.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        logger.info("Database manager initialized")
    
    async def close(self) -> None:
        """Close all connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database manager connections closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get session from manager."""
        if not self.session_factory:
            raise RuntimeError("Database manager not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_raw_sql(self, sql: str, **params) -> None:
        """
        Execute raw SQL statement.
        
        Args:
            sql: SQL statement to execute
            **params: Parameters for the SQL statement
        """
        async with self.session() as session:
            await session.execute(text(sql), params)
    
    async def migrate_database(self) -> None:
        """
        Run database migrations.
        This is a placeholder - actual migrations would use Alembic.
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        # Import models to ensure they're registered
        from floatchat.domain.entities.argo_entities import Base
        
        # Create tables (in production, use Alembic migrations)
        if settings.is_development or settings.testing:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")
        else:
            logger.info("Production environment - skipping table creation")
    
    async def health_check(self) -> dict:
        """
        Check database health and return status.
        
        Returns:
            Health check results
        """
        if not self.engine:
            return {
                "status": "unhealthy",
                "message": "Database not initialized"
            }
        
        try:
            async with self.session() as session:
                start_time = asyncio.get_event_loop().time()
                await session.execute(text("SELECT 1"))
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "message": "Database connection successful"
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Database error: {str(e)}"
            }


# Global database manager instance
db_manager = DatabaseManager()


async def create_database_if_not_exists() -> None:
    """
    Create database if it doesn't exist (PostgreSQL only).
    """
    if settings.testing or "sqlite" in get_database_url():
        return
    
    # Connect to postgres database to create main database
    admin_url = get_database_url().replace(
        f"/{settings.database.postgres_db}",
        "/postgres"
    )
    
    admin_engine = create_async_engine(admin_url)
    
    try:
        async with admin_engine.connect() as conn:
            # Set autocommit to create database
            await conn.execution_options(autocommit=True)
            
            # Check if database exists
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": settings.database.postgres_db}
            )
            
            if not result.fetchone():
                await conn.execute(
                    text(f"CREATE DATABASE {settings.database.postgres_db}")
                )
                logger.info(f"Created database: {settings.database.postgres_db}")
            else:
                logger.info(f"Database already exists: {settings.database.postgres_db}")
                
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise
    finally:
        await admin_engine.dispose()


# Convenience functions for backward compatibility
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Alias for get_session()."""
    async with get_session() as session:
        yield session