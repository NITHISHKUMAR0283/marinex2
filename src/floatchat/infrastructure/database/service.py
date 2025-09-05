"""
Database service layer for ARGO data operations.

This module provides high-level database operations, migration management,
and service layer abstractions for the FloatChat application.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from floatchat.infrastructure.database.connection import (
    get_session, db_manager, create_database_if_not_exists
)
from floatchat.infrastructure.repositories.argo_repository import ArgoRepository
from floatchat.core.config import settings
from floatchat.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseService:
    """High-level database service for ARGO data operations."""
    
    def __init__(self):
        self.migration_path = Path(__file__).parent.parent.parent.parent.parent / "migrations"
    
    async def initialize_database(self) -> None:
        """Initialize database with tables and seed data."""
        try:
            # Create database if it doesn't exist
            await create_database_if_not_exists()
            
            # Initialize connection manager
            await db_manager.initialize()
            
            # Run migrations
            await self.run_migrations()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def run_migrations(self) -> None:
        """Run all pending database migrations."""
        try:
            # Get list of migration files
            migration_files = sorted([
                f for f in self.migration_path.glob("*.sql")
                if f.name.endswith('.sql')
            ])
            
            async with get_session() as session:
                # Create migrations tracking table if it doesn't exist
                await self._create_migrations_table(session)
                
                # Get applied migrations
                applied_migrations = await self._get_applied_migrations(session)
                
                # Apply new migrations
                for migration_file in migration_files:
                    migration_name = migration_file.stem
                    
                    if migration_name not in applied_migrations:
                        logger.info(f"Applying migration: {migration_name}")
                        
                        # Read and execute migration SQL
                        migration_sql = migration_file.read_text(encoding='utf-8')
                        
                        # Execute migration in chunks (split by semicolon)
                        statements = [
                            stmt.strip() for stmt in migration_sql.split(';')
                            if stmt.strip() and not stmt.strip().startswith('--')
                        ]
                        
                        for statement in statements:
                            if statement:
                                await session.execute(text(statement))
                        
                        # Record migration as applied
                        await session.execute(
                            text(
                                "INSERT INTO schema_migrations (migration_name, applied_at) "
                                "VALUES (:name, :applied_at)"
                            ),
                            {
                                "name": migration_name,
                                "applied_at": datetime.utcnow()
                            }
                        )
                        
                        logger.info(f"Migration {migration_name} applied successfully")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def _create_migrations_table(self, session: AsyncSession) -> None:
        """Create migrations tracking table."""
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
    
    async def _get_applied_migrations(self, session: AsyncSession) -> List[str]:
        """Get list of applied migrations."""
        try:
            result = await session.execute(
                text("SELECT migration_name FROM schema_migrations")
            )
            return [row[0] for row in result.fetchall()]
        except Exception:
            # Table might not exist yet
            return []
    
    async def get_repository(self) -> ArgoRepository:
        """Get ARGO repository instance."""
        session = await anext(get_session())
        return ArgoRepository(session)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check."""
        try:
            start_time = datetime.utcnow()
            
            # Basic connection test
            db_health = await db_manager.health_check()
            
            if db_health["status"] != "healthy":
                return db_health
            
            # Extended health checks
            async with get_session() as session:
                # Check table existence
                table_check = await session.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('floats', 'profiles', 'measurements')
                """))
                
                table_count = table_check.scalar()
                
                # Check data counts
                if table_count >= 3:
                    counts = {}
                    for table in ['floats', 'profiles', 'measurements']:
                        count_result = await session.execute(
                            text(f"SELECT COUNT(*) FROM {table}")
                        )
                        counts[table] = count_result.scalar()
                else:
                    counts = {"note": "Core tables not found - database may need initialization"}
                
                # Calculate total response time
                total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return {
                    "status": "healthy",
                    "response_time_ms": round(total_time, 2),
                    "database_url": settings.database.postgres_server,
                    "tables_found": int(table_count),
                    "data_counts": counts,
                    "migrations_applied": len(await self._get_applied_migrations(session)),
                    "message": "Database is operational"
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Health check error: {str(e)}",
                "error_type": type(e).__name__
            }
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            async with get_session() as session:
                repo = ArgoRepository(session)
                stats = await repo.get_database_statistics()
                
                # Add system-level stats
                system_stats = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """))
                
                table_stats = []
                for row in system_stats.fetchall():
                    table_stats.append({
                        'table': row[1],
                        'inserts': int(row[2]),
                        'updates': int(row[3]),
                        'deletes': int(row[4])
                    })
                
                stats['table_statistics'] = table_stats
                stats['collection_time'] = datetime.utcnow().isoformat()
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {"error": str(e)}
    
    async def reset_database(self) -> None:
        """Reset database (development/testing only)."""
        if settings.is_production:
            raise RuntimeError("Database reset not allowed in production")
        
        logger.warning("Resetting database - all data will be lost")
        
        try:
            async with get_session() as session:
                # Drop all tables in correct order (respecting foreign keys)
                tables_to_drop = [
                    'measurements', 'profile_statistics', 'monthly_climatology',
                    'profiles', 'floats', 'ocean_regions', 'dacs', 'schema_migrations'
                ]
                
                for table in tables_to_drop:
                    try:
                        await session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                    except Exception as e:
                        logger.warning(f"Could not drop table {table}: {e}")
                
                await session.commit()
                
            # Recreate schema
            await self.run_migrations()
            
            logger.info("Database reset completed")
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            raise
    
    async def vacuum_analyze(self) -> None:
        """Run VACUUM ANALYZE on all tables for performance optimization."""
        try:
            async with get_session() as session:
                # Get list of tables
                tables_query = await session.execute(text("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                """))
                
                tables = [row[0] for row in tables_query.fetchall()]
                
                for table in tables:
                    logger.info(f"Running VACUUM ANALYZE on {table}")
                    await session.execute(text(f"VACUUM ANALYZE {table}"))
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    async def backup_database(self, backup_path: Optional[Path] = None) -> str:
        """Create database backup (placeholder for pg_dump integration)."""
        if not backup_path:
            backup_path = Path(f"backup_floatchat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql")
        
        # In a real implementation, this would use pg_dump
        logger.info(f"Database backup functionality not implemented. Would backup to {backup_path}")
        
        return str(backup_path)
    
    async def close(self) -> None:
        """Close database connections gracefully."""
        await db_manager.close()
        logger.info("Database service closed")


# Global service instance
db_service = DatabaseService()


class DatabaseContext:
    """Context manager for database operations."""
    
    def __init__(self):
        self.session: Optional[AsyncSession] = None
        self.repository: Optional[ArgoRepository] = None
    
    async def __aenter__(self):
        session_gen = get_session()
        self.session = await anext(session_gen)
        self.repository = ArgoRepository(self.session)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                await self.session.rollback()
            else:
                await self.session.commit()
            await self.session.close()


# Convenience functions
async def get_argo_repository() -> ArgoRepository:
    """Get ARGO repository with active session."""
    session = await anext(get_session())
    return ArgoRepository(session)


async def execute_with_retry(
    operation, 
    max_retries: int = 3, 
    delay: float = 1.0
):
    """Execute database operation with retry logic."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"Database operation failed after {max_retries + 1} attempts")
                raise last_exception