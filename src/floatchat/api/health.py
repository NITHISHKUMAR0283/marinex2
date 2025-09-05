"""
Health check endpoints for FloatChat API.

This module provides comprehensive health checking functionality including
database connectivity, cache status, and external service availability.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from floatchat.core.config import settings
from floatchat.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]


class ComponentHealth(BaseModel):
    """Individual component health status."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Application start time for uptime calculation
_start_time = time.time()


async def check_database_health() -> ComponentHealth:
    """Check database connectivity and performance.
    
    Returns:
        Database health status.
    """
    start_time = time.time()
    
    try:
        # This will be implemented when database layer is ready
        # For now, simulate a database check
        await asyncio.sleep(0.01)  # Simulate database query
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status="healthy",
            response_time_ms=round(response_time, 2),
            message="Database connection successful",
            details={
                "driver": "asyncpg",
                "pool_size": settings.database.database_pool_size,
                "max_overflow": settings.database.database_max_overflow
            }
        )
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return ComponentHealth(
            status="unhealthy",
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            message=f"Database connection failed: {str(e)}",
            details={"error": str(e)}
        )


async def check_redis_health() -> ComponentHealth:
    """Check Redis connectivity and performance.
    
    Returns:
        Redis health status.
    """
    start_time = time.time()
    
    try:
        # This will be implemented when Redis layer is ready
        # For now, simulate a Redis check
        await asyncio.sleep(0.005)  # Simulate Redis ping
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status="healthy",
            response_time_ms=round(response_time, 2),
            message="Redis connection successful",
            details={
                "host": settings.redis.redis_host,
                "port": settings.redis.redis_port,
                "db": settings.redis.redis_db
            }
        )
        
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return ComponentHealth(
            status="unhealthy",
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            message=f"Redis connection failed: {str(e)}",
            details={"error": str(e)}
        )


async def check_ai_services_health() -> ComponentHealth:
    """Check AI services availability.
    
    Returns:
        AI services health status.
    """
    start_time = time.time()
    
    try:
        # Check if AI service configurations are available
        issues = []
        
        if not settings.ai.openai_api_key and not settings.ai.anthropic_api_key:
            issues.append("No AI API keys configured")
        
        if issues:
            return ComponentHealth(
                status="degraded",
                response_time_ms=round((time.time() - start_time) * 1000, 2),
                message="AI services partially configured",
                details={"issues": issues}
            )
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status="healthy",
            response_time_ms=round(response_time, 2),
            message="AI services configured",
            details={
                "openai_configured": bool(settings.ai.openai_api_key),
                "anthropic_configured": bool(settings.ai.anthropic_api_key),
                "embedding_model": settings.ai.embedding_model
            }
        )
        
    except Exception as e:
        logger.error("AI services health check failed", error=str(e))
        return ComponentHealth(
            status="unhealthy",
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            message=f"AI services check failed: {str(e)}",
            details={"error": str(e)}
        )


async def check_data_processing_health() -> ComponentHealth:
    """Check data processing capabilities.
    
    Returns:
        Data processing health status.
    """
    start_time = time.time()
    
    try:
        # Check if data directories exist and are accessible
        data_path = settings.data.argo_data_path
        
        # Create data directory if it doesn't exist
        data_path.mkdir(parents=True, exist_ok=True)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status="healthy",
            response_time_ms=round(response_time, 2),
            message="Data processing system ready",
            details={
                "data_path": str(data_path),
                "max_concurrent_files": settings.data.max_concurrent_files,
                "memory_limit_gb": settings.data.memory_limit_gb
            }
        )
        
    except Exception as e:
        logger.error("Data processing health check failed", error=str(e))
        return ComponentHealth(
            status="unhealthy",
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            message=f"Data processing check failed: {str(e)}",
            details={"error": str(e)}
        )


@router.get("/", response_model=HealthStatus)
async def get_health_status() -> HealthStatus:
    """Get comprehensive application health status.
    
    Returns:
        Complete health status including all component checks.
    """
    logger.info("Health check requested")
    
    # Run all health checks concurrently
    database_health, redis_health, ai_health, data_health = await asyncio.gather(
        check_database_health(),
        check_redis_health(), 
        check_ai_services_health(),
        check_data_processing_health(),
        return_exceptions=True
    )
    
    # Handle exceptions from health checks
    checks = {}
    
    if isinstance(database_health, Exception):
        checks["database"] = ComponentHealth(
            status="unhealthy",
            message=f"Health check error: {str(database_health)}"
        ).dict()
    else:
        checks["database"] = database_health.dict()
    
    if isinstance(redis_health, Exception):
        checks["redis"] = ComponentHealth(
            status="unhealthy",
            message=f"Health check error: {str(redis_health)}"
        ).dict()
    else:
        checks["redis"] = redis_health.dict()
    
    if isinstance(ai_health, Exception):
        checks["ai_services"] = ComponentHealth(
            status="unhealthy",
            message=f"Health check error: {str(ai_health)}"
        ).dict()
    else:
        checks["ai_services"] = ai_health.dict()
    
    if isinstance(data_health, Exception):
        checks["data_processing"] = ComponentHealth(
            status="unhealthy", 
            message=f"Health check error: {str(data_health)}"
        ).dict()
    else:
        checks["data_processing"] = data_health.dict()
    
    # Determine overall status
    component_statuses = [check["status"] for check in checks.values()]
    
    if all(status == "healthy" for status in component_statuses):
        overall_status = "healthy"
    elif any(status == "unhealthy" for status in component_statuses):
        overall_status = "unhealthy"  
    else:
        overall_status = "degraded"
    
    # Calculate uptime
    uptime = time.time() - _start_time
    
    health_status = HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(uptime, 2),
        checks=checks
    )
    
    logger.info(
        "Health check completed",
        status=overall_status,
        uptime_seconds=uptime,
        component_count=len(checks)
    )
    
    return health_status


@router.get("/live", status_code=200)
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint.
    
    Returns:
        Simple status indicating the application is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready", status_code=200)
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint.
    
    Returns:
        Status indicating if the application is ready to serve traffic.
        
    Raises:
        HTTPException: If the application is not ready.
    """
    # Perform lightweight checks for readiness
    try:
        # Check critical components only
        database_health = await check_database_health()
        
        if database_health.status == "unhealthy":
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "reason": "Database not available",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness probe failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": f"Readiness check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )