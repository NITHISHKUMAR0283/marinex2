"""
Main FastAPI application for FloatChat.

This module sets up the FastAPI application with all necessary middleware,
routing, and configuration for production deployment.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from floatchat.core.config import settings
from floatchat.core.logging import get_logger, request_id_var, log_performance, log_error
from floatchat.core.exceptions import FloatChatException, get_error_context
from floatchat.api.health import router as health_router
from floatchat.infrastructure.database.service import db_service

# Initialize logger
logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'floatchat_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'floatchat_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Counter(
    'floatchat_active_connections',
    'Active connections'
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting FloatChat application", version=settings.app_version)
    
    try:
        # Initialize application components
        await initialize_application()
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error("Application startup failed", exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down FloatChat application")
        await cleanup_application()
        logger.info("Application shutdown completed")


async def initialize_application() -> None:
    """Initialize application components during startup."""
    try:
        # Initialize database
        logger.info("Initializing database...")
        await db_service.initialize_database()
        logger.info("Database initialization completed")
        
        # Additional initialization will be added in later phases
        # - Redis cache initialization
        # - AI model loading
        # - Background task startup
        
    except Exception as e:
        logger.error(f"Failed to initialize application components: {e}")
        raise


async def cleanup_application() -> None:
    """Cleanup application resources during shutdown."""
    try:
        # Close database connections
        logger.info("Closing database connections...")
        await db_service.close()
        logger.info("Database connections closed")
        
        # Additional cleanup will be added in later phases
        # - Redis connection cleanup
        # - Background task shutdown
        # - AI model cleanup
        
    except Exception as e:
        logger.error(f"Error during application cleanup: {e}")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add routers
    setup_routers(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Set up middleware for the application.
    
    Args:
        app: FastAPI application instance.
    """
    
    # Trusted Host middleware (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.is_development else ["localhost", "127.0.0.1"]
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware for logging and metrics
    @app.middleware("http")
    async def logging_and_metrics_middleware(request: Request, call_next):
        """Custom middleware for request logging and metrics collection."""
        
        # Generate request ID
        request_id = f"{int(time.time() * 1000)}-{id(request)}"
        request_id_var.set(request_id)
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=round(duration, 4)
            )
            
            # Log performance if slow
            if duration > 1.0:  # Log slow requests (>1s)
                log_performance(
                    operation=f"{request.method} {request.url.path}",
                    duration=duration,
                    request_id=request_id,
                    status_code=response.status_code
                )
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Update metrics for errors
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            # Log error
            log_error(
                error=e,
                operation=f"{request.method} {request.url.path}",
                request_id=request_id,
                duration_seconds=round(duration, 4)
            )
            
            raise


def setup_routers(app: FastAPI) -> None:
    """Set up API routers.
    
    Args:
        app: FastAPI application instance.
    """
    
    # Health check router
    app.include_router(health_router, prefix="/health", tags=["health"])
    
    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Additional routers will be added in later phases
    # app.include_router(data_router, prefix="/api/v1/data", tags=["data"])
    # app.include_router(ai_router, prefix="/api/v1/ai", tags=["ai"])
    # app.include_router(viz_router, prefix="/api/v1/visualization", tags=["visualization"])


def setup_exception_handlers(app: FastAPI) -> None:
    """Set up custom exception handlers.
    
    Args:
        app: FastAPI application instance.
    """
    
    @app.exception_handler(FloatChatException)
    async def floatchat_exception_handler(request: Request, exc: FloatChatException):
        """Handle FloatChat custom exceptions."""
        
        logger.error(
            "FloatChat exception occurred",
            **get_error_context(exc),
            request_id=request_id_var.get(),
            url=str(request.url)
        )
        
        # Determine HTTP status code based on exception type
        from floatchat.core.exceptions import (
            AuthenticationError, AuthorizationError, ValidationError,
            ResourceNotFoundError, ConflictError, RateLimitExceededError
        )
        
        status_code_mapping = {
            AuthenticationError: 401,
            AuthorizationError: 403,
            ValidationError: 422,
            ResourceNotFoundError: 404,
            ConflictError: 409,
            RateLimitExceededError: 429,
        }
        
        status_code = status_code_mapping.get(type(exc), 500)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.to_dict(),
                "request_id": request_id_var.get()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        
        logger.warning(
            "Request validation failed",
            errors=exc.errors(),
            request_id=request_id_var.get(),
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": exc.errors()
                },
                "request_id": request_id_var.get()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            request_id=request_id_var.get(),
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "HTTPException",
                    "message": exc.detail,
                    "status_code": exc.status_code
                },
                "request_id": request_id_var.get()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        
        log_error(
            error=exc,
            operation=f"{request.method} {request.url.path}",
            request_id=request_id_var.get(),
            url=str(request.url)
        )
        
        # Don't expose internal error details in production
        error_detail = str(exc) if settings.is_development else "Internal server error"
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError", 
                    "message": error_detail
                },
                "request_id": request_id_var.get()
            }
        )


# Create the application instance
app = create_application()


def run_server() -> None:
    """Run the FastAPI server.
    
    This function is used by the CLI command to start the server.
    """
    import uvicorn
    
    uvicorn.run(
        "floatchat.api.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        workers=settings.api.api_workers,
        reload=settings.api.api_reload and settings.is_development,
        log_config=None,  # We use our own logging configuration
        access_log=False,  # We handle access logging in middleware
    )


if __name__ == "__main__":
    run_server()