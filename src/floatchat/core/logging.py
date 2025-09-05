"""
Structured logging configuration for FloatChat.

This module provides comprehensive logging setup with structured output,
performance monitoring, and observability features for production deployment.
"""

import sys
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional
from contextvars import ContextVar

import structlog
from structlog.types import FilteringBoundLogger
from rich.console import Console
from rich.logging import RichHandler

from floatchat.core.config import settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


def add_request_context(
    logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add request context to log entries."""
    request_id = request_id_var.get()
    user_id = user_id_var.get()
    
    if request_id:
        event_dict["request_id"] = request_id
    if user_id:
        event_dict["user_id"] = user_id
        
    return event_dict


def add_performance_context(
    logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add performance context to log entries."""
    # Add memory usage if available
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        event_dict["memory_usage_mb"] = round(memory_mb, 2)
    except ImportError:
        pass
    
    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_request_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add performance monitoring in development
    if settings.is_development:
        processors.insert(-1, add_performance_context)
    
    # Configure output format based on environment
    if settings.monitoring.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        context_class=structlog.threadlocal.wrap_dict(dict),
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structlog": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": (
                    structlog.processors.JSONRenderer()
                    if settings.monitoring.log_format == "json"
                    else structlog.dev.ConsoleRenderer(colors=True)
                ),
            },
            "rich": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            },
        },
        "handlers": {
            "default": {
                "level": settings.monitoring.log_level,
                "class": "logging.StreamHandler",
                "formatter": "structlog",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": settings.monitoring.log_level,
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
            },
            "uvicorn.access": {
                "level": "INFO" if settings.is_development else "WARNING",
            },
            "sqlalchemy.engine": {
                "level": "INFO" if settings.debug else "WARNING",
            },
            "aioredis": {
                "level": "INFO",
            },
            "httpx": {
                "level": "WARNING",
            },
        },
    }
    
    # Add file handler if configured
    if settings.monitoring.log_file:
        log_file = Path(settings.monitoring.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging_config["handlers"]["file"] = {
            "level": settings.monitoring.log_level,
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "formatter": "structlog",
        }
        
        # Add file handler to all loggers
        for logger_config in logging_config["loggers"].values():
            if "handlers" in logger_config:
                logger_config["handlers"].append("file")
    
    # Add Rich handler for development
    if settings.is_development:
        console = Console(force_terminal=True)
        rich_handler = RichHandler(
            console=console,
            show_path=True,
            show_time=True,
            rich_tracebacks=True,
            markup=True,
        )
        
        # Replace default handler with Rich handler
        logging_config["handlers"]["rich"] = {
            "level": settings.monitoring.log_level,
            "class": "rich.logging.RichHandler",
            "formatter": "rich",
        }
        
        for logger_config in logging_config["loggers"].values():
            if "handlers" in logger_config and "default" in logger_config["handlers"]:
                logger_config["handlers"] = ["rich" if h == "default" else h for h in logger_config["handlers"]]
    
    logging.config.dictConfig(logging_config)


def get_logger(name: Optional[str] = None) -> FilteringBoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name. If None, uses the calling module name.
        
    Returns:
        Configured structured logger instance.
    """
    if name is None:
        # Get the calling module name
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "floatchat")
    
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> FilteringBoundLogger:
        """Get logger for this class."""
        class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(class_name)


def log_function_call(func_name: str, **kwargs) -> None:
    """Log function call with parameters.
    
    Args:
        func_name: Name of the function being called.
        **kwargs: Function parameters to log.
    """
    logger = get_logger()
    logger.info(
        "Function called",
        function=func_name,
        parameters={k: v for k, v in kwargs.items() if not k.startswith('_')}
    )


def log_performance(operation: str, duration: float, **context) -> None:
    """Log performance metrics.
    
    Args:
        operation: Name of the operation.
        duration: Duration in seconds.
        **context: Additional context information.
    """
    logger = get_logger("performance")
    logger.info(
        "Performance metric",
        operation=operation,
        duration_seconds=round(duration, 4),
        **context
    )


def log_error(error: Exception, operation: str, **context) -> None:
    """Log error with context.
    
    Args:
        error: Exception that occurred.
        operation: Operation that failed.
        **context: Additional context information.
    """
    logger = get_logger("error")
    logger.error(
        "Operation failed",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
        exc_info=True
    )


# Initialize logging on module import
configure_logging()

# Export commonly used items
__all__ = [
    "get_logger",
    "LoggerMixin", 
    "log_function_call",
    "log_performance",
    "log_error",
    "request_id_var",
    "user_id_var",
]