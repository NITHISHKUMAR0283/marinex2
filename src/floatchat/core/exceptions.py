"""
Custom exception classes for FloatChat.

This module provides comprehensive exception handling for different error scenarios
in the oceanographic data processing and AI system components.
"""

from typing import Optional, Dict, Any


class FloatChatException(Exception):
    """Base exception class for FloatChat application.
    
    All custom exceptions should inherit from this base class to maintain
    consistent error handling throughout the application.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize FloatChat exception.
        
        Args:
            message: Human-readable error message.
            error_code: Unique error code for programmatic handling.
            details: Additional error context and debugging information.
            cause: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


# Configuration and Initialization Exceptions
class ConfigurationError(FloatChatException):
    """Raised when there are configuration-related errors."""
    pass


class InitializationError(FloatChatException):
    """Raised when application or component initialization fails."""
    pass


# Data Processing Exceptions
class DataProcessingError(FloatChatException):
    """Base class for data processing related errors."""
    pass


class NetCDFProcessingError(DataProcessingError):
    """Raised when NetCDF file processing fails."""
    pass


class DataValidationError(DataProcessingError):
    """Raised when data validation checks fail."""
    pass


class CoordinateTransformationError(DataProcessingError):
    """Raised when coordinate system transformation fails."""
    pass


class QualityControlError(DataProcessingError):
    """Raised when quality control checks fail."""
    pass


class MemoryLimitExceededError(DataProcessingError):
    """Raised when processing exceeds memory limits."""
    pass


# Database Exceptions
class DatabaseError(FloatChatException):
    """Base class for database related errors."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Raised when database query execution fails."""
    pass


class MigrationError(DatabaseError):
    """Raised when database migration fails."""
    pass


class IntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""
    pass


# AI/ML Exceptions
class AIModelError(FloatChatException):
    """Base class for AI/ML model related errors."""
    pass


class EmbeddingGenerationError(AIModelError):
    """Raised when embedding generation fails."""
    pass


class VectorSearchError(AIModelError):
    """Raised when vector similarity search fails."""
    pass


class RAGPipelineError(AIModelError):
    """Raised when RAG pipeline execution fails."""
    pass


class LLMIntegrationError(AIModelError):
    """Raised when LLM API integration fails."""
    pass


class MCPError(AIModelError):
    """Raised when Model Context Protocol operations fail."""
    pass


class QueryUnderstandingError(AIModelError):
    """Raised when natural language query understanding fails."""
    pass


# API Exceptions
class APIError(FloatChatException):
    """Base class for API related errors."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization checks fail."""
    pass


class RateLimitExceededError(APIError):
    """Raised when rate limits are exceeded."""
    pass


class ValidationError(APIError):
    """Raised when request validation fails."""
    pass


class ResourceNotFoundError(APIError):
    """Raised when requested resource is not found."""
    pass


class ConflictError(APIError):
    """Raised when resource conflicts occur."""
    pass


# External Service Exceptions
class ExternalServiceError(FloatChatException):
    """Base class for external service integration errors."""
    pass


class ARGODataSourceError(ExternalServiceError):
    """Raised when ARGO data source access fails."""
    pass


class CacheServiceError(ExternalServiceError):
    """Raised when cache service operations fail."""
    pass


class MonitoringServiceError(ExternalServiceError):
    """Raised when monitoring service integration fails."""
    pass


# User Interface Exceptions
class UIError(FloatChatException):
    """Base class for user interface related errors."""
    pass


class VisualizationError(UIError):
    """Raised when data visualization generation fails."""
    pass


class DashboardError(UIError):
    """Raised when dashboard operations fail."""
    pass


class ExportError(UIError):
    """Raised when data export operations fail."""
    pass


# Performance and Resource Exceptions
class PerformanceError(FloatChatException):
    """Base class for performance related errors."""
    pass


class TimeoutError(PerformanceError):
    """Raised when operations exceed timeout limits."""
    pass


class ResourceExhaustionError(PerformanceError):
    """Raised when system resources are exhausted."""
    pass


class ConcurrencyLimitError(PerformanceError):
    """Raised when concurrency limits are exceeded."""
    pass


# Utility Functions for Exception Handling
def handle_exception_chain(exception: Exception) -> FloatChatException:
    """Convert standard exceptions to FloatChat exceptions.
    
    Args:
        exception: The original exception to convert.
        
    Returns:
        Appropriate FloatChat exception with original as cause.
    """
    if isinstance(exception, FloatChatException):
        return exception
    
    # Map common standard exceptions to FloatChat exceptions
    exception_mapping = {
        ValueError: DataValidationError,
        KeyError: ResourceNotFoundError,
        FileNotFoundError: ARGODataSourceError,
        MemoryError: MemoryLimitExceededError,
        ConnectionRefusedError: ConnectionError,
        TimeoutError: TimeoutError,
    }
    
    exception_class = exception_mapping.get(type(exception), FloatChatException)
    
    return exception_class(
        message=str(exception),
        error_code=type(exception).__name__,
        cause=exception
    )


def get_error_context(exception: Exception) -> Dict[str, Any]:
    """Extract error context from exception for logging.
    
    Args:
        exception: The exception to extract context from.
        
    Returns:
        Dictionary containing error context information.
    """
    context = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
    }
    
    if isinstance(exception, FloatChatException):
        context.update({
            "error_code": exception.error_code,
            "details": exception.details,
        })
        
        if exception.cause:
            context["cause"] = {
                "type": type(exception.cause).__name__,
                "message": str(exception.cause)
            }
    
    return context