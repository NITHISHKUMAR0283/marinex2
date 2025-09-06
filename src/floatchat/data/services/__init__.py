"""
Data services for ARGO processing.

This package contains high-level services for data ingestion,
processing orchestration, and batch operations.
"""

from .ingestion_service import (
    IngestionService, 
    IngestionJob, 
    IngestionProgress,
    ingestion_service,
    ingest_argo_directory,
    get_ingestion_progress
)

__all__ = [
    "IngestionService",
    "IngestionJob", 
    "IngestionProgress",
    "ingestion_service",
    "ingest_argo_directory",
    "get_ingestion_progress"
]