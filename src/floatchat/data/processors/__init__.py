"""
Data processors for ARGO NetCDF files.

This package contains high-performance processors for parsing and validating
ARGO oceanographic data from NetCDF files.
"""

from .netcdf_processor import NetCDFProcessor, ProcessingStats, ArgoProfileData, ArgoFloatMetadata

__all__ = [
    "NetCDFProcessor",
    "ProcessingStats", 
    "ArgoProfileData",
    "ArgoFloatMetadata"
]