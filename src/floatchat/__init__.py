"""
FloatChat: AI-Powered Conversational Interface for ARGO Ocean Data Discovery and Visualization.

A sophisticated conversational AI system that democratizes access to ARGO oceanographic data
through natural language queries, interactive visualizations, and AI-powered insights.

This package provides:
- High-performance ARGO NetCDF data processing
- Advanced RAG (Retrieval-Augmented Generation) pipeline with MCP integration  
- Multi-modal embeddings for oceanographic data
- Interactive Streamlit dashboard with geospatial visualizations
- Natural language querying interface for complex oceanographic analysis

Built for Smart India Hackathon 2025 with production-ready architecture.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("floatchat")
except PackageNotFoundError:
    __version__ = "unknown"

__title__ = "FloatChat"
__description__ = "AI-Powered Conversational Interface for ARGO Ocean Data Discovery"
__author__ = "FloatChat Development Team"
__email__ = "floatchat@sih2025.dev"
__license__ = "MIT"
__copyright__ = "Copyright 2025 FloatChat Development Team"

# Public API exports
from floatchat.core.config import settings
from floatchat.core.logging import get_logger

__all__ = [
    "__version__",
    "__title__", 
    "__description__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    "settings",
    "get_logger",
]