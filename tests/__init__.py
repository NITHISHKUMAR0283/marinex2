"""
FloatChat Testing Suite
Comprehensive end-to-end testing and performance validation
"""

# Test configuration and utilities
import sys
import os
from pathlib import Path

# Add src to path for imports
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Test categories
__all__ = [
    'test_embeddings',
    'test_vector_database', 
    'test_rag_pipeline',
    'test_llm_orchestration',
    'test_nl2sql',
    'test_mcp_integration',
    'test_streamlit_interface',
    'performance_tests',
    'integration_tests'
]