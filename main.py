#!/usr/bin/env python3
"""
FloatChat - Oceanographic AI Assistant
Main application entry point for the Streamlit interface

Smart India Hackathon 2025 - Advanced AI-Powered Oceanographic Data Analysis System
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the main application
from floatchat.interface.streamlit_app import main as run_streamlit_app

def main():
    """Main entry point for FloatChat application."""
    print("🌊 Starting FloatChat - Oceanographic AI Assistant")
    print("=" * 60)
    print("Smart India Hackathon 2025")
    print("Advanced AI-Powered Oceanographic Data Analysis System")
    print("=" * 60)
    print()
    print("Features:")
    print("✅ Multi-Modal Embeddings (Text + Spatial + Temporal + Parametric)")
    print("✅ RAG Pipeline with FAISS Vector Database")
    print("✅ Multi-Provider LLM Orchestration (OpenAI, Anthropic, Groq)")
    print("✅ Model Context Protocol (MCP) Integration")
    print("✅ Natural Language to SQL Engine")
    print("✅ Interactive Streamlit Interface")
    print("✅ Advanced Data Visualization")
    print("✅ Real-time ARGO Float Data Analysis")
    print()
    print("🚀 Launching Streamlit interface...")
    print("📊 Processing 120+ ARGO floats with 960K+ measurements")
    print("🌍 Coverage: Indian Ocean (2000m depth)")
    print()
    
    # Run the Streamlit application
    run_streamlit_app()

if __name__ == "__main__":
    main()