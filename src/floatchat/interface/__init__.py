"""
FloatChat User Interface Module.
Streamlit-based conversational interface for oceanographic AI system.
"""

from .streamlit_app import FloatChatApp
from .chat_components import ChatInterface, MessageHandler, VisualizationEngine
from .data_explorer import DataExplorer, QueryVisualizer
from .config_manager import UIConfig, SessionManager

__all__ = [
    'FloatChatApp',
    'ChatInterface',
    'MessageHandler', 
    'VisualizationEngine',
    'DataExplorer',
    'QueryVisualizer',
    'UIConfig',
    'SessionManager'
]