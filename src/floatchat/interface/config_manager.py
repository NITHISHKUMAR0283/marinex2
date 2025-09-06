"""
Configuration and Session Management for FloatChat Interface
Handles UI configuration, user sessions, and application state
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class UIConfig:
    """UI configuration settings."""
    theme: str = "light"
    primary_color: str = "#1f77b4"
    background_color: str = "#ffffff"
    sidebar_expanded: bool = True
    max_chat_history: int = 100
    auto_scroll: bool = True
    show_confidence_scores: bool = True
    show_source_citations: bool = True
    enable_voice_input: bool = False
    default_llm_provider: str = "OpenAI GPT-4"
    default_query_mode: str = "Conversational RAG"
    visualization_theme: str = "plotly"
    map_style: str = "open-street-map"
    enable_notifications: bool = True
    session_timeout_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIConfig':
        """Create from dictionary."""
        return cls(**data)

class SessionManager:
    """Manages user sessions and application state."""
    
    def __init__(self, config_dir: str = "./config"):
        """Initialize session manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "ui_config.json"
        self.session_file = self.config_dir / "session_data.json"
        
        # Load or create configuration
        self.config = self._load_config()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _load_config(self) -> UIConfig:
        """Load UI configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return UIConfig.from_dict(config_data)
            except Exception as e:
                st.warning(f"Failed to load config: {e}. Using defaults.")
        
        return UIConfig()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            st.error(f"Failed to save config: {e}")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Core application state
        if 'ui_config' not in st.session_state:
            st.session_state.ui_config = self.config
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now()
        
        # Chat and interaction state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if 'favorite_queries' not in st.session_state:
            st.session_state.favorite_queries = []
        
        # AI component state
        if 'selected_llm_provider' not in st.session_state:
            st.session_state.selected_llm_provider = self.config.default_llm_provider
        
        if 'query_mode' not in st.session_state:
            st.session_state.query_mode = self.config.default_query_mode
        
        # UI state
        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False
        
        if 'show_explorer' not in st.session_state:
            st.session_state.show_explorer = False
        
        if 'show_help' not in st.session_state:
            st.session_state.show_help = False
        
        # Performance metrics
        if 'query_metrics' not in st.session_state:
            st.session_state.query_metrics = {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'avg_response_time': 0,
                'total_response_time': 0
            }
    
    def update_config(self, **kwargs):
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                st.session_state.ui_config = self.config
        
        self.save_config()
    
    def add_to_chat_history(self, message: Dict[str, Any]):
        """Add message to chat history with limits."""
        st.session_state.chat_history.append(message)
        
        # Limit chat history size
        max_history = self.config.max_chat_history
        if len(st.session_state.chat_history) > max_history:
            st.session_state.chat_history = st.session_state.chat_history[-max_history:]
    
    def add_to_query_history(self, query: str):
        """Add query to query history."""
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        
        # Keep only last 50 queries
        if len(st.session_state.query_history) > 50:
            st.session_state.query_history = st.session_state.query_history[-50:]
    
    def add_favorite_query(self, query: str, description: str = ""):
        """Add query to favorites."""
        favorite = {
            'query': query,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'usage_count': 0
        }
        
        # Check if already exists
        existing = [fav for fav in st.session_state.favorite_queries if fav['query'] == query]
        if not existing:
            st.session_state.favorite_queries.append(favorite)
    
    def update_query_metrics(self, success: bool, response_time: float):
        """Update query performance metrics."""
        metrics = st.session_state.query_metrics
        
        metrics['total_queries'] += 1
        metrics['total_response_time'] += response_time
        
        if success:
            metrics['successful_queries'] += 1
        else:
            metrics['failed_queries'] += 1
        
        metrics['avg_response_time'] = metrics['total_response_time'] / metrics['total_queries']
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        session_duration = datetime.now() - st.session_state.session_start_time
        metrics = st.session_state.query_metrics
        
        return {
            'session_id': st.session_state.session_id,
            'session_duration': str(session_duration),
            'total_messages': len(st.session_state.chat_history),
            'queries_this_session': len(st.session_state.query_history),
            'favorite_queries': len(st.session_state.favorite_queries),
            'success_rate': (metrics['successful_queries'] / max(metrics['total_queries'], 1)) * 100,
            'avg_response_time': metrics['avg_response_time'],
            'total_queries': metrics['total_queries']
        }
    
    def save_session_data(self):
        """Save session data to file."""
        session_data = {
            'session_id': st.session_state.session_id,
            'start_time': st.session_state.session_start_time.isoformat(),
            'chat_history': st.session_state.chat_history[-20:],  # Save last 20 messages
            'query_history': st.session_state.query_history,
            'favorite_queries': st.session_state.favorite_queries,
            'query_metrics': st.session_state.query_metrics,
            'selected_llm_provider': st.session_state.selected_llm_provider,
            'query_mode': st.session_state.query_mode
        }
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save session data: {e}")
    
    def load_session_data(self) -> bool:
        """Load previous session data."""
        if not self.session_file.exists():
            return False
        
        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Restore session state
            if 'query_history' in session_data:
                st.session_state.query_history = session_data['query_history']
            
            if 'favorite_queries' in session_data:
                st.session_state.favorite_queries = session_data['favorite_queries']
            
            if 'query_metrics' in session_data:
                st.session_state.query_metrics = session_data['query_metrics']
            
            if 'selected_llm_provider' in session_data:
                st.session_state.selected_llm_provider = session_data['selected_llm_provider']
            
            if 'query_mode' in session_data:
                st.session_state.query_mode = session_data['query_mode']
            
            return True
            
        except Exception as e:
            st.warning(f"Failed to load session data: {e}")
            return False
    
    def render_settings_panel(self):
        """Render the settings configuration panel."""
        if not st.session_state.show_settings:
            return
        
        with st.sidebar.expander("⚙️ Settings", expanded=True):
            st.write("### UI Configuration")
            
            # Theme settings
            new_theme = st.selectbox(
                "Theme:",
                ["light", "dark"],
                index=0 if self.config.theme == "light" else 1
            )
            
            # Display settings
            new_show_confidence = st.checkbox(
                "Show Confidence Scores",
                value=self.config.show_confidence_scores
            )
            
            new_show_citations = st.checkbox(
                "Show Source Citations",
                value=self.config.show_source_citations
            )
            
            new_auto_scroll = st.checkbox(
                "Auto-scroll Chat",
                value=self.config.auto_scroll
            )
            
            # Chat settings
            new_max_history = st.slider(
                "Max Chat History:",
                10, 200, self.config.max_chat_history
            )
            
            # Default settings
            st.write("### Default Preferences")
            
            new_default_provider = st.selectbox(
                "Default LLM Provider:",
                ["OpenAI GPT-4", "Anthropic Claude", "Groq Llama3"],
                index=["OpenAI GPT-4", "Anthropic Claude", "Groq Llama3"].index(self.config.default_llm_provider)
            )
            
            new_default_mode = st.selectbox(
                "Default Query Mode:",
                ["Conversational RAG", "Direct SQL", "Hybrid Analysis"],
                index=["Conversational RAG", "Direct SQL", "Hybrid Analysis"].index(self.config.default_query_mode)
            )
            
            # Save settings
            if st.button("💾 Save Settings"):
                self.update_config(
                    theme=new_theme,
                    show_confidence_scores=new_show_confidence,
                    show_source_citations=new_show_citations,
                    auto_scroll=new_auto_scroll,
                    max_chat_history=new_max_history,
                    default_llm_provider=new_default_provider,
                    default_query_mode=new_default_mode
                )
                st.success("Settings saved!")
                st.experimental_rerun()
    
    def render_session_info(self):
        """Render session information panel."""
        stats = self.get_session_statistics()
        
        with st.sidebar.expander("📊 Session Info", expanded=False):
            st.metric("Session Duration", stats['session_duration'])
            st.metric("Total Messages", stats['total_messages'])
            st.metric("Queries Made", stats['total_queries'])
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
            st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
            
            if st.button("💾 Save Session"):
                self.save_session_data()
                st.success("Session saved!")
            
            if st.button("🔄 Load Previous Session"):
                if self.load_session_data():
                    st.success("Previous session loaded!")
                    st.experimental_rerun()
                else:
                    st.warning("No previous session found.")
    
    def render_query_history(self):
        """Render query history panel."""
        if not st.session_state.query_history:
            return
        
        with st.sidebar.expander("🕒 Query History", expanded=False):
            for i, query in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                if st.button(f"{i}. {query[:30]}...", key=f"history_{i}"):
                    st.session_state.user_input = query
                    st.experimental_rerun()
    
    def render_favorite_queries(self):
        """Render favorite queries panel."""
        with st.sidebar.expander("⭐ Favorite Queries", expanded=False):
            
            # Add current query to favorites
            if st.session_state.get('user_input'):
                if st.button("⭐ Add Current Query to Favorites"):
                    self.add_favorite_query(st.session_state.user_input)
                    st.success("Added to favorites!")
            
            # Display favorites
            if st.session_state.favorite_queries:
                for i, fav in enumerate(st.session_state.favorite_queries):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if st.button(f"{fav['query'][:25]}...", key=f"fav_{i}"):
                            st.session_state.user_input = fav['query']
                            fav['usage_count'] += 1
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"del_fav_{i}"):
                            st.session_state.favorite_queries.remove(fav)
                            st.experimental_rerun()
            else:
                st.info("No favorite queries yet.")
    
    def cleanup_session(self):
        """Clean up session resources."""
        # Save current session before cleanup
        self.save_session_data()
        
        # Clear temporary data
        temp_keys = ['temp_data', 'processing_state', 'intermediate_results']
        for key in temp_keys:
            if key in st.session_state:
                del st.session_state[key]