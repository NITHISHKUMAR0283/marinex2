"""
FloatChat Streamlit Application - Main Interface
Advanced conversational AI for oceanographic data analysis
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Any

# Import FloatChat AI components
from ..ai.rag.rag_pipeline import OceanographicRAGPipeline
from ..ai.llm.llm_orchestrator import LLMOrchestrator, LLMProvider, LLMConfig
from ..ai.nl2sql.nl2sql_engine import NL2SQLEngine
from ..ai.vector_database.faiss_vector_store import OceanographicVectorStore
from ..core.database import DatabaseManager
from .chat_components import ChatInterface, MessageHandler, VisualizationEngine
from .data_explorer import DataExplorer

class FloatChatApp:
    """Main Streamlit application for FloatChat oceanographic AI."""
    
    def __init__(self):
        """Initialize the FloatChat application."""
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="FloatChat - Oceanographic AI",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UI
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .chat-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .query-box {
            background: #e8f4fd;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
            
        if 'llm_orchestrator' not in st.session_state:
            st.session_state.llm_orchestrator = None
            
        if 'nl2sql_engine' not in st.session_state:
            st.session_state.nl2sql_engine = None
            
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
            
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = None
            
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    async def initialize_ai_components(self):
        """Initialize all AI components asynchronously."""
        try:
            with st.spinner("Initializing AI components..."):
                # Database configuration
                db_config = {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'floatchat_indian_ocean',
                    'user': 'postgres',
                    'password': 'password'
                }
                
                # Initialize database manager
                st.session_state.db_manager = DatabaseManager(db_config)
                
                # Initialize vector store
                st.session_state.vector_store = OceanographicVectorStore(
                    index_path="./data/vector_index",
                    dimension=1536
                )
                
                # Initialize LLM orchestrator with all providers
                llm_configs = {
                    LLMProvider.OPENAI: LLMConfig(
                        api_key=os.getenv('OPENAI_API_KEY', ''),
                        model_name='gpt-4-turbo-preview',
                        max_tokens=4000,
                        temperature=0.1
                    ),
                    LLMProvider.ANTHROPIC: LLMConfig(
                        api_key=os.getenv('ANTHROPIC_API_KEY', ''),
                        model_name='claude-3-sonnet-20240229',
                        max_tokens=4000,
                        temperature=0.1
                    ),
                    LLMProvider.GROQ: LLMConfig(
                        api_key=os.getenv('GROQ_API_KEY', ''),
                        model_name='llama3-70b-8192',
                        max_tokens=4000,
                        temperature=0.1
                    )
                }
                
                st.session_state.llm_orchestrator = LLMOrchestrator(llm_configs)
                
                # Initialize NL2SQL engine
                st.session_state.nl2sql_engine = NL2SQLEngine(db_config)
                
                # Initialize RAG pipeline
                st.session_state.rag_pipeline = OceanographicRAGPipeline(
                    vector_store=st.session_state.vector_store,
                    llm_orchestrator=st.session_state.llm_orchestrator,
                    db_manager=st.session_state.db_manager
                )
                
                st.session_state.system_initialized = True
                st.success("AI components initialized successfully!")
                
        except Exception as e:
            st.error(f"Failed to initialize AI components: {str(e)}")
            st.session_state.system_initialized = False
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🌊 FloatChat - Oceanographic AI Assistant</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARGO Floats", "120+", "Global Coverage")
        with col2:
            st.metric("Measurements", "960K+", "2000m Depth")
        with col3:
            st.metric("AI Models", "3 Providers", "RAG + MCP")
    
    def render_sidebar(self):
        """Render the application sidebar with controls."""
        with st.sidebar:
            st.header("🔧 System Controls")
            
            # System status
            if st.session_state.system_initialized:
                st.success("✅ AI System Ready")
            else:
                st.warning("⏳ Initializing...")
                if st.button("Initialize AI Components"):
                    asyncio.run(self.initialize_ai_components())
            
            st.divider()
            
            # LLM Provider Selection
            st.header("🤖 AI Configuration")
            selected_provider = st.selectbox(
                "Primary LLM Provider",
                ["OpenAI GPT-4", "Anthropic Claude", "Groq Llama3"],
                index=0
            )
            
            # Query mode selection
            query_mode = st.radio(
                "Query Mode",
                ["Conversational RAG", "Direct SQL", "Hybrid Analysis"],
                index=0
            )
            
            st.session_state.query_mode = query_mode
            st.session_state.selected_provider = selected_provider
            
            st.divider()
            
            # Quick actions
            st.header("⚡ Quick Actions")
            
            if st.button("📊 System Statistics"):
                self.show_system_stats()
            
            if st.button("🗺️ Data Explorer"):
                st.session_state.show_explorer = True
            
            if st.button("🔄 Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    def render_main_interface(self):
        """Render the main chat interface."""
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                self.display_message(message)
        
        # Query input
        with st.form("query_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_query = st.text_area(
                    "Ask about oceanographic data:",
                    placeholder="e.g., Show me temperature profiles in the Bay of Bengal last month",
                    height=100,
                    key="user_input"
                )
            
            with col2:
                st.write("")  # Spacer
                submit_button = st.form_submit_button("🚀 Analyze", use_container_width=True)
                
                # Example queries
                st.markdown("**Quick Examples:**")
                examples = [
                    "Temperature near equator",
                    "Salinity in Arabian Sea",
                    "Oxygen below 1000m",
                    "Float WMO 2902746 data"
                ]
                
                for example in examples:
                    if st.button(f"💡 {example}", key=f"example_{example}"):
                        st.session_state.user_input = example
                        submit_button = True
            
            # Process query
            if submit_button and user_query:
                await self.process_user_query(user_query)
    
    async def process_user_query(self, query: str):
        """Process user query through the AI pipeline."""
        if not st.session_state.system_initialized:
            st.error("Please initialize AI components first!")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query,
            'timestamp': datetime.now()
        })
        
        try:
            with st.spinner("🔍 Analyzing your query..."):
                
                # Determine processing mode
                if st.session_state.query_mode == "Direct SQL":
                    response = await self.process_sql_query(query)
                elif st.session_state.query_mode == "Conversational RAG":
                    response = await self.process_rag_query(query)
                else:  # Hybrid Analysis
                    response = await self.process_hybrid_query(query)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                
                st.experimental_rerun()
                
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            st.session_state.chat_history.append({
                'type': 'error',
                'content': error_msg,
                'timestamp': datetime.now()
            })
            st.experimental_rerun()
    
    async def process_rag_query(self, query: str) -> dict:
        """Process query through RAG pipeline."""
        rag_response = await st.session_state.rag_pipeline.process_query(
            query=query,
            k_retrievals=5,
            context_window=4000
        )
        
        return {
            'type': 'rag_response',
            'answer': rag_response['answer'],
            'confidence': rag_response['confidence_score'],
            'sources': rag_response['retrieved_contexts'][:3],  # Top 3 sources
            'metadata': rag_response.get('metadata', {})
        }
    
    async def process_sql_query(self, query: str) -> dict:
        """Process query through NL2SQL engine."""
        sql_response = await st.session_state.nl2sql_engine.process_query(query)
        
        if sql_response.get('error'):
            return {
                'type': 'error',
                'message': sql_response['error']
            }
        
        # Execute SQL query
        try:
            results = await st.session_state.db_manager.execute_query(
                sql_response['sql_query'],
                sql_response['parameters']
            )
            
            return {
                'type': 'sql_response',
                'sql_query': sql_response['sql_query'],
                'results': results,
                'confidence': sql_response['confidence_score'],
                'metadata': {
                    'estimated_rows': sql_response['estimated_rows'],
                    'performance_notes': sql_response['performance_notes']
                }
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"SQL execution failed: {str(e)}"
            }
    
    async def process_hybrid_query(self, query: str) -> dict:
        """Process query through hybrid RAG + SQL approach."""
        # Get both RAG and SQL responses
        rag_response = await self.process_rag_query(query)
        sql_response = await self.process_sql_query(query)
        
        return {
            'type': 'hybrid_response',
            'rag_analysis': rag_response,
            'sql_analysis': sql_response,
            'combined_confidence': (
                rag_response.get('confidence', 0) + 
                sql_response.get('confidence', 0)
            ) / 2
        }
    
    def display_message(self, message: dict):
        """Display a chat message with appropriate formatting."""
        timestamp = message['timestamp'].strftime("%H:%M:%S")
        
        if message['type'] == 'user':
            with st.chat_message("user"):
                st.write(f"**{timestamp}** - {message['content']}")
        
        elif message['type'] == 'assistant':
            with st.chat_message("assistant"):
                self.display_ai_response(message['content'], timestamp)
        
        elif message['type'] == 'error':
            with st.chat_message("assistant"):
                st.error(f"**{timestamp}** - {message['content']}")
    
    def display_ai_response(self, response: dict, timestamp: str):
        """Display AI response with visualizations."""
        st.write(f"**{timestamp}** - AI Analysis")
        
        if response['type'] == 'rag_response':
            self.display_rag_response(response)
        elif response['type'] == 'sql_response':
            self.display_sql_response(response)
        elif response['type'] == 'hybrid_response':
            self.display_hybrid_response(response)
    
    def display_rag_response(self, response: dict):
        """Display RAG pipeline response."""
        st.write("**🧠 AI Analysis:**")
        st.write(response['answer'])
        
        # Confidence indicator
        confidence = response['confidence']
        color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
        st.markdown(f"**Confidence:** <span style='color: {color}'>{confidence:.2%}</span>", 
                   unsafe_allow_html=True)
        
        # Sources
        if response.get('sources'):
            with st.expander("📚 Source Data"):
                for i, source in enumerate(response['sources'], 1):
                    st.write(f"**Source {i}:** {source.get('content', '')[:200]}...")
    
    def display_sql_response(self, response: dict):
        """Display SQL query response with results."""
        st.write("**💾 Database Query:**")
        
        # Show SQL query
        st.code(response['sql_query'], language='sql')
        
        # Show results
        if response.get('results'):
            df = pd.DataFrame(response['results'])
            st.dataframe(df, use_container_width=True)
            
            # Auto-generate visualizations for numerical data
            self.auto_visualize_data(df)
        
        # Metadata
        if response.get('metadata'):
            with st.expander("ℹ️ Query Metadata"):
                st.json(response['metadata'])
    
    def display_hybrid_response(self, response: dict):
        """Display hybrid analysis response."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🧠 RAG Analysis**")
            if response['rag_analysis']['type'] != 'error':
                self.display_rag_response(response['rag_analysis'])
            else:
                st.error("RAG analysis failed")
        
        with col2:
            st.write("**💾 SQL Analysis**")
            if response['sql_analysis']['type'] != 'error':
                self.display_sql_response(response['sql_analysis'])
            else:
                st.error("SQL analysis failed")
    
    def auto_visualize_data(self, df: pd.DataFrame):
        """Automatically create visualizations for data."""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) == 0:
            return
        
        # Time series plot if we have date/time columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols and len(numeric_cols) > 0:
            fig = px.line(df, x=date_cols[0], y=numeric_cols[0], 
                         title=f"{numeric_cols[0]} Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot for multiple numeric columns
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Histogram for single numeric column
        elif len(numeric_cols) == 1:
            fig = px.histogram(df, x=numeric_cols[0], 
                             title=f"Distribution of {numeric_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_system_stats(self):
        """Display system statistics."""
        with st.expander("📊 System Statistics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**AI Components Status:**")
                components = {
                    "RAG Pipeline": "✅ Active" if st.session_state.rag_pipeline else "❌ Not loaded",
                    "LLM Orchestrator": "✅ Active" if st.session_state.llm_orchestrator else "❌ Not loaded",
                    "Vector Store": "✅ Active" if st.session_state.vector_store else "❌ Not loaded",
                    "NL2SQL Engine": "✅ Active" if st.session_state.nl2sql_engine else "❌ Not loaded"
                }
                
                for component, status in components.items():
                    st.write(f"- {component}: {status}")
            
            with col2:
                st.write("**Chat Statistics:**")
                total_messages = len(st.session_state.chat_history)
                user_messages = len([m for m in st.session_state.chat_history if m['type'] == 'user'])
                st.write(f"- Total messages: {total_messages}")
                st.write(f"- User queries: {user_messages}")
                st.write(f"- AI responses: {total_messages - user_messages}")
    
    def run(self):
        """Main application entry point."""
        # Initialize components if not already done
        if not st.session_state.system_initialized:
            asyncio.run(self.initialize_ai_components())
        
        # Render UI components
        self.render_header()
        self.render_sidebar()
        
        # Main interface
        st.divider()
        self.render_main_interface()
        
        # Data explorer (if requested)
        if st.session_state.get('show_explorer', False):
            st.divider()
            st.header("🗺️ Data Explorer")
            data_explorer = DataExplorer(st.session_state.db_manager)
            data_explorer.render()

# Application entry point
def main():
    """Run the FloatChat Streamlit application."""
    app = FloatChatApp()
    app.run()

if __name__ == "__main__":
    main()