"""
Chat Interface Components for FloatChat
Handles message display, visualization, and user interactions
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

class ChatInterface:
    """Manages chat interface components and interactions."""
    
    def __init__(self):
        """Initialize chat interface."""
        self.message_handler = MessageHandler()
        self.visualization_engine = VisualizationEngine()
    
    def display_chat_history(self, messages: List[Dict]):
        """Display complete chat history with formatting."""
        for message in messages:
            self.message_handler.display_message(message)
    
    def get_user_input(self) -> Optional[str]:
        """Get user input with enhanced UI."""
        with st.form("chat_input_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask me about oceanographic data:",
                    placeholder="e.g., What's the average temperature at 500m depth in the Bay of Bengal?",
                    height=80,
                    key="chat_input"
                )
            
            with col2:
                st.write("")  # Spacer
                send_button = st.form_submit_button("🚀 Send", use_container_width=True)
            
            with col3:
                st.write("")  # Spacer
                voice_button = st.form_submit_button("🎤 Voice", use_container_width=True)
            
            if send_button and user_input:
                return user_input
            
            if voice_button:
                st.info("Voice input feature coming soon!")
                return None
        
        return None
    
    def display_quick_actions(self):
        """Display quick action buttons for common queries."""
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🌡️ Temperature Profiles", use_container_width=True):
                return "Show me temperature profiles in the Indian Ocean"
        
        with col2:
            if st.button("🧂 Salinity Analysis", use_container_width=True):
                return "Analyze salinity levels in different regions"
        
        with col3:
            if st.button("💨 Oxygen Levels", use_container_width=True):
                return "Show oxygen concentration below 1000m"
        
        with col4:
            if st.button("📊 Recent Data", use_container_width=True):
                return "What's the latest data from ARGO floats?"
        
        return None

class MessageHandler:
    """Handles individual message display and formatting."""
    
    def display_message(self, message: Dict):
        """Display a single message with appropriate formatting."""
        msg_type = message.get('type', 'unknown')
        timestamp = message.get('timestamp', datetime.now())
        content = message.get('content', '')
        
        # Format timestamp
        time_str = timestamp.strftime("%H:%M:%S") if isinstance(timestamp, datetime) else str(timestamp)
        
        if msg_type == 'user':
            self._display_user_message(content, time_str)
        elif msg_type == 'assistant':
            self._display_ai_message(content, time_str)
        elif msg_type == 'error':
            self._display_error_message(content, time_str)
        elif msg_type == 'system':
            self._display_system_message(content, time_str)
    
    def _display_user_message(self, content: str, timestamp: str):
        """Display user message."""
        with st.chat_message("user"):
            st.markdown(f"**{timestamp}** | {content}")
    
    def _display_ai_message(self, content: Any, timestamp: str):
        """Display AI response message."""
        with st.chat_message("assistant"):
            st.markdown(f"**{timestamp}** | AI Response")
            
            if isinstance(content, dict):
                self._render_structured_response(content)
            else:
                st.write(content)
    
    def _display_error_message(self, content: str, timestamp: str):
        """Display error message."""
        with st.chat_message("assistant"):
            st.error(f"**{timestamp}** | Error: {content}")
    
    def _display_system_message(self, content: str, timestamp: str):
        """Display system message."""
        with st.chat_message("assistant"):
            st.info(f"**{timestamp}** | System: {content}")
    
    def _render_structured_response(self, response: Dict):
        """Render structured AI response with different components."""
        response_type = response.get('type', 'unknown')
        
        if response_type == 'rag_response':
            self._render_rag_response(response)
        elif response_type == 'sql_response':
            self._render_sql_response(response)
        elif response_type == 'hybrid_response':
            self._render_hybrid_response(response)
        elif response_type == 'error':
            st.error(response.get('message', 'Unknown error occurred'))
        else:
            st.json(response)  # Fallback to JSON display
    
    def _render_rag_response(self, response: Dict):
        """Render RAG pipeline response."""
        # Main answer
        answer = response.get('answer', 'No answer provided')
        st.markdown(f"**🧠 Analysis:** {answer}")
        
        # Confidence score
        confidence = response.get('confidence', 0)
        confidence_color = self._get_confidence_color(confidence)
        st.markdown(
            f"**Confidence:** <span style='color: {confidence_color}; font-weight: bold;'>"
            f"{confidence:.1%}</span>", 
            unsafe_allow_html=True
        )
        
        # Sources and metadata
        sources = response.get('sources', [])
        if sources:
            with st.expander("📚 Supporting Data Sources"):
                for i, source in enumerate(sources[:5], 1):  # Limit to top 5
                    st.markdown(f"**Source {i}:**")
                    st.text(source.get('content', '')[:300] + "..." if len(source.get('content', '')) > 300 else source.get('content', ''))
                    if source.get('metadata'):
                        st.caption(f"Relevance: {source.get('score', 0):.3f}")
        
        # Additional metadata
        metadata = response.get('metadata', {})
        if metadata:
            with st.expander("ℹ️ Analysis Metadata"):
                st.json(metadata)
    
    def _render_sql_response(self, response: Dict):
        """Render SQL query response."""
        # SQL query
        sql_query = response.get('sql_query', '')
        if sql_query:
            st.markdown("**💾 Generated SQL Query:**")
            st.code(sql_query, language='sql')
        
        # Query results
        results = response.get('results', [])
        if results:
            df = pd.DataFrame(results)
            
            st.markdown(f"**📊 Results ({len(df)} rows):**")
            st.dataframe(df, use_container_width=True)
            
            # Auto-generate visualizations
            viz_engine = VisualizationEngine()
            viz_engine.auto_visualize_dataframe(df)
        
        # Confidence and metadata
        confidence = response.get('confidence', 0)
        if confidence > 0:
            confidence_color = self._get_confidence_color(confidence)
            st.markdown(
                f"**Query Confidence:** <span style='color: {confidence_color}; font-weight: bold;'>"
                f"{confidence:.1%}</span>", 
                unsafe_allow_html=True
            )
        
        metadata = response.get('metadata', {})
        if metadata:
            with st.expander("📈 Query Performance"):
                if 'estimated_rows' in metadata:
                    st.metric("Estimated Rows", metadata['estimated_rows'])
                if 'performance_notes' in metadata:
                    for note in metadata['performance_notes']:
                        st.info(note)
    
    def _render_hybrid_response(self, response: Dict):
        """Render hybrid analysis response."""
        st.markdown("**🔄 Hybrid Analysis (RAG + SQL)**")
        
        # Combined confidence
        combined_confidence = response.get('combined_confidence', 0)
        confidence_color = self._get_confidence_color(combined_confidence)
        st.markdown(
            f"**Overall Confidence:** <span style='color: {confidence_color}; font-weight: bold;'>"
            f"{combined_confidence:.1%}</span>", 
            unsafe_allow_html=True
        )
        
        # Display both analyses in tabs
        tab1, tab2 = st.tabs(["🧠 RAG Analysis", "💾 SQL Analysis"])
        
        with tab1:
            rag_response = response.get('rag_analysis', {})
            if rag_response.get('type') != 'error':
                self._render_rag_response(rag_response)
            else:
                st.error("RAG analysis failed")
        
        with tab2:
            sql_response = response.get('sql_analysis', {})
            if sql_response.get('type') != 'error':
                self._render_sql_response(sql_response)
            else:
                st.error("SQL analysis failed")
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level."""
        if confidence >= 0.8:
            return "#4CAF50"  # Green
        elif confidence >= 0.6:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red

class VisualizationEngine:
    """Handles data visualization and chart generation."""
    
    def auto_visualize_dataframe(self, df: pd.DataFrame):
        """Automatically create appropriate visualizations for a dataframe."""
        if df.empty:
            st.info("No data to visualize")
            return
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove date columns from categorical
        categorical_cols = [col for col in categorical_cols if col not in date_cols]
        
        if not numeric_cols:
            st.info("No numeric data available for visualization")
            return
        
        st.markdown("**📊 Data Visualizations:**")
        
        # Time series visualization
        if date_cols and numeric_cols:
            self._create_time_series_plot(df, date_cols[0], numeric_cols[:3])
        
        # Distribution plots for numeric columns
        if len(numeric_cols) >= 1:
            self._create_distribution_plots(df, numeric_cols[:4])
        
        # Correlation heatmap for multiple numeric columns
        if len(numeric_cols) >= 2:
            self._create_correlation_heatmap(df, numeric_cols)
        
        # Scatter plots for relationships
        if len(numeric_cols) >= 2:
            self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1], 
                                   categorical_cols[0] if categorical_cols else None)
        
        # Geographic visualization if lat/lon columns exist
        lat_cols = [col for col in df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in df.columns if 'lon' in col.lower()]
        if lat_cols and lon_cols:
            self._create_map_visualization(df, lat_cols[0], lon_cols[0], numeric_cols[0] if numeric_cols else None)
    
    def _create_time_series_plot(self, df: pd.DataFrame, date_col: str, value_cols: List[str]):
        """Create time series plot."""
        try:
            fig = go.Figure()
            
            for col in value_cols:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[date_col],
                        y=df[col],
                        mode='lines+markers',
                        name=col.replace('_', ' ').title()
                    ))
            
            fig.update_layout(
                title="Time Series Analysis",
                xaxis_title=date_col.replace('_', ' ').title(),
                yaxis_title="Values",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create time series plot: {str(e)}")
    
    def _create_distribution_plots(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Create distribution plots for numeric columns."""
        try:
            cols_per_row = 2
            rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
            
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols[:4]):  # Limit to 4 columns
                if col in df.columns:
                    fig.add_trace(go.Histogram(
                        x=df[col],
                        name=col.replace('_', ' ').title(),
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title="Data Distributions",
                xaxis_title="Values",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create distribution plots: {str(e)}")
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Create correlation heatmap."""
        try:
            # Calculate correlations
            corr_df = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=corr_df.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                xaxis_title="Parameters",
                yaxis_title="Parameters"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create correlation heatmap: {str(e)}")
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None):
        """Create scatter plot."""
        try:
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                color=color_col,
                title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
                hover_data=[col for col in df.columns if col in [x_col, y_col, color_col]]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create scatter plot: {str(e)}")
    
    def _create_map_visualization(self, df: pd.DataFrame, lat_col: str, lon_col: str, value_col: Optional[str] = None):
        """Create map visualization for geographic data."""
        try:
            fig = go.Figure()
            
            if value_col:
                fig.add_trace(go.Scattermapbox(
                    lat=df[lat_col],
                    lon=df[lon_col],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df[value_col],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=value_col.replace('_', ' ').title())
                    ),
                    text=[f"{value_col}: {val}" for val in df[value_col]],
                    name="Data Points"
                ))
            else:
                fig.add_trace(go.Scattermapbox(
                    lat=df[lat_col],
                    lon=df[lon_col],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name="Data Points"
                ))
            
            # Center the map on the data
            center_lat = df[lat_col].mean()
            center_lon = df[lon_col].mean()
            
            fig.update_layout(
                title="Geographic Distribution",
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=4
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to create map visualization: {str(e)}")