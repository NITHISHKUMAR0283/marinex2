"""
Data Explorer Components for FloatChat
Provides interactive data exploration and visualization capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio

class DataExplorer:
    """Interactive data exploration interface."""
    
    def __init__(self, db_manager):
        """Initialize data explorer with database manager."""
        self.db_manager = db_manager
        self.query_visualizer = QueryVisualizer()
    
    def render(self):
        """Render the complete data explorer interface."""
        st.subheader("🗺️ Oceanographic Data Explorer")
        
        # Explorer tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌊 Quick Stats", 
            "📊 Custom Query", 
            "🗺️ Geographic View", 
            "📈 Trend Analysis"
        ])
        
        with tab1:
            self._render_quick_stats()
        
        with tab2:
            self._render_custom_query()
        
        with tab3:
            self._render_geographic_view()
        
        with tab4:
            self._render_trend_analysis()
    
    def _render_quick_stats(self):
        """Render quick statistics dashboard."""
        st.write("### 📊 System Overview")
        
        # Get basic statistics
        try:
            stats = asyncio.run(self._get_system_stats())
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Floats",
                    stats.get('total_floats', 0),
                    delta=f"+{stats.get('active_floats', 0)} active"
                )
            
            with col2:
                st.metric(
                    "Total Profiles", 
                    f"{stats.get('total_profiles', 0):,}",
                    delta=f"Avg {stats.get('profiles_per_float', 0):.1f}/float"
                )
            
            with col3:
                st.metric(
                    "Total Measurements",
                    f"{stats.get('total_measurements', 0):,}",
                    delta=f"Max depth: {stats.get('max_depth', 0)}m"
                )
            
            with col4:
                st.metric(
                    "Date Range",
                    stats.get('date_range_days', 0),
                    delta=f"Updated: {stats.get('last_update', 'Unknown')}"
                )
            
            # Parameter distribution
            st.write("### 🌡️ Parameter Availability")
            param_stats = stats.get('parameter_stats', {})
            
            if param_stats:
                param_df = pd.DataFrame([
                    {'Parameter': param, 'Count': count, 'Percentage': (count / stats.get('total_measurements', 1)) * 100}
                    for param, count in param_stats.items()
                ])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        param_df, 
                        x='Parameter', 
                        y='Count',
                        title="Measurements by Parameter",
                        text='Count'
                    )
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(param_df, use_container_width=True)
            
            # Geographic distribution
            st.write("### 🗺️ Geographic Coverage")
            geo_stats = asyncio.run(self._get_geographic_stats())
            
            if geo_stats:
                fig = go.Figure()
                
                fig.add_trace(go.Scattermapbox(
                    lat=geo_stats['latitudes'],
                    lon=geo_stats['longitudes'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=geo_stats['measurement_counts'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Measurements")
                    ),
                    text=[f"Float {fid}: {count} measurements" 
                          for fid, count in zip(geo_stats['float_ids'], geo_stats['measurement_counts'])],
                    name="ARGO Floats"
                ))
                
                fig.update_layout(
                    title="ARGO Float Positions",
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(lat=0, lon=75),  # Indian Ocean center
                        zoom=3
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load system statistics: {str(e)}")
    
    def _render_custom_query(self):
        """Render custom query interface."""
        st.write("### 💾 Custom Data Query")
        
        # Query builder interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Parameter selection
            available_params = ['temperature', 'salinity', 'pressure', 'oxygen', 'ph', 'nitrate']
            selected_params = st.multiselect(
                "Select Parameters:",
                available_params,
                default=['temperature', 'salinity']
            )
            
            # Depth range
            depth_range = st.slider(
                "Depth Range (m):",
                0, 2000, (0, 500)
            )
            
            # Date range
            date_range = st.date_input(
                "Date Range:",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
            
            # Geographic bounds
            st.write("**Geographic Bounds:**")
            lat_range = st.slider("Latitude Range:", -90.0, 90.0, (-20.0, 30.0))
            lon_range = st.slider("Longitude Range:", -180.0, 180.0, (40.0, 100.0))
        
        with col2:
            st.write("**Query Options:**")
            
            limit_results = st.checkbox("Limit Results", value=True)
            max_results = st.number_input("Max Results:", 1, 10000, 1000) if limit_results else None
            
            include_metadata = st.checkbox("Include Metadata", value=True)
            aggregate_data = st.checkbox("Aggregate by Float", value=False)
            
            # Execute query button
            if st.button("🚀 Execute Query", type="primary", use_container_width=True):
                self._execute_custom_query(
                    selected_params, depth_range, date_range, 
                    lat_range, lon_range, max_results, 
                    include_metadata, aggregate_data
                )
    
    def _render_geographic_view(self):
        """Render geographic data visualization."""
        st.write("### 🗺️ Geographic Data Visualization")
        
        # Parameter and visualization options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            viz_param = st.selectbox(
                "Visualization Parameter:",
                ['temperature', 'salinity', 'oxygen', 'pressure'],
                index=0
            )
        
        with col2:
            depth_level = st.selectbox(
                "Depth Level:",
                ['Surface (0-50m)', 'Thermocline (50-200m)', 'Intermediate (200-1000m)', 'Deep (1000m+)'],
                index=0
            )
        
        with col3:
            time_period = st.selectbox(
                "Time Period:",
                ['Last 7 days', 'Last 30 days', 'Last 3 months', 'All time'],
                index=1
            )
        
        # Generate visualization
        if st.button("🗺️ Generate Map", type="primary"):
            self._generate_geographic_visualization(viz_param, depth_level, time_period)
    
    def _render_trend_analysis(self):
        """Render trend analysis interface."""
        st.write("### 📈 Temporal Trend Analysis")
        
        # Analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            trend_params = st.multiselect(
                "Parameters for Analysis:",
                ['temperature', 'salinity', 'oxygen', 'ph'],
                default=['temperature', 'salinity']
            )
            
            analysis_region = st.selectbox(
                "Analysis Region:",
                ['Entire Indian Ocean', 'Bay of Bengal', 'Arabian Sea', 'Southern Indian Ocean'],
                index=0
            )
        
        with col2:
            trend_period = st.selectbox(
                "Analysis Period:",
                ['Last 6 months', 'Last year', 'Last 2 years', 'All available data'],
                index=2
            )
            
            analysis_depth = st.selectbox(
                "Depth Focus:",
                ['Mixed Layer (0-100m)', 'Thermocline (100-500m)', 'Deep Water (500m+)', 'Full Profile'],
                index=0
            )
        
        # Generate trend analysis
        if st.button("📊 Analyze Trends", type="primary"):
            self._generate_trend_analysis(trend_params, analysis_region, trend_period, analysis_depth)
    
    async def _get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Total floats
            float_query = "SELECT COUNT(DISTINCT wmo_id) as total_floats FROM measurements"
            float_result = await self.db_manager.execute_query(float_query)
            total_floats = float_result[0]['total_floats'] if float_result else 0
            
            # Total profiles
            profile_query = "SELECT COUNT(DISTINCT profile_id) as total_profiles FROM measurements"
            profile_result = await self.db_manager.execute_query(profile_query)
            total_profiles = profile_result[0]['total_profiles'] if profile_result else 0
            
            # Total measurements
            measurement_query = "SELECT COUNT(*) as total_measurements FROM measurements"
            measurement_result = await self.db_manager.execute_query(measurement_query)
            total_measurements = measurement_result[0]['total_measurements'] if measurement_result else 0
            
            # Parameter statistics
            param_query = """
                SELECT 
                    SUM(CASE WHEN temperature IS NOT NULL THEN 1 ELSE 0 END) as temperature,
                    SUM(CASE WHEN salinity IS NOT NULL THEN 1 ELSE 0 END) as salinity,
                    SUM(CASE WHEN pressure IS NOT NULL THEN 1 ELSE 0 END) as pressure,
                    SUM(CASE WHEN oxygen IS NOT NULL THEN 1 ELSE 0 END) as oxygen
                FROM measurements
            """
            param_result = await self.db_manager.execute_query(param_query)
            parameter_stats = param_result[0] if param_result else {}
            
            # Date range
            date_query = """
                SELECT 
                    MIN(measurement_date) as min_date,
                    MAX(measurement_date) as max_date
                FROM measurements
            """
            date_result = await self.db_manager.execute_query(date_query)
            date_info = date_result[0] if date_result else {}
            
            # Calculate date range
            date_range_days = 0
            if date_info.get('min_date') and date_info.get('max_date'):
                min_date = date_info['min_date']
                max_date = date_info['max_date']
                if isinstance(min_date, str):
                    min_date = datetime.fromisoformat(min_date.replace('Z', '+00:00'))
                if isinstance(max_date, str):
                    max_date = datetime.fromisoformat(max_date.replace('Z', '+00:00'))
                date_range_days = (max_date - min_date).days
            
            # Max depth
            depth_query = "SELECT MAX(depth) as max_depth FROM measurements"
            depth_result = await self.db_manager.execute_query(depth_query)
            max_depth = depth_result[0]['max_depth'] if depth_result else 0
            
            return {
                'total_floats': total_floats,
                'total_profiles': total_profiles,
                'total_measurements': total_measurements,
                'parameter_stats': parameter_stats,
                'date_range_days': date_range_days,
                'max_depth': max_depth,
                'profiles_per_float': total_profiles / max(total_floats, 1),
                'last_update': date_info.get('max_date', 'Unknown')
            }
            
        except Exception as e:
            st.error(f"Error getting system stats: {str(e)}")
            return {}
    
    async def _get_geographic_stats(self) -> Dict[str, List]:
        """Get geographic distribution statistics."""
        try:
            geo_query = """
                SELECT 
                    wmo_id,
                    AVG(latitude) as avg_lat,
                    AVG(longitude) as avg_lon,
                    COUNT(*) as measurement_count
                FROM measurements 
                GROUP BY wmo_id
                HAVING AVG(latitude) IS NOT NULL AND AVG(longitude) IS NOT NULL
            """
            
            results = await self.db_manager.execute_query(geo_query)
            
            if not results:
                return {}
            
            return {
                'float_ids': [r['wmo_id'] for r in results],
                'latitudes': [r['avg_lat'] for r in results],
                'longitudes': [r['avg_lon'] for r in results],
                'measurement_counts': [r['measurement_count'] for r in results]
            }
            
        except Exception as e:
            st.error(f"Error getting geographic stats: {str(e)}")
            return {}
    
    def _execute_custom_query(self, params, depth_range, date_range, lat_range, lon_range, 
                            max_results, include_metadata, aggregate_data):
        """Execute custom query with specified parameters."""
        try:
            with st.spinner("Executing custom query..."):
                # Build SQL query
                select_columns = ['wmo_id', 'profile_id', 'depth', 'measurement_date', 'latitude', 'longitude']
                select_columns.extend(params)
                
                query = f"SELECT {', '.join(select_columns)} FROM measurements WHERE 1=1"
                query_params = []
                
                # Add depth filter
                query += " AND depth BETWEEN %s AND %s"
                query_params.extend([depth_range[0], depth_range[1]])
                
                # Add date filter
                if len(date_range) == 2:
                    query += " AND measurement_date BETWEEN %s AND %s"
                    query_params.extend([date_range[0], date_range[1]])
                
                # Add geographic filter
                query += " AND latitude BETWEEN %s AND %s AND longitude BETWEEN %s AND %s"
                query_params.extend([lat_range[0], lat_range[1], lon_range[0], lon_range[1]])
                
                # Add parameter filters (only non-null)
                param_conditions = " OR ".join([f"{param} IS NOT NULL" for param in params])
                query += f" AND ({param_conditions})"
                
                # Add limit
                if max_results:
                    query += f" LIMIT {max_results}"
                
                # Execute query
                results = asyncio.run(self.db_manager.execute_query(query, query_params))
                
                if not results:
                    st.warning("No data found matching the specified criteria.")
                    return
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                st.success(f"Retrieved {len(df)} records")
                
                # Display results
                self._display_query_results(df, params, aggregate_data, include_metadata)
                
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
    
    def _display_query_results(self, df: pd.DataFrame, params: List[str], 
                              aggregate_data: bool, include_metadata: bool):
        """Display query results with visualizations."""
        # Data preview
        st.write("### 📋 Query Results")
        st.dataframe(df.head(100), use_container_width=True)  # Show first 100 rows
        
        # Summary statistics
        if params:
            st.write("### 📊 Summary Statistics")
            param_df = df[params].describe()
            st.dataframe(param_df, use_container_width=True)
        
        # Visualizations
        if len(df) > 0:
            self.query_visualizer.visualize_results(df, params)
        
        # Metadata
        if include_metadata:
            st.write("### ℹ️ Query Metadata")
            metadata = {
                'Total Records': len(df),
                'Unique Floats': df['wmo_id'].nunique() if 'wmo_id' in df.columns else 'N/A',
                'Unique Profiles': df['profile_id'].nunique() if 'profile_id' in df.columns else 'N/A',
                'Date Range': f"{df['measurement_date'].min()} to {df['measurement_date'].max()}" if 'measurement_date' in df.columns else 'N/A',
                'Depth Range': f"{df['depth'].min():.1f}m to {df['depth'].max():.1f}m" if 'depth' in df.columns else 'N/A'
            }
            st.json(metadata)
    
    def _generate_geographic_visualization(self, param: str, depth_level: str, time_period: str):
        """Generate geographic visualization for specified parameters."""
        try:
            with st.spinner("Generating geographic visualization..."):
                # Convert depth level to actual depth range
                depth_mapping = {
                    'Surface (0-50m)': (0, 50),
                    'Thermocline (50-200m)': (50, 200),
                    'Intermediate (200-1000m)': (200, 1000),
                    'Deep (1000m+)': (1000, 2000)
                }
                depth_range = depth_mapping.get(depth_level, (0, 500))
                
                # Convert time period to date range
                now = datetime.now()
                time_mapping = {
                    'Last 7 days': now - timedelta(days=7),
                    'Last 30 days': now - timedelta(days=30),
                    'Last 3 months': now - timedelta(days=90),
                    'All time': datetime(2020, 1, 1)
                }
                start_date = time_mapping.get(time_period, now - timedelta(days=30))
                
                # Build and execute query
                query = f"""
                    SELECT 
                        wmo_id,
                        AVG(latitude) as latitude,
                        AVG(longitude) as longitude,
                        AVG({param}) as avg_{param},
                        COUNT(*) as measurement_count
                    FROM measurements 
                    WHERE {param} IS NOT NULL 
                        AND depth BETWEEN %s AND %s
                        AND measurement_date >= %s
                    GROUP BY wmo_id
                    HAVING AVG(latitude) IS NOT NULL AND AVG(longitude) IS NOT NULL
                """
                
                results = asyncio.run(self.db_manager.execute_query(
                    query, [depth_range[0], depth_range[1], start_date]
                ))
                
                if not results:
                    st.warning("No data available for the specified criteria.")
                    return
                
                # Create visualization
                df = pd.DataFrame(results)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=dict(
                        size=np.sqrt(df['measurement_count']) * 2,
                        color=df[f'avg_{param}'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=f"Average {param.title()}")
                    ),
                    text=[f"Float {fid}<br>{param.title()}: {val:.2f}<br>Measurements: {count}"
                          for fid, val, count in zip(df['wmo_id'], df[f'avg_{param}'], df['measurement_count'])],
                    name=f"{param.title()} Distribution"
                ))
                
                fig.update_layout(
                    title=f"{param.title()} Distribution - {depth_level} ({time_period})",
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(lat=0, lon=75),
                        zoom=3
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.write("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Active Floats", len(df))
                with col2:
                    st.metric(f"Avg {param.title()}", f"{df[f'avg_{param}'].mean():.2f}")
                with col3:
                    st.metric(f"Min {param.title()}", f"{df[f'avg_{param}'].min():.2f}")
                with col4:
                    st.metric(f"Max {param.title()}", f"{df[f'avg_{param}'].max():.2f}")
                
        except Exception as e:
            st.error(f"Failed to generate geographic visualization: {str(e)}")
    
    def _generate_trend_analysis(self, params: List[str], region: str, period: str, depth: str):
        """Generate temporal trend analysis."""
        try:
            with st.spinner("Analyzing temporal trends..."):
                # This would implement comprehensive trend analysis
                # For now, showing a placeholder
                st.info("🚧 Advanced trend analysis is being implemented. This will include:")
                st.write("- Seasonal variation analysis")
                st.write("- Long-term trend detection") 
                st.write("- Anomaly identification")
                st.write("- Cross-parameter correlations")
                st.write("- Regional comparison")
                
                # Placeholder visualization
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                fig = go.Figure()
                
                for param in params:
                    # Generate sample trend data
                    trend_data = np.random.randn(len(dates)).cumsum() + 20
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=trend_data,
                        mode='lines',
                        name=param.title()
                    ))
                
                fig.update_layout(
                    title=f"Trend Analysis - {region} ({period})",
                    xaxis_title="Date",
                    yaxis_title="Parameter Values"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to generate trend analysis: {str(e)}")

class QueryVisualizer:
    """Specialized visualizations for query results."""
    
    def visualize_results(self, df: pd.DataFrame, params: List[str]):
        """Create comprehensive visualizations for query results."""
        st.write("### 📊 Data Visualizations")
        
        # Parameter trends over depth
        if 'depth' in df.columns and params:
            self._create_depth_profiles(df, params)
        
        # Time series if date column exists
        if 'measurement_date' in df.columns and params:
            self._create_time_series(df, params)
        
        # Geographic scatter if lat/lon exists
        if 'latitude' in df.columns and 'longitude' in df.columns:
            self._create_geographic_scatter(df, params)
    
    def _create_depth_profiles(self, df: pd.DataFrame, params: List[str]):
        """Create depth profile visualizations."""
        fig = make_subplots(
            rows=1, 
            cols=len(params),
            subplot_titles=[param.title() for param in params],
            shared_yaxes=True
        )
        
        for i, param in enumerate(params, 1):
            if param in df.columns:
                # Average by depth bins
                depth_bins = np.arange(0, df['depth'].max() + 100, 50)
                df['depth_bin'] = pd.cut(df['depth'], bins=depth_bins)
                depth_avg = df.groupby('depth_bin')[param].mean().reset_index()
                depth_avg['depth_center'] = depth_avg['depth_bin'].apply(lambda x: x.mid)
                
                fig.add_trace(
                    go.Scatter(
                        x=depth_avg[param],
                        y=depth_avg['depth_center'],
                        mode='lines+markers',
                        name=param.title()
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title="Average Profiles by Depth",
            height=500
        )
        
        # Invert y-axis for depth
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_time_series(self, df: pd.DataFrame, params: List[str]):
        """Create time series visualizations."""
        # Convert date column
        df['measurement_date'] = pd.to_datetime(df['measurement_date'])
        
        # Aggregate by date
        daily_avg = df.groupby('measurement_date')[params].mean().reset_index()
        
        fig = go.Figure()
        
        for param in params:
            if param in daily_avg.columns:
                fig.add_trace(go.Scatter(
                    x=daily_avg['measurement_date'],
                    y=daily_avg[param],
                    mode='lines+markers',
                    name=param.title()
                ))
        
        fig.update_layout(
            title="Parameter Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Parameter Values"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_geographic_scatter(self, df: pd.DataFrame, params: List[str]):
        """Create geographic scatter plot."""
        if params and params[0] in df.columns:
            fig = px.scatter_mapbox(
                df.sample(min(1000, len(df))),  # Sample for performance
                lat='latitude',
                lon='longitude',
                color=params[0],
                size_max=10,
                color_continuous_scale='Viridis',
                title=f"Geographic Distribution - {params[0].title()}"
            )
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=0, lon=75), zoom=3),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)