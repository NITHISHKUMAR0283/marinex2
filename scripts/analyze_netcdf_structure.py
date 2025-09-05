#!/usr/bin/env python3
"""
ARGO NetCDF File Structure Analyzer

This script performs comprehensive analysis of ARGO NetCDF files to understand:
1. Data dimensions, variables, and attributes
2. Quality control flags and their meanings
3. Coordinate systems and temporal structures
4. Data patterns and relationships
5. Optimal database schema design for LLM queries

Usage: python scripts/analyze_netcdf_structure.py [netcdf_file_path]
"""

import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


class ARGONetCDFAnalyzer:
    """Comprehensive ARGO NetCDF file analyzer."""
    
    def __init__(self, file_path: Path):
        """Initialize analyzer with NetCDF file path."""
        self.file_path = Path(file_path)
        self.dataset: Optional[xr.Dataset] = None
        self.analysis_results: Dict[str, Any] = {}
        
    def load_dataset(self) -> None:
        """Load NetCDF dataset with comprehensive error handling."""
        try:
            rprint(f"[bold blue]Loading NetCDF file:[/bold blue] {self.file_path}")
            self.dataset = xr.open_dataset(self.file_path)
            rprint("[green]✓ Dataset loaded successfully[/green]")
        except Exception as e:
            rprint(f"[red]✗ Error loading dataset: {e}[/red]")
            sys.exit(1)
    
    def analyze_dimensions(self) -> Dict[str, Any]:
        """Analyze dataset dimensions and their relationships."""
        rprint("\n[bold cyan]📏 ANALYZING DIMENSIONS[/bold cyan]")
        
        dimensions_info = {}
        
        for dim_name, dim_size in self.dataset.dims.items():
            dimensions_info[dim_name] = {
                'size': int(dim_size),
                'description': self._get_dimension_description(dim_name),
                'is_unlimited': dim_name in self.dataset.dims and self.dataset.dims[dim_name] is None
            }
        
        # Create dimensions table
        table = Table(title="Dataset Dimensions")
        table.add_column("Dimension", style="cyan")
        table.add_column("Size", justify="right", style="green")
        table.add_column("Description", style="yellow")
        
        for dim_name, info in dimensions_info.items():
            table.add_row(
                dim_name,
                str(info['size']),
                info['description']
            )
        
        console.print(table)
        
        self.analysis_results['dimensions'] = dimensions_info
        return dimensions_info
    
    def analyze_variables(self) -> Dict[str, Any]:
        """Analyze all variables in the dataset."""
        rprint("\n[bold cyan]🔬 ANALYZING VARIABLES[/bold cyan]")
        
        variables_info = {}
        
        for var_name, variable in self.dataset.variables.items():
            var_info = {
                'dimensions': list(variable.dims),
                'shape': list(variable.shape),
                'dtype': str(variable.dtype),
                'attributes': dict(variable.attrs),
                'has_fill_value': '_FillValue' in variable.attrs,
                'fill_value': variable.attrs.get('_FillValue'),
                'units': variable.attrs.get('units', 'unknown'),
                'long_name': variable.attrs.get('long_name', var_name),
                'standard_name': variable.attrs.get('standard_name', ''),
                'valid_min': variable.attrs.get('valid_min'),
                'valid_max': variable.attrs.get('valid_max'),
                'data_statistics': self._calculate_variable_statistics(variable)
            }
            
            variables_info[var_name] = var_info
        
        # Categorize variables by type
        categorized_vars = self._categorize_variables(variables_info)
        
        # Display categorized variables
        for category, vars_list in categorized_vars.items():
            if vars_list:
                rprint(f"\n[bold yellow]{category.upper()} VARIABLES:[/bold yellow]")
                
                table = Table()
                table.add_column("Variable", style="cyan")
                table.add_column("Dimensions", style="blue")
                table.add_column("Shape", style="green")
                table.add_column("Units", style="yellow")
                table.add_column("Description", style="dim")
                
                for var_name in vars_list:
                    var_info = variables_info[var_name]
                    table.add_row(
                        var_name,
                        str(var_info['dimensions']),
                        str(var_info['shape']),
                        var_info['units'],
                        var_info['long_name'][:50] + "..." if len(var_info['long_name']) > 50 else var_info['long_name']
                    )
                
                console.print(table)
        
        self.analysis_results['variables'] = variables_info
        self.analysis_results['variable_categories'] = categorized_vars
        return variables_info
    
    def analyze_coordinates(self) -> Dict[str, Any]:
        """Analyze coordinate variables and systems."""
        rprint("\n[bold cyan]🌐 ANALYZING COORDINATES[/bold cyan]")
        
        coordinate_info = {}
        
        # Identify coordinate variables
        coord_vars = ['LATITUDE', 'LONGITUDE', 'JULD', 'PRES', 'DEPH']
        
        for coord_name in coord_vars:
            if coord_name in self.dataset.variables:
                coord_var = self.dataset.variables[coord_name]
                
                coordinate_info[coord_name] = {
                    'values_sample': self._get_sample_values(coord_var),
                    'range': self._get_value_range(coord_var),
                    'units': coord_var.attrs.get('units', 'unknown'),
                    'reference': coord_var.attrs.get('reference', ''),
                    'axis': coord_var.attrs.get('axis', ''),
                    'statistics': self._calculate_variable_statistics(coord_var)
                }
        
        # Create coordinates table
        table = Table(title="Coordinate Variables Analysis")
        table.add_column("Coordinate", style="cyan")
        table.add_column("Range", style="green")
        table.add_column("Units", style="yellow")
        table.add_column("Reference", style="blue")
        table.add_column("Sample Values", style="dim")
        
        for coord_name, info in coordinate_info.items():
            sample_vals = ', '.join([f"{v:.3f}" if isinstance(v, float) else str(v) 
                                    for v in info['values_sample'][:3]])
            table.add_row(
                coord_name,
                f"{info['range'][0]:.3f} to {info['range'][1]:.3f}" if info['range'][0] is not None else "N/A",
                info['units'],
                info['reference'][:20] + "..." if len(info['reference']) > 20 else info['reference'],
                sample_vals
            )
        
        console.print(table)
        
        self.analysis_results['coordinates'] = coordinate_info
        return coordinate_info
    
    def analyze_quality_flags(self) -> Dict[str, Any]:
        """Analyze quality control flags and their meanings."""
        rprint("\n[bold cyan]🏁 ANALYZING QUALITY FLAGS[/bold cyan]")
        
        qc_info = {}
        qc_variables = [var for var in self.dataset.variables if '_QC' in var or var.endswith('_FLAG')]
        
        # ARGO QC flag meanings (standard)
        qc_flag_meanings = {
            '0': 'No QC was performed',
            '1': 'Good data',
            '2': 'Probably good data', 
            '3': 'Bad data that are potentially correctable',
            '4': 'Bad data',
            '5': 'Value changed',
            '6': 'Not used',
            '7': 'Not used',
            '8': 'Estimated value',
            '9': 'Missing value'
        }
        
        for qc_var_name in qc_variables:
            qc_var = self.dataset.variables[qc_var_name]
            
            # Get unique flag values
            unique_flags = np.unique(qc_var.values.flatten())
            unique_flags = unique_flags[~np.isnan(unique_flags.astype(float))]
            
            qc_info[qc_var_name] = {
                'unique_values': [str(int(f)) for f in unique_flags if not np.isnan(f)],
                'distribution': self._calculate_flag_distribution(qc_var),
                'dimensions': list(qc_var.dims),
                'related_variable': qc_var_name.replace('_QC', '').replace('_FLAG', ''),
                'flag_meanings': {str(int(f)): qc_flag_meanings.get(str(int(f)), 'Unknown') 
                                for f in unique_flags if not np.isnan(f)}
            }
        
        # Display QC analysis
        if qc_info:
            table = Table(title="Quality Control Flags Analysis")
            table.add_column("QC Variable", style="cyan")
            table.add_column("Related Variable", style="blue")
            table.add_column("Unique Flags", style="green")
            table.add_column("Flag Distribution", style="yellow")
            
            for qc_var, info in qc_info.items():
                flags_str = ', '.join(info['unique_values'])
                dist_str = ', '.join([f"{k}:{v}" for k, v in list(info['distribution'].items())[:3]])
                
                table.add_row(
                    qc_var,
                    info['related_variable'],
                    flags_str,
                    dist_str + "..." if len(info['distribution']) > 3 else dist_str
                )
            
            console.print(table)
            
            # Display flag meanings
            rprint("\n[bold yellow]QC Flag Meanings:[/bold yellow]")
            for flag, meaning in qc_flag_meanings.items():
                color = {
                    '1': 'green', '2': 'green', '5': 'yellow', '8': 'yellow',
                    '3': 'red', '4': 'red', '0': 'blue', '9': 'red'
                }.get(flag, 'white')
                rprint(f"[{color}]{flag}:[/{color}] {meaning}")
        
        self.analysis_results['quality_flags'] = qc_info
        return qc_info
    
    def analyze_global_attributes(self) -> Dict[str, Any]:
        """Analyze global attributes and metadata."""
        rprint("\n[bold cyan]🌍 ANALYZING GLOBAL ATTRIBUTES[/bold cyan]")
        
        global_attrs = dict(self.dataset.attrs)
        
        # Categorize attributes
        important_attrs = {
            'identification': ['title', 'institution', 'source', 'references', 'comment'],
            'temporal': ['date_creation', 'date_update', 'time_coverage_start', 'time_coverage_end'],
            'spatial': ['geospatial_lat_min', 'geospatial_lat_max', 'geospatial_lon_min', 'geospatial_lon_max'],
            'technical': ['format_version', 'handbook_version', 'data_type', 'platform_number'],
            'quality': ['data_mode', 'processing_version', 'quality_control_indicator']
        }
        
        categorized_attrs = {category: {} for category in important_attrs.keys()}
        other_attrs = {}
        
        for attr_name, attr_value in global_attrs.items():
            found_category = None
            for category, attr_list in important_attrs.items():
                if any(key_attr in attr_name.lower() for key_attr in attr_list):
                    categorized_attrs[category][attr_name] = attr_value
                    found_category = True
                    break
            
            if not found_category:
                other_attrs[attr_name] = attr_value
        
        # Display important attributes by category
        for category, attrs in categorized_attrs.items():
            if attrs:
                rprint(f"\n[bold yellow]{category.upper()} ATTRIBUTES:[/bold yellow]")
                for attr_name, attr_value in attrs.items():
                    attr_str = str(attr_value)[:100] + "..." if len(str(attr_value)) > 100 else str(attr_value)
                    rprint(f"[cyan]{attr_name}:[/cyan] {attr_str}")
        
        self.analysis_results['global_attributes'] = {
            'categorized': categorized_attrs,
            'other': other_attrs,
            'total_count': len(global_attrs)
        }
        
        return global_attrs
    
    def analyze_data_patterns(self) -> Dict[str, Any]:
        """Analyze data patterns and relationships for schema design."""
        rprint("\n[bold cyan]📊 ANALYZING DATA PATTERNS[/bold cyan]")
        
        patterns = {}
        
        # Analyze profile structure
        if 'N_PROF' in self.dataset.dims:
            n_profiles = self.dataset.dims['N_PROF']
            patterns['profile_structure'] = {
                'total_profiles': n_profiles,
                'profile_dimension': 'N_PROF'
            }
            
            if 'N_LEVELS' in self.dataset.dims:
                n_levels = self.dataset.dims['N_LEVELS']
                patterns['level_structure'] = {
                    'max_levels_per_profile': n_levels,
                    'level_dimension': 'N_LEVELS'
                }
        
        # Analyze parameter availability
        core_params = ['TEMP', 'PSAL', 'PRES']  # Temperature, Salinity, Pressure
        bgc_params = ['DOXY', 'CHLA', 'BBP700', 'PH_IN_SITU_TOTAL', 'NITRATE']  # BGC parameters
        
        available_params = {
            'core': [p for p in core_params if p in self.dataset.variables],
            'bgc': [p for p in bgc_params if p in self.dataset.variables],
            'other': [v for v in self.dataset.variables 
                     if v not in core_params + bgc_params + ['LATITUDE', 'LONGITUDE', 'JULD']]
        }
        
        patterns['available_parameters'] = available_params
        
        # Analyze temporal patterns
        if 'JULD' in self.dataset.variables:
            juld_var = self.dataset.variables['JULD']
            time_stats = self._analyze_temporal_patterns(juld_var)
            patterns['temporal_patterns'] = time_stats
        
        # Display patterns
        rprint("\n[bold yellow]DATA STRUCTURE PATTERNS:[/bold yellow]")
        
        if 'profile_structure' in patterns:
            rprint(f"[green]• Total Profiles:[/green] {patterns['profile_structure']['total_profiles']}")
        
        if 'level_structure' in patterns:
            rprint(f"[green]• Max Levels per Profile:[/green] {patterns['level_structure']['max_levels_per_profile']}")
        
        rprint(f"[green]• Core Parameters:[/green] {', '.join(available_params['core'])}")
        rprint(f"[green]• BGC Parameters:[/green] {', '.join(available_params['bgc'])}")
        
        self.analysis_results['data_patterns'] = patterns
        return patterns
    
    def design_optimal_schema(self) -> Dict[str, Any]:
        """Design optimal database schema based on analysis for LLM queries."""
        rprint("\n[bold cyan]🏗️ DESIGNING OPTIMAL DATABASE SCHEMA[/bold cyan]")
        
        schema_design = {
            'tables': {},
            'indexes': [],
            'views': [],
            'partitioning_strategy': {},
            'query_optimization': {}
        }
        
        # 1. Float/Platform table (one record per float)
        schema_design['tables']['floats'] = {
            'description': 'ARGO float metadata and deployment information',
            'columns': {
                'float_id': 'VARCHAR(20) PRIMARY KEY',
                'wmo_number': 'INTEGER UNIQUE',
                'platform_number': 'VARCHAR(20)',
                'program_name': 'VARCHAR(100)',
                'institution': 'VARCHAR(200)',
                'deployment_date': 'TIMESTAMP',
                'deployment_latitude': 'DECIMAL(8,5)',
                'deployment_longitude': 'DECIMAL(8,5)',
                'float_type': 'VARCHAR(50)',
                'sensor_configuration': 'JSONB',
                'current_status': 'VARCHAR(50)',
                'last_location_date': 'TIMESTAMP',
                'last_location_lat': 'DECIMAL(8,5)',
                'last_location_lon': 'DECIMAL(8,5)',
                'total_profiles': 'INTEGER DEFAULT 0',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            },
            'indexes': [
                'CREATE INDEX idx_floats_wmo ON floats(wmo_number)',
                'CREATE INDEX idx_floats_deployment_location ON floats USING GIST(ST_Point(deployment_longitude, deployment_latitude))',
                'CREATE INDEX idx_floats_deployment_date ON floats(deployment_date)',
                'CREATE INDEX idx_floats_program ON floats(program_name)',
                'CREATE INDEX idx_floats_status ON floats(current_status)'
            ]
        }
        
        # 2. Profiles table (one record per profile/cycle)
        schema_design['tables']['profiles'] = {
            'description': 'ARGO profile metadata and location information',
            'columns': {
                'profile_id': 'SERIAL PRIMARY KEY',
                'float_id': 'VARCHAR(20) REFERENCES floats(float_id) ON DELETE CASCADE',
                'cycle_number': 'INTEGER',
                'profile_date': 'TIMESTAMP',
                'location': 'GEOGRAPHY(POINT, 4326)',
                'latitude': 'DECIMAL(8,5)',
                'longitude': 'DECIMAL(8,5)',
                'profile_type': 'VARCHAR(20)',  # 'primary', 'secondary', 'intermediate'
                'direction': 'CHAR(1)',  # 'A' for ascending, 'D' for descending
                'data_mode': 'CHAR(1)',  # 'R' for real-time, 'A' for adjusted, 'D' for delayed
                'positioning_system': 'VARCHAR(20)',
                'quality_flag': 'INTEGER',
                'measurement_count': 'INTEGER DEFAULT 0',
                'max_pressure': 'DECIMAL(8,2)',
                'processing_version': 'VARCHAR(20)',
                'file_path': 'TEXT',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            },
            'partitioning': 'PARTITION BY RANGE (profile_date)',
            'indexes': [
                'CREATE INDEX idx_profiles_float_cycle ON profiles(float_id, cycle_number)',
                'CREATE INDEX idx_profiles_location ON profiles USING GIST(location)',
                'CREATE INDEX idx_profiles_date ON profiles(profile_date)',
                'CREATE INDEX idx_profiles_spatial_temporal ON profiles USING GIST(location, profile_date)',
                'CREATE INDEX idx_profiles_data_mode ON profiles(data_mode)',
                'CREATE INDEX idx_profiles_quality ON profiles(quality_flag)'
            ]
        }
        
        # 3. Measurements table (optimized for massive scale)
        schema_design['tables']['measurements'] = {
            'description': 'ARGO measurement data with optimized storage',
            'columns': {
                'measurement_id': 'BIGSERIAL PRIMARY KEY',
                'profile_id': 'INTEGER REFERENCES profiles(profile_id) ON DELETE CASCADE',
                'pressure': 'DECIMAL(8,2)',
                'depth': 'DECIMAL(8,2)',
                'temperature': 'DECIMAL(6,3)',
                'temperature_qc': 'SMALLINT',
                'salinity': 'DECIMAL(7,4)',
                'salinity_qc': 'SMALLINT',
                'dissolved_oxygen': 'DECIMAL(8,3)',
                'dissolved_oxygen_qc': 'SMALLINT',
                'chlorophyll_a': 'DECIMAL(8,4)',
                'chlorophyll_a_qc': 'SMALLINT',
                'ph': 'DECIMAL(5,3)',
                'ph_qc': 'SMALLINT',
                'nitrate': 'DECIMAL(7,3)',
                'nitrate_qc': 'SMALLINT',
                'backscattering': 'DECIMAL(10,6)',
                'backscattering_qc': 'SMALLINT',
                'additional_params': 'JSONB',  # For flexible parameter storage
                'quality_flags': 'JSONB'       # Consolidated QC information
            },
            'partitioning': 'PARTITION BY RANGE (profile_id)',  # Partition by profile ranges
            'indexes': [
                'CREATE INDEX idx_measurements_profile ON measurements(profile_id)',
                'CREATE INDEX idx_measurements_pressure ON measurements(pressure)',
                'CREATE INDEX idx_measurements_temp_sal ON measurements(temperature, salinity) WHERE temperature_qc IN (1,2,5,8) AND salinity_qc IN (1,2,5,8)',
                'CREATE INDEX idx_measurements_surface ON measurements(profile_id, temperature, salinity) WHERE pressure < 10',
                'CREATE INDEX idx_measurements_bgc ON measurements(dissolved_oxygen, chlorophyll_a, ph, nitrate) WHERE dissolved_oxygen_qc IN (1,2,5,8)',
                'CREATE INDEX idx_measurements_quality ON measurements(temperature_qc, salinity_qc) WHERE temperature_qc IN (1,2,5,8) AND salinity_qc IN (1,2,5,8)'
            ]
        }
        
        # 4. Aggregated data table for fast queries
        schema_design['tables']['profile_statistics'] = {
            'description': 'Pre-computed profile statistics for fast LLM queries',
            'columns': {
                'profile_id': 'INTEGER REFERENCES profiles(profile_id) PRIMARY KEY',
                'surface_temperature': 'DECIMAL(6,3)',
                'surface_salinity': 'DECIMAL(7,4)',
                'max_depth': 'DECIMAL(8,2)',
                'temperature_range': 'NUMRANGE',
                'salinity_range': 'NUMRANGE',
                'mixed_layer_depth': 'DECIMAL(8,2)',
                'thermocline_depth': 'DECIMAL(8,2)',
                'measurement_counts': 'JSONB',  # Count of each parameter
                'data_quality_score': 'DECIMAL(3,2)',
                'computed_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            },
            'indexes': [
                'CREATE INDEX idx_profile_stats_surface ON profile_statistics(surface_temperature, surface_salinity)',
                'CREATE INDEX idx_profile_stats_depth ON profile_statistics(max_depth, mixed_layer_depth)',
                'CREATE INDEX idx_profile_stats_quality ON profile_statistics(data_quality_score)'
            ]
        }
        
        # 5. Views for common LLM queries
        schema_design['views'] = [
            {
                'name': 'float_summary',
                'definition': '''
                CREATE VIEW float_summary AS
                SELECT 
                    f.*,
                    COUNT(p.profile_id) as total_profiles,
                    MIN(p.profile_date) as first_profile_date,
                    MAX(p.profile_date) as last_profile_date,
                    AVG(ps.surface_temperature) as avg_surface_temperature,
                    AVG(ps.surface_salinity) as avg_surface_salinity
                FROM floats f
                LEFT JOIN profiles p ON f.float_id = p.float_id
                LEFT JOIN profile_statistics ps ON p.profile_id = ps.profile_id
                GROUP BY f.float_id;
                '''
            },
            {
                'name': 'recent_profiles',
                'definition': '''
                CREATE VIEW recent_profiles AS
                SELECT 
                    p.*,
                    f.wmo_number,
                    f.platform_number,
                    ps.surface_temperature,
                    ps.surface_salinity,
                    ps.max_depth
                FROM profiles p
                JOIN floats f ON p.float_id = f.float_id
                LEFT JOIN profile_statistics ps ON p.profile_id = ps.profile_id
                WHERE p.profile_date >= NOW() - INTERVAL '30 days'
                AND p.quality_flag IN (1, 2, 5, 8);
                '''
            }
        ]
        
        # 6. Partitioning strategy
        schema_design['partitioning_strategy'] = {
            'profiles': {
                'method': 'Range partitioning by profile_date',
                'partition_size': 'Monthly partitions',
                'retention': 'Keep 10 years of data',
                'example': '''
                CREATE TABLE profiles_y2023m01 PARTITION OF profiles
                FOR VALUES FROM ('2023-01-01') TO ('2023-02-01');
                '''
            },
            'measurements': {
                'method': 'Range partitioning by profile_id',
                'partition_size': '1M profile ranges',
                'benefits': 'Faster profile-based queries',
                'example': '''
                CREATE TABLE measurements_p1to1000000 PARTITION OF measurements
                FOR VALUES FROM (1) TO (1000000);
                '''
            }
        }
        
        # 7. LLM Query Optimization
        schema_design['query_optimization'] = {
            'spatial_queries': {
                'description': 'Optimize for location-based queries',
                'techniques': [
                    'PostGIS spatial indexing with GIST',
                    'Spatial clustering for nearby floats',
                    'Pre-computed bounding boxes for regions'
                ]
            },
            'temporal_queries': {
                'description': 'Optimize for time-based queries',
                'techniques': [
                    'B-tree indexing on timestamps',
                    'Temporal partitioning by month/year',
                    'Materialized views for common time ranges'
                ]
            },
            'parameter_queries': {
                'description': 'Optimize for oceanographic parameter queries',
                'techniques': [
                    'Composite indexes on (temperature, salinity)',
                    'Quality flag filtering in WHERE clauses',
                    'JSONB indexing for flexible parameters'
                ]
            },
            'aggregation_queries': {
                'description': 'Optimize for statistical and summary queries',
                'techniques': [
                    'Pre-computed statistics table',
                    'Materialized views for common aggregations',
                    'Window functions for trend analysis'
                ]
            }
        }
        
        # Display schema design
        rprint("\n[bold green]📋 OPTIMAL SCHEMA DESIGN COMPLETED[/bold green]")
        
        table = Table(title="Database Tables Overview")
        table.add_column("Table", style="cyan")
        table.add_column("Purpose", style="yellow")
        table.add_column("Key Features", style="green")
        
        for table_name, table_info in schema_design['tables'].items():
            key_features = []
            if 'partitioning' in table_info:
                key_features.append("Partitioned")
            if len(table_info.get('indexes', [])) > 0:
                key_features.append(f"{len(table_info['indexes'])} indexes")
            
            table.add_row(
                table_name,
                table_info['description'][:50] + "..." if len(table_info['description']) > 50 else table_info['description'],
                ", ".join(key_features)
            )
        
        console.print(table)
        
        self.analysis_results['schema_design'] = schema_design
        return schema_design
    
    def _get_dimension_description(self, dim_name: str) -> str:
        """Get human-readable description for dimension."""
        descriptions = {
            'N_PROF': 'Number of profiles in the file',
            'N_LEVELS': 'Maximum number of pressure levels per profile',
            'N_PARAM': 'Number of parameters measured',
            'N_CALIB': 'Number of calibration coefficients',
            'N_HISTORY': 'Number of history records',
            'STRING2': '2-character string dimension',
            'STRING4': '4-character string dimension',
            'STRING8': '8-character string dimension',
            'STRING16': '16-character string dimension',
            'STRING32': '32-character string dimension',
            'STRING64': '64-character string dimension',
            'STRING256': '256-character string dimension',
            'DATE_TIME': 'Date/time string dimension (14 characters)'
        }
        return descriptions.get(dim_name, f'Dimension: {dim_name}')
    
    def _categorize_variables(self, variables_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize variables by their purpose."""
        categories = {
            'coordinates': [],
            'core_parameters': [],
            'bgc_parameters': [],
            'quality_flags': [],
            'metadata': [],
            'technical': []
        }
        
        for var_name, var_info in variables_info.items():
            if var_name in ['LATITUDE', 'LONGITUDE', 'JULD', 'PRES']:
                categories['coordinates'].append(var_name)
            elif var_name in ['TEMP', 'PSAL', 'CNDC']:
                categories['core_parameters'].append(var_name)
            elif var_name in ['DOXY', 'CHLA', 'BBP700', 'PH_IN_SITU_TOTAL', 'NITRATE']:
                categories['bgc_parameters'].append(var_name)
            elif '_QC' in var_name or 'FLAG' in var_name:
                categories['quality_flags'].append(var_name)
            elif var_name in ['PLATFORM_NUMBER', 'PROJECT_NAME', 'PI_NAME', 'CYCLE_NUMBER']:
                categories['metadata'].append(var_name)
            else:
                categories['technical'].append(var_name)
        
        return categories
    
    def _calculate_variable_statistics(self, variable) -> Dict[str, Any]:
        """Calculate basic statistics for a variable."""
        try:
            values = variable.values
            if np.issubdtype(values.dtype, np.number):
                # Remove fill values and NaN
                fill_value = variable.attrs.get('_FillValue')
                if fill_value is not None:
                    values = values[values != fill_value]
                values = values[~np.isnan(values)]
                
                if len(values) > 0:
                    return {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': int(len(values)),
                        'null_count': int(np.sum(np.isnan(variable.values)))
                    }
            
            return {
                'count': int(variable.size),
                'null_count': int(np.sum(pd.isna(variable.values)))
            }
        except Exception:
            return {'error': 'Could not calculate statistics'}
    
    def _get_sample_values(self, variable, n_samples: int = 5) -> List[Any]:
        """Get sample values from a variable."""
        try:
            values = variable.values.flatten()
            fill_value = variable.attrs.get('_FillValue')
            if fill_value is not None:
                values = values[values != fill_value]
            values = values[~np.isnan(values)]
            
            if len(values) > n_samples:
                indices = np.linspace(0, len(values)-1, n_samples, dtype=int)
                return [values[i] for i in indices]
            else:
                return values.tolist()
        except Exception:
            return []
    
    def _get_value_range(self, variable) -> tuple:
        """Get min/max range of variable values."""
        try:
            values = variable.values
            fill_value = variable.attrs.get('_FillValue')
            if fill_value is not None:
                values = values[values != fill_value]
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                return (float(np.min(values)), float(np.max(values)))
            return (None, None)
        except Exception:
            return (None, None)
    
    def _calculate_flag_distribution(self, qc_variable) -> Dict[str, int]:
        """Calculate distribution of QC flag values."""
        try:
            values = qc_variable.values.flatten()
            values = values[~np.isnan(values.astype(float))]
            unique, counts = np.unique(values, return_counts=True)
            return {str(int(flag)): int(count) for flag, count in zip(unique, counts)}
        except Exception:
            return {}
    
    def _analyze_temporal_patterns(self, juld_variable) -> Dict[str, Any]:
        """Analyze temporal patterns in JULD variable."""
        try:
            # ARGO uses Julian days since 1950-01-01
            reference_date = pd.Timestamp('1950-01-01')
            
            values = juld_variable.values.flatten()
            fill_value = juld_variable.attrs.get('_FillValue')
            if fill_value is not None:
                values = values[values != fill_value]
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                # Convert to datetime
                timestamps = [reference_date + pd.Timedelta(days=float(d)) for d in values]
                
                return {
                    'earliest_date': str(min(timestamps)),
                    'latest_date': str(max(timestamps)),
                    'time_span_days': int(max(values) - min(values)),
                    'total_profiles': len(values),
                    'sample_dates': [str(ts) for ts in timestamps[:5]]
                }
            
            return {'error': 'No valid temporal data found'}
        except Exception as e:
            return {'error': f'Temporal analysis failed: {str(e)}'}
    
    def save_analysis_results(self, output_file: Path) -> None:
        """Save comprehensive analysis results to JSON file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj
        
        # Deep convert the results
        json_results = json.loads(json.dumps(self.analysis_results, default=convert_numpy))
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        rprint(f"\n[bold green]💾 Analysis results saved to:[/bold green] {output_file}")
    
    def run_complete_analysis(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Run complete ARGO NetCDF analysis."""
        rprint(Panel.fit(
            "[bold cyan]ARGO NetCDF Comprehensive Analysis[/bold cyan]\n"
            f"File: {self.file_path}\n"
            f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue"
        ))
        
        # Load dataset
        self.load_dataset()
        
        # Run all analyses
        self.analyze_dimensions()
        self.analyze_variables()
        self.analyze_coordinates()
        self.analyze_quality_flags()
        self.analyze_global_attributes()
        self.analyze_data_patterns()
        self.design_optimal_schema()
        
        # Save results if requested
        if output_file:
            self.save_analysis_results(output_file)
        
        rprint(Panel.fit(
            "[bold green]✅ ANALYSIS COMPLETE[/bold green]\n"
            f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Database schema design optimized for LLM queries ready for implementation.",
            border_style="green"
        ))
        
        return self.analysis_results


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        rprint("[red]Usage: python analyze_netcdf_structure.py <netcdf_file_path> [output_json_path][/red]")
        sys.exit(1)
    
    netcdf_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    if not netcdf_file.exists():
        rprint(f"[red]Error: NetCDF file {netcdf_file} does not exist[/red]")
        sys.exit(1)
    
    # Run analysis
    analyzer = ARGONetCDFAnalyzer(netcdf_file)
    results = analyzer.run_complete_analysis(output_file)
    
    rprint(f"\n[bold blue]Schema design ready for implementation in Phase 1.2![/bold blue]")


if __name__ == "__main__":
    main()