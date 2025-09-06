"""
Oceanographic database schema definitions for SQL generation.
Maps oceanographic concepts to database structure with optimization hints.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of oceanographic parameters."""
    PHYSICAL = "physical"
    CHEMICAL = "chemical" 
    BIOLOGICAL = "biological"
    DERIVED = "derived"
    METADATA = "metadata"


class DataQuality(Enum):
    """ARGO data quality levels."""
    GOOD = "1"
    PROBABLY_GOOD = "2"
    PROBABLY_BAD = "3"
    BAD = "4"
    CHANGED = "5"
    NOT_USED = "8"
    MISSING = "9"


@dataclass
class ColumnInfo:
    """Information about a database column."""
    column_name: str
    data_type: str
    parameter_type: ParameterType
    units: str
    description: str
    quality_column: Optional[str] = None
    index_type: Optional[str] = None
    nullable: bool = True
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass 
class TableInfo:
    """Information about a database table."""
    table_name: str
    primary_key: str
    columns: Dict[str, ColumnInfo]
    foreign_keys: Dict[str, str]
    indexes: List[str]
    description: str


class OceanographicSchema:
    """Complete schema definition for oceanographic database."""
    
    def __init__(self):
        self.tables = self._initialize_schema()
        self.parameter_mappings = self._initialize_parameter_mappings()
        self.spatial_reference_system = 'EPSG:4326'  # WGS84
        
        logger.info("Oceanographic schema initialized")
    
    def _initialize_schema(self) -> Dict[str, TableInfo]:
        """Initialize complete database schema."""
        
        # DACs (Data Assembly Centers) table
        dacs_table = TableInfo(
            table_name="dacs",
            primary_key="id",
            description="ARGO Data Assembly Centers",
            foreign_keys={},
            indexes=["idx_dacs_code"],
            columns={
                "id": ColumnInfo("id", "INTEGER", ParameterType.METADATA, "", "Primary key"),
                "code": ColumnInfo("code", "VARCHAR(10)", ParameterType.METADATA, "", "DAC code", index_type="unique"),
                "name": ColumnInfo("name", "VARCHAR(255)", ParameterType.METADATA, "", "DAC full name"),
                "country": ColumnInfo("country", "VARCHAR(100)", ParameterType.METADATA, "", "Country"),
                "institution": ColumnInfo("institution", "VARCHAR(255)", ParameterType.METADATA, "", "Institution name")
            }
        )
        
        # Floats table
        floats_table = TableInfo(
            table_name="floats",
            primary_key="id",
            description="ARGO float metadata",
            foreign_keys={"dac_id": "dacs.id"},
            indexes=["idx_floats_platform_number", "idx_floats_deployment_location", "idx_floats_active"],
            columns={
                "id": ColumnInfo("id", "INTEGER", ParameterType.METADATA, "", "Primary key"),
                "platform_number": ColumnInfo("platform_number", "VARCHAR(20)", ParameterType.METADATA, "", 
                                             "Float platform number", index_type="unique",
                                             aliases=["float_id", "platform_id", "wmo_id"]),
                "dac_id": ColumnInfo("dac_id", "INTEGER", ParameterType.METADATA, "", "Data Assembly Center ID"),
                "platform_type": ColumnInfo("platform_type", "VARCHAR(50)", ParameterType.METADATA, "", 
                                           "Float platform type", aliases=["float_type"]),
                "deployment_date": ColumnInfo("deployment_date", "DATE", ParameterType.METADATA, "", "Deployment date"),
                "deployment_latitude": ColumnInfo("deployment_latitude", "REAL", ParameterType.PHYSICAL, "degrees_north", 
                                                 "Deployment latitude", aliases=["deploy_lat"]),
                "deployment_longitude": ColumnInfo("deployment_longitude", "REAL", ParameterType.PHYSICAL, "degrees_east", 
                                                  "Deployment longitude", aliases=["deploy_lon"]),
                "project_name": ColumnInfo("project_name", "VARCHAR(255)", ParameterType.METADATA, "", "Project name"),
                "pi_name": ColumnInfo("pi_name", "VARCHAR(255)", ParameterType.METADATA, "", "Principal investigator"),
                "is_active": ColumnInfo("is_active", "BOOLEAN", ParameterType.METADATA, "", "Float active status",
                                       index_type="btree", aliases=["active"])
            }
        )
        
        # Profiles table
        profiles_table = TableInfo(
            table_name="profiles",
            primary_key="id",
            description="ARGO float profiles (individual casts)",
            foreign_keys={"float_id": "floats.id"},
            indexes=["idx_profiles_float_cycle", "idx_profiles_location", "idx_profiles_date", "idx_profiles_spatial"],
            columns={
                "id": ColumnInfo("id", "INTEGER", ParameterType.METADATA, "", "Primary key"),
                "float_id": ColumnInfo("float_id", "INTEGER", ParameterType.METADATA, "", "Float reference"),
                "cycle_number": ColumnInfo("cycle_number", "INTEGER", ParameterType.METADATA, "", "Profile cycle number",
                                         aliases=["cycle", "profile_number"]),
                "direction": ColumnInfo("direction", "VARCHAR(1)", ParameterType.METADATA, "", "Profile direction (A/D)"),
                "latitude": ColumnInfo("latitude", "REAL", ParameterType.PHYSICAL, "degrees_north", 
                                     "Profile latitude", index_type="spatial", aliases=["lat"]),
                "longitude": ColumnInfo("longitude", "REAL", ParameterType.PHYSICAL, "degrees_east", 
                                       "Profile longitude", index_type="spatial", aliases=["lon"]),
                "measurement_date": ColumnInfo("measurement_date", "TIMESTAMP", ParameterType.METADATA, "", 
                                              "Measurement date/time", index_type="btree", 
                                              aliases=["date", "datetime", "time"]),
                "data_mode": ColumnInfo("data_mode", "VARCHAR(1)", ParameterType.METADATA, "", 
                                       "Data mode (R/D/A)", aliases=["mode"]),
                "max_depth_m": ColumnInfo("max_depth_m", "REAL", ParameterType.PHYSICAL, "meters", 
                                         "Maximum depth of profile", aliases=["max_depth"]),
                "min_temperature_c": ColumnInfo("min_temperature_c", "REAL", ParameterType.PHYSICAL, "degrees_celsius", 
                                               "Minimum temperature in profile"),
                "max_temperature_c": ColumnInfo("max_temperature_c", "REAL", ParameterType.PHYSICAL, "degrees_celsius", 
                                               "Maximum temperature in profile"),
                "min_salinity_psu": ColumnInfo("min_salinity_psu", "REAL", ParameterType.PHYSICAL, "psu", 
                                              "Minimum salinity in profile"),
                "max_salinity_psu": ColumnInfo("max_salinity_psu", "REAL", ParameterType.PHYSICAL, "psu", 
                                              "Maximum salinity in profile"),
                "valid_measurements_count": ColumnInfo("valid_measurements_count", "INTEGER", ParameterType.METADATA, "", 
                                                      "Number of valid measurements")
            }
        )
        
        # Measurements table
        measurements_table = TableInfo(
            table_name="measurements",
            primary_key="id",
            description="Individual oceanographic measurements",
            foreign_keys={"profile_id": "profiles.id"},
            indexes=["idx_measurements_profile_depth", "idx_measurements_params", "idx_measurements_quality"],
            columns={
                "id": ColumnInfo("id", "INTEGER", ParameterType.METADATA, "", "Primary key"),
                "profile_id": ColumnInfo("profile_id", "INTEGER", ParameterType.METADATA, "", "Profile reference"),
                "depth_level": ColumnInfo("depth_level", "INTEGER", ParameterType.METADATA, "", "Depth level index"),
                
                # Core physical parameters
                "pressure_db": ColumnInfo("pressure_db", "REAL", ParameterType.PHYSICAL, "decibar", 
                                         "Pressure", quality_column="pressure_qc", 
                                         aliases=["pressure", "pres"]),
                "depth_m": ColumnInfo("depth_m", "REAL", ParameterType.PHYSICAL, "meters", 
                                     "Depth", aliases=["depth"]),
                "temperature_c": ColumnInfo("temperature_c", "REAL", ParameterType.PHYSICAL, "degrees_celsius", 
                                           "Temperature", quality_column="temperature_qc", 
                                           aliases=["temperature", "temp"]),
                "salinity_psu": ColumnInfo("salinity_psu", "REAL", ParameterType.PHYSICAL, "psu", 
                                          "Salinity", quality_column="salinity_qc", 
                                          aliases=["salinity", "sal"]),
                
                # Biogeochemical parameters
                "oxygen_ml_l": ColumnInfo("oxygen_ml_l", "REAL", ParameterType.CHEMICAL, "ml/l", 
                                         "Dissolved oxygen", quality_column="oxygen_qc", 
                                         aliases=["oxygen", "o2", "dissolved_oxygen"]),
                "nitrate_umol_kg": ColumnInfo("nitrate_umol_kg", "REAL", ParameterType.CHEMICAL, "umol/kg", 
                                             "Nitrate", quality_column="nitrate_qc", 
                                             aliases=["nitrate", "no3"]),
                "phosphate_umol_kg": ColumnInfo("phosphate_umol_kg", "REAL", ParameterType.CHEMICAL, "umol/kg", 
                                               "Phosphate", quality_column="phosphate_qc", 
                                               aliases=["phosphate", "po4"]),
                "silicate_umol_kg": ColumnInfo("silicate_umol_kg", "REAL", ParameterType.CHEMICAL, "umol/kg", 
                                              "Silicate", quality_column="silicate_qc", 
                                              aliases=["silicate", "sio4"]),
                "ph_total": ColumnInfo("ph_total", "REAL", ParameterType.CHEMICAL, "pH_units", 
                                      "pH total scale", quality_column="ph_qc", 
                                      aliases=["ph", "acidity"]),
                "chlorophyll_mg_m3": ColumnInfo("chlorophyll_mg_m3", "REAL", ParameterType.BIOLOGICAL, "mg/m3", 
                                               "Chlorophyll-a", quality_column="chlorophyll_qc", 
                                               aliases=["chlorophyll", "chl", "chla"]),
                
                # Quality control flags
                "pressure_qc": ColumnInfo("pressure_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Pressure QC flag"),
                "temperature_qc": ColumnInfo("temperature_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Temperature QC flag"),
                "salinity_qc": ColumnInfo("salinity_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Salinity QC flag"),
                "oxygen_qc": ColumnInfo("oxygen_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Oxygen QC flag"),
                "nitrate_qc": ColumnInfo("nitrate_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Nitrate QC flag"),
                "phosphate_qc": ColumnInfo("phosphate_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Phosphate QC flag"),
                "silicate_qc": ColumnInfo("silicate_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Silicate QC flag"),
                "ph_qc": ColumnInfo("ph_qc", "VARCHAR(1)", ParameterType.METADATA, "", "pH QC flag"),
                "chlorophyll_qc": ColumnInfo("chlorophyll_qc", "VARCHAR(1)", ParameterType.METADATA, "", "Chlorophyll QC flag"),
                
                # Derived/calculated fields
                "is_valid": ColumnInfo("is_valid", "BOOLEAN", ParameterType.METADATA, "", 
                                      "Overall measurement validity", index_type="btree")
            }
        )
        
        return {
            "dacs": dacs_table,
            "floats": floats_table, 
            "profiles": profiles_table,
            "measurements": measurements_table
        }
    
    def _initialize_parameter_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize parameter to column mappings with metadata."""
        
        return {
            # Physical parameters
            'temperature': {
                'column': 'temperature_c',
                'table': 'measurements',
                'type': ParameterType.PHYSICAL,
                'units': 'degrees_celsius',
                'typical_range': (-2.0, 35.0),
                'qc_column': 'temperature_qc',
                'aliases': ['temp', 'sst', 'sea_surface_temperature']
            },
            'salinity': {
                'column': 'salinity_psu',
                'table': 'measurements',
                'type': ParameterType.PHYSICAL,
                'units': 'psu',
                'typical_range': (30.0, 37.0),
                'qc_column': 'salinity_qc',
                'aliases': ['sal', 'sss', 'sea_surface_salinity']
            },
            'pressure': {
                'column': 'pressure_db',
                'table': 'measurements',
                'type': ParameterType.PHYSICAL,
                'units': 'decibar',
                'typical_range': (0.0, 6000.0),
                'qc_column': 'pressure_qc',
                'aliases': ['pres']
            },
            'depth': {
                'column': 'depth_m',
                'table': 'measurements',
                'type': ParameterType.PHYSICAL,
                'units': 'meters',
                'typical_range': (0.0, 6000.0),
                'aliases': ['depth_m']
            },
            
            # Chemical parameters
            'oxygen': {
                'column': 'oxygen_ml_l',
                'table': 'measurements',
                'type': ParameterType.CHEMICAL,
                'units': 'ml/l',
                'typical_range': (0.0, 15.0),
                'qc_column': 'oxygen_qc',
                'aliases': ['o2', 'dissolved_oxygen', 'do']
            },
            'nitrate': {
                'column': 'nitrate_umol_kg',
                'table': 'measurements',
                'type': ParameterType.CHEMICAL,
                'units': 'umol/kg',
                'typical_range': (0.0, 50.0),
                'qc_column': 'nitrate_qc',
                'aliases': ['no3', 'nitrogen']
            },
            'phosphate': {
                'column': 'phosphate_umol_kg',
                'table': 'measurements',
                'type': ParameterType.CHEMICAL,
                'units': 'umol/kg',
                'typical_range': (0.0, 5.0),
                'qc_column': 'phosphate_qc',
                'aliases': ['po4', 'phosphorus']
            },
            'ph': {
                'column': 'ph_total',
                'table': 'measurements',
                'type': ParameterType.CHEMICAL,
                'units': 'pH_units',
                'typical_range': (7.5, 8.5),
                'qc_column': 'ph_qc',
                'aliases': ['acidity', 'ph_total']
            },
            
            # Biological parameters
            'chlorophyll': {
                'column': 'chlorophyll_mg_m3',
                'table': 'measurements',
                'type': ParameterType.BIOLOGICAL,
                'units': 'mg/m3',
                'typical_range': (0.0, 30.0),
                'qc_column': 'chlorophyll_qc',
                'aliases': ['chl', 'chla', 'phytoplankton']
            },
            
            # Derived parameters
            'density': {
                'column': '(1025 + 0.7 * salinity_psu - 0.2 * temperature_c + 0.0005 * pressure_db)',
                'table': 'measurements',
                'type': ParameterType.DERIVED,
                'units': 'kg/m3',
                'typical_range': (1020.0, 1030.0),
                'aliases': ['sigma', 'rho', 'potential_density']
            },
            'sound_speed': {
                'column': '(1449.2 + 4.6 * temperature_c - 0.055 * temperature_c^2 + 0.00029 * temperature_c^3 + (1.34 - 0.01 * temperature_c) * (salinity_psu - 35) + 0.016 * depth_m)',
                'table': 'measurements',
                'type': ParameterType.DERIVED,
                'units': 'm/s',
                'typical_range': (1450.0, 1550.0),
                'aliases': ['sound_velocity']
            },
            
            # Spatial/temporal parameters
            'latitude': {
                'column': 'latitude',
                'table': 'profiles',
                'type': ParameterType.PHYSICAL,
                'units': 'degrees_north',
                'typical_range': (-90.0, 90.0),
                'aliases': ['lat']
            },
            'longitude': {
                'column': 'longitude',
                'table': 'profiles',
                'type': ParameterType.PHYSICAL,
                'units': 'degrees_east',
                'typical_range': (-180.0, 180.0),
                'aliases': ['lon']
            },
            'measurement_date': {
                'column': 'measurement_date',
                'table': 'profiles',
                'type': ParameterType.METADATA,
                'units': 'datetime',
                'aliases': ['date', 'time', 'datetime']
            }
        }
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Get information about a specific table."""
        return self.tables.get(table_name)
    
    def get_column_info(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        """Get information about a specific column."""
        table = self.tables.get(table_name)
        if table:
            return table.columns.get(column_name)
        return None
    
    def get_parameter_info(self, parameter_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an oceanographic parameter."""
        # Direct lookup
        if parameter_name in self.parameter_mappings:
            return self.parameter_mappings[parameter_name]
        
        # Alias lookup
        for param, info in self.parameter_mappings.items():
            if parameter_name in info.get('aliases', []):
                return info
        
        return None
    
    def get_measurement_column(self, parameter_name: str) -> Optional[str]:
        """Get the measurement table column for a parameter."""
        param_info = self.get_parameter_info(parameter_name)
        if param_info and param_info['table'] == 'measurements':
            return param_info['column']
        return None
    
    def get_quality_column(self, parameter_name: str) -> Optional[str]:
        """Get the quality control column for a parameter."""
        param_info = self.get_parameter_info(parameter_name)
        if param_info:
            return param_info.get('qc_column')
        return None
    
    def get_parameter_units(self, parameter_name: str) -> Optional[str]:
        """Get units for a parameter."""
        param_info = self.get_parameter_info(parameter_name)
        if param_info:
            return param_info.get('units')
        return None
    
    def get_parameter_range(self, parameter_name: str) -> Optional[Tuple[float, float]]:
        """Get typical range for a parameter."""
        param_info = self.get_parameter_info(parameter_name)
        if param_info:
            return param_info.get('typical_range')
        return None
    
    def get_spatial_columns(self) -> List[str]:
        """Get columns that contain spatial information."""
        return ['profiles.latitude', 'profiles.longitude', 'floats.deployment_latitude', 'floats.deployment_longitude']
    
    def get_temporal_columns(self) -> List[str]:
        """Get columns that contain temporal information."""
        return ['profiles.measurement_date', 'floats.deployment_date']
    
    def get_quality_columns(self) -> Dict[str, str]:
        """Get mapping of parameter columns to their quality control columns."""
        quality_mapping = {}
        for param, info in self.parameter_mappings.items():
            if 'qc_column' in info and info['table'] == 'measurements':
                quality_mapping[info['column']] = info['qc_column']
        return quality_mapping
    
    def get_indexed_columns(self) -> Dict[str, List[str]]:
        """Get columns with indexes for query optimization."""
        indexed_columns = {}
        for table_name, table_info in self.tables.items():
            indexed_cols = []
            for col_name, col_info in table_info.columns.items():
                if col_info.index_type:
                    indexed_cols.append(col_name)
            if indexed_cols:
                indexed_columns[table_name] = indexed_cols
        return indexed_columns
    
    def get_join_paths(self, from_table: str, to_table: str) -> List[List[str]]:
        """Get possible join paths between two tables."""
        
        # Define known join paths
        join_paths = {
            ('dacs', 'floats'): [['dacs.id', 'floats.dac_id']],
            ('floats', 'profiles'): [['floats.id', 'profiles.float_id']],
            ('profiles', 'measurements'): [['profiles.id', 'measurements.profile_id']],
            
            # Multi-hop paths
            ('dacs', 'profiles'): [
                ['dacs.id', 'floats.dac_id', 'floats.id', 'profiles.float_id']
            ],
            ('dacs', 'measurements'): [
                ['dacs.id', 'floats.dac_id', 'floats.id', 'profiles.float_id', 'profiles.id', 'measurements.profile_id']
            ],
            ('floats', 'measurements'): [
                ['floats.id', 'profiles.float_id', 'profiles.id', 'measurements.profile_id']
            ]
        }
        
        # Return paths for both directions
        key = (from_table, to_table)
        reverse_key = (to_table, from_table)
        
        if key in join_paths:
            return join_paths[key]
        elif reverse_key in join_paths:
            # Reverse the join path
            paths = join_paths[reverse_key]
            reversed_paths = []
            for path in paths:
                reversed_path = []
                for i in range(len(path) - 1, -1, -1):
                    reversed_path.append(path[i])
                reversed_paths.append(reversed_path)
            return reversed_paths
        
        return []
    
    def get_aggregatable_parameters(self) -> List[str]:
        """Get parameters suitable for aggregation operations."""
        aggregatable = []
        for param, info in self.parameter_mappings.items():
            if info['type'] in [ParameterType.PHYSICAL, ParameterType.CHEMICAL, ParameterType.BIOLOGICAL]:
                aggregatable.append(param)
        return aggregatable
    
    def get_filterable_parameters(self) -> List[str]:
        """Get parameters suitable for filtering operations."""
        return list(self.parameter_mappings.keys())
    
    def validate_parameter_value(self, parameter_name: str, value: float) -> bool:
        """Validate if a parameter value is within reasonable range."""
        param_range = self.get_parameter_range(parameter_name)
        if param_range:
            min_val, max_val = param_range
            return min_val <= value <= max_val
        return True  # No range info, assume valid
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the database schema."""
        summary = {
            'tables': {},
            'total_columns': 0,
            'parameters': len(self.parameter_mappings),
            'spatial_reference': self.spatial_reference_system
        }
        
        for table_name, table_info in self.tables.items():
            summary['tables'][table_name] = {
                'columns': len(table_info.columns),
                'indexes': len(table_info.indexes),
                'foreign_keys': len(table_info.foreign_keys),
                'description': table_info.description
            }
            summary['total_columns'] += len(table_info.columns)
        
        return summary