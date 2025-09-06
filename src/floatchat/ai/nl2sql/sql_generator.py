"""
Advanced SQL Generator for oceanographic database queries.
Converts parsed natural language structures into optimized PostgreSQL queries
with spatial, temporal, and parameter constraints.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
import json
import re
from dataclasses import dataclass

from .query_parser import (
    QueryStructure, QueryIntent, SpatialConstraint, TemporalConstraint, 
    ParameterConstraint
)
from .oceanographic_schema import OceanographicSchema

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuery:
    """Generated SQL query with metadata and optimization information."""
    sql: str
    parameters: Dict[str, Any]
    estimated_complexity: str  # 'simple', 'moderate', 'complex'
    execution_time_estimate: float  # seconds
    result_size_estimate: int
    optimization_notes: List[str]
    safety_checks: List[str]
    confidence_score: float


class SQLTemplateEngine:
    """Template engine for common oceanographic query patterns."""
    
    def __init__(self):
        # Common query templates optimized for oceanographic data
        self.templates = {
            'profile_data_retrieval': """
                SELECT 
                    f.platform_number,
                    p.cycle_number,
                    p.latitude,
                    p.longitude,
                    p.measurement_date,
                    m.depth_m,
                    {parameter_columns}
                FROM floats f
                JOIN profiles p ON f.id = p.float_id
                JOIN measurements m ON p.id = m.profile_id
                WHERE {conditions}
                {groupby_clause}
                {orderby_clause}
                {limit_clause}
            """,
            
            'spatial_aggregation': """
                SELECT 
                    {aggregation_columns},
                    COUNT(*) as measurement_count,
                    MIN(p.measurement_date) as earliest_date,
                    MAX(p.measurement_date) as latest_date
                FROM floats f
                JOIN profiles p ON f.id = p.float_id
                JOIN measurements m ON p.id = m.profile_id
                WHERE {conditions}
                GROUP BY {groupby_fields}
                {having_clause}
                {orderby_clause}
                {limit_clause}
            """,
            
            'temporal_analysis': """
                WITH temporal_data AS (
                    SELECT 
                        DATE_TRUNC('{time_granularity}', p.measurement_date) as time_period,
                        {parameter_columns},
                        p.latitude,
                        p.longitude,
                        m.depth_m
                    FROM floats f
                    JOIN profiles p ON f.id = p.float_id
                    JOIN measurements m ON p.id = m.profile_id
                    WHERE {conditions}
                )
                SELECT 
                    time_period,
                    {aggregation_columns}
                FROM temporal_data
                GROUP BY time_period
                {orderby_clause}
                {limit_clause}
            """,
            
            'comparative_analysis': """
                WITH dataset_a AS (
                    SELECT {parameter_columns}
                    FROM floats f
                    JOIN profiles p ON f.id = p.float_id
                    JOIN measurements m ON p.id = m.profile_id
                    WHERE {conditions_a}
                ),
                dataset_b AS (
                    SELECT {parameter_columns}
                    FROM floats f
                    JOIN profiles p ON f.id = p.float_id
                    JOIN measurements m ON p.id = m.profile_id
                    WHERE {conditions_b}
                )
                SELECT 
                    'dataset_a' as dataset_name,
                    {aggregation_columns}
                FROM dataset_a
                UNION ALL
                SELECT 
                    'dataset_b' as dataset_name,
                    {aggregation_columns}
                FROM dataset_b
            """,
            
            'water_mass_analysis': """
                SELECT 
                    f.platform_number,
                    p.latitude,
                    p.longitude,
                    p.measurement_date,
                    m.depth_m,
                    m.temperature_c,
                    m.salinity_psu,
                    m.pressure_db,
                    -- Water mass classification indicators
                    CASE 
                        WHEN m.temperature_c > 15 AND m.salinity_psu > 35.5 THEN 'Surface Water'
                        WHEN m.temperature_c BETWEEN 5 AND 15 AND m.salinity_psu BETWEEN 34.5 AND 35.5 THEN 'Intermediate Water'
                        WHEN m.temperature_c < 5 THEN 'Deep Water'
                        ELSE 'Unclassified'
                    END as water_mass_type,
                    -- Density approximation
                    (1025 + 0.7 * m.salinity_psu - 0.2 * m.temperature_c + 0.0005 * m.pressure_db) as density_approx
                FROM floats f
                JOIN profiles p ON f.id = p.float_id
                JOIN measurements m ON p.id = m.profile_id
                WHERE {conditions}
                {orderby_clause}
                {limit_clause}
            """
        }
        
        # Performance optimization templates
        self.optimization_patterns = {
            'spatial_index_hint': "AND ST_Contains(ST_MakeEnvelope({bbox}, 4326), ST_Point(p.longitude, p.latitude))",
            'temporal_index_hint': "AND p.measurement_date >= %s AND p.measurement_date <= %s",
            'depth_optimization': "AND m.depth_m BETWEEN %s AND %s",
            'quality_filter': "AND m.{parameter}_qc IN ('1', '2')",
            'float_active_filter': "AND f.is_active = true"
        }
    
    def get_template(self, query_structure: QueryStructure) -> str:
        """Select appropriate template based on query structure."""
        
        if query_structure.intent == QueryIntent.COMPARISON:
            return self.templates['comparative_analysis']
        elif query_structure.intent == QueryIntent.TEMPORAL_ANALYSIS:
            return self.templates['temporal_analysis']
        elif query_structure.intent == QueryIntent.SPATIAL_ANALYSIS:
            return self.templates['spatial_aggregation']
        elif query_structure.intent == QueryIntent.WATER_MASS_ANALYSIS:
            return self.templates['water_mass_analysis']
        elif query_structure.aggregation_type:
            return self.templates['spatial_aggregation']
        else:
            return self.templates['profile_data_retrieval']


class SQLSafetyValidator:
    """Comprehensive SQL safety validation for oceanographic queries."""
    
    def __init__(self):
        # Dangerous SQL patterns to block
        self.dangerous_patterns = [
            r';\s*(drop|delete|truncate|alter|create|insert|update)\s+',
            r'\bunion\s+select\b',
            r'--\s*',
            r'/\*.*?\*/',
            r'\bxp_cmdshell\b',
            r'\bsp_executesql\b',
            r'\bexec\s*\(',
            r'\bevalute\s*\(',
            r'\bsystem\s*\(',
            r'\bdbms_\w+',
            r'\butl_\w+'
        ]
        
        # Resource limits
        self.max_query_length = 5000
        self.max_joins = 10
        self.max_conditions = 50
        self.default_row_limit = 10000
    
    def validate_sql(self, sql: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive SQL safety validation."""
        issues = []
        
        # 1. Length check
        if len(sql) > self.max_query_length:
            issues.append(f"Query too long: {len(sql)} > {self.max_query_length} characters")
        
        # 2. Dangerous pattern detection
        sql_lower = sql.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, sql_lower, re.IGNORECASE | re.MULTILINE):
                issues.append(f"Potentially dangerous SQL pattern detected: {pattern}")
        
        # 3. Join complexity check
        join_count = len(re.findall(r'\bjoin\b', sql_lower))
        if join_count > self.max_joins:
            issues.append(f"Too many JOINs: {join_count} > {self.max_joins}")
        
        # 4. WHERE condition complexity
        where_conditions = len(re.findall(r'\band\b|\bor\b', sql_lower))
        if where_conditions > self.max_conditions:
            issues.append(f"Too many WHERE conditions: {where_conditions} > {self.max_conditions}")
        
        # 5. Ensure read-only operations
        if not sql_lower.strip().startswith('select') and not sql_lower.strip().startswith('with'):
            issues.append("Only SELECT queries are allowed")
        
        # 6. Parameter validation
        for param_name, param_value in parameters.items():
            if isinstance(param_value, str):
                if len(param_value) > 1000:
                    issues.append(f"Parameter '{param_name}' too long: {len(param_value)} characters")
                
                # Check for SQL injection in parameters
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, param_value, re.IGNORECASE):
                        issues.append(f"Dangerous pattern in parameter '{param_name}': {pattern}")
        
        # 7. Ensure LIMIT clause exists for large queries
        if 'limit' not in sql_lower and 'count(' not in sql_lower:
            if join_count > 2:  # Complex queries should have limits
                issues.append("Complex query missing LIMIT clause - potential performance issue")
        
        return len(issues) == 0, issues
    
    def sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize query parameters."""
        sanitized = {}
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized_value = re.sub(r'[;\'"\\]', '', value)
                sanitized_value = sanitized_value.strip()[:1000]  # Limit length
                sanitized[key] = sanitized_value
            elif isinstance(value, (int, float)):
                # Validate numeric ranges
                if isinstance(value, float):
                    if -1000000 <= value <= 1000000:  # Reasonable oceanographic ranges
                        sanitized[key] = value
                    else:
                        logger.warning(f"Parameter {key} value {value} out of reasonable range")
                        sanitized[key] = 0.0
                else:
                    sanitized[key] = int(value)
            elif isinstance(value, datetime):
                sanitized[key] = value
            else:
                # Convert other types to string and sanitize
                sanitized[key] = str(value)[:100]
        
        return sanitized


class OceanographicSQLGenerator:
    """Advanced SQL generator for oceanographic database queries."""
    
    def __init__(self):
        self.schema = OceanographicSchema()
        self.template_engine = SQLTemplateEngine()
        self.safety_validator = SQLSafetyValidator()
        
        # Performance estimation models
        self.complexity_weights = {
            'base_query': 0.1,
            'join_count': 0.05,
            'where_conditions': 0.02,
            'aggregations': 0.08,
            'spatial_operations': 0.15,
            'temporal_operations': 0.05,
            'result_size_factor': 0.0001
        }
        
        logger.info("Oceanographic SQL Generator initialized")
    
    async def generate_sql(self, query_structure: QueryStructure) -> GeneratedQuery:
        """Generate optimized SQL query from parsed structure."""
        
        try:
            # 1. Select appropriate template
            template = self.template_engine.get_template(query_structure)
            
            # 2. Build query components
            parameter_columns = self._build_parameter_columns(query_structure.parameters)
            conditions, query_parameters = self._build_conditions(query_structure)
            groupby_clause = self._build_groupby_clause(query_structure)
            orderby_clause = self._build_orderby_clause(query_structure)
            limit_clause = self._build_limit_clause(query_structure)
            aggregation_columns = self._build_aggregation_columns(query_structure)
            
            # 3. Handle special query types
            if query_structure.intent == QueryIntent.TEMPORAL_ANALYSIS:
                time_granularity = self._determine_time_granularity(query_structure)
                sql = template.format(
                    parameter_columns=parameter_columns,
                    conditions=conditions,
                    aggregation_columns=aggregation_columns,
                    time_granularity=time_granularity,
                    orderby_clause=orderby_clause,
                    limit_clause=limit_clause
                )
            elif query_structure.intent == QueryIntent.COMPARISON:
                # For comparative analysis, split conditions
                conditions_a, conditions_b = self._split_comparison_conditions(query_structure)
                sql = template.format(
                    parameter_columns=parameter_columns,
                    conditions_a=conditions_a,
                    conditions_b=conditions_b,
                    aggregation_columns=aggregation_columns
                )
            else:
                # Standard template formatting
                sql = template.format(
                    parameter_columns=parameter_columns,
                    conditions=conditions,
                    groupby_clause=groupby_clause,
                    orderby_clause=orderby_clause,
                    limit_clause=limit_clause,
                    aggregation_columns=aggregation_columns,
                    groupby_fields=', '.join(query_structure.groupby_fields) if query_structure.groupby_fields else '1',
                    having_clause=''  # Can be extended for HAVING conditions
                )
            
            # 4. Clean up and optimize SQL
            sql = self._clean_and_optimize_sql(sql)
            
            # 5. Safety validation
            is_safe, safety_issues = self.safety_validator.validate_sql(sql, query_parameters)
            if not is_safe:
                raise ValueError(f"SQL safety validation failed: {'; '.join(safety_issues)}")
            
            # 6. Sanitize parameters
            clean_parameters = self.safety_validator.sanitize_parameters(query_parameters)
            
            # 7. Performance estimation
            complexity = self._estimate_query_complexity(sql, query_structure)
            execution_time_estimate = self._estimate_execution_time(sql, query_structure)
            result_size_estimate = self._estimate_result_size(query_structure)
            
            # 8. Generate optimization notes
            optimization_notes = self._generate_optimization_notes(sql, query_structure)
            
            # 9. Calculate confidence score
            confidence_score = self._calculate_generation_confidence(query_structure, sql)
            
            return GeneratedQuery(
                sql=sql,
                parameters=clean_parameters,
                estimated_complexity=complexity,
                execution_time_estimate=execution_time_estimate,
                result_size_estimate=result_size_estimate,
                optimization_notes=optimization_notes,
                safety_checks=safety_issues,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            # Return a safe fallback query
            return self._generate_fallback_query(query_structure, str(e))
    
    def _build_parameter_columns(self, parameters: List[str]) -> str:
        """Build parameter column selections."""
        if not parameters:
            parameters = ['temperature', 'salinity', 'pressure']
        
        column_mappings = {
            'temperature': 'm.temperature_c',
            'salinity': 'm.salinity_psu', 
            'pressure': 'm.pressure_db',
            'oxygen': 'm.oxygen_ml_l',
            'density': '(1025 + 0.7 * m.salinity_psu - 0.2 * m.temperature_c + 0.0005 * m.pressure_db) as density_kgm3'
        }
        
        columns = []
        for param in parameters:
            if param in column_mappings:
                columns.append(column_mappings[param])
            else:
                # Try to find column in schema
                schema_column = self.schema.get_measurement_column(param)
                if schema_column:
                    columns.append(f'm.{schema_column}')
        
        return ',\n                    '.join(columns) if columns else 'm.temperature_c, m.salinity_psu'
    
    def _build_conditions(self, query_structure: QueryStructure) -> Tuple[str, Dict[str, Any]]:
        """Build WHERE clause conditions and parameters."""
        conditions = []
        parameters = {}
        param_counter = 1
        
        # Base conditions (always include)
        conditions.append("f.id = p.float_id")
        conditions.append("p.id = m.profile_id")
        
        # Spatial constraints
        for spatial in query_structure.spatial_constraints:
            if spatial.constraint_type == 'bounding_box' and spatial.coordinates:
                bbox = spatial.coordinates  # [min_lat, min_lon, max_lat, max_lon]
                conditions.append(f"p.latitude BETWEEN %s AND %s")
                conditions.append(f"p.longitude BETWEEN %s AND %s")
                parameters[f'lat_min_{param_counter}'] = bbox[0]
                parameters[f'lon_min_{param_counter}'] = bbox[1]
                parameters[f'lat_max_{param_counter}'] = bbox[2]
                parameters[f'lon_max_{param_counter}'] = bbox[3]
                param_counter += 1
                
            elif spatial.constraint_type == 'circle' and spatial.center_lat and spatial.center_lon:
                # Use Haversine distance for circular constraints
                conditions.append("""
                    (6371 * acos(cos(radians(%s)) * cos(radians(p.latitude)) * 
                     cos(radians(p.longitude) - radians(%s)) + 
                     sin(radians(%s)) * sin(radians(p.latitude)))) <= %s
                """.strip())
                parameters[f'center_lat_{param_counter}'] = spatial.center_lat
                parameters[f'center_lon_{param_counter}'] = spatial.center_lon
                parameters[f'center_lat2_{param_counter}'] = spatial.center_lat
                parameters[f'radius_{param_counter}'] = spatial.radius_km
                param_counter += 1
        
        # Temporal constraints
        for temporal in query_structure.temporal_constraints:
            if temporal.start_date and temporal.end_date:
                conditions.append("p.measurement_date BETWEEN %s AND %s")
                parameters[f'start_date_{param_counter}'] = temporal.start_date
                parameters[f'end_date_{param_counter}'] = temporal.end_date
                param_counter += 1
            elif temporal.start_date:
                conditions.append("p.measurement_date >= %s")
                parameters[f'start_date_{param_counter}'] = temporal.start_date
                param_counter += 1
            elif temporal.end_date:
                conditions.append("p.measurement_date <= %s")
                parameters[f'end_date_{param_counter}'] = temporal.end_date
                param_counter += 1
            
            if temporal.season:
                # Add seasonal filtering
                season_months = self._get_season_months(temporal.season)
                if season_months:
                    month_conditions = " OR ".join([f"EXTRACT(MONTH FROM p.measurement_date) = {month}" for month in season_months])
                    conditions.append(f"({month_conditions})")
        
        # Parameter constraints  
        for param_constraint in query_structure.parameter_constraints:
            column_name = self.schema.get_measurement_column(param_constraint.parameter_name)
            if not column_name:
                continue
                
            if param_constraint.constraint_type == 'range':
                conditions.append(f"m.{column_name} BETWEEN %s AND %s")
                parameters[f'{param_constraint.parameter_name}_min_{param_counter}'] = param_constraint.min_value
                parameters[f'{param_constraint.parameter_name}_max_{param_counter}'] = param_constraint.max_value
                param_counter += 1
            elif param_constraint.constraint_type == 'greater_than':
                conditions.append(f"m.{column_name} > %s")
                parameters[f'{param_constraint.parameter_name}_min_{param_counter}'] = param_constraint.min_value
                param_counter += 1
            elif param_constraint.constraint_type == 'less_than':
                conditions.append(f"m.{column_name} < %s")
                parameters[f'{param_constraint.parameter_name}_max_{param_counter}'] = param_constraint.max_value
                param_counter += 1
            elif param_constraint.constraint_type == 'equals':
                conditions.append(f"m.{column_name} = %s")
                parameters[f'{param_constraint.parameter_name}_val_{param_counter}'] = param_constraint.target_value
                param_counter += 1
            
            # Add quality constraint if specified
            if param_constraint.quality_threshold:
                qc_column = f"{column_name.split('_')[0]}_qc"
                conditions.append(f"m.{qc_column} IN ('1', '2')")
        
        # Default quality filters for good data
        conditions.append("(m.temperature_qc IN ('1', '2') OR m.temperature_qc IS NULL)")
        conditions.append("(m.salinity_qc IN ('1', '2') OR m.salinity_qc IS NULL)")
        
        return " AND ".join(conditions), parameters
    
    def _build_groupby_clause(self, query_structure: QueryStructure) -> str:
        """Build GROUP BY clause."""
        if not query_structure.groupby_fields and not query_structure.aggregation_type:
            return ""
        
        if query_structure.groupby_fields:
            return f"GROUP BY {', '.join(query_structure.groupby_fields)}"
        
        return ""
    
    def _build_orderby_clause(self, query_structure: QueryStructure) -> str:
        """Build ORDER BY clause."""
        if not query_structure.orderby_fields:
            # Default ordering
            return "ORDER BY p.measurement_date DESC, f.platform_number, m.depth_m"
        
        order_fields = []
        for field in query_structure.orderby_fields:
            if field == 'measurement_date':
                order_fields.append("p.measurement_date DESC")
            elif field == 'depth':
                order_fields.append("m.depth_m")
            elif field == 'platform_number':
                order_fields.append("f.platform_number")
            else:
                order_fields.append(field)
        
        return f"ORDER BY {', '.join(order_fields)}"
    
    def _build_limit_clause(self, query_structure: QueryStructure) -> str:
        """Build LIMIT clause."""
        limit = query_structure.limit or 1000  # Default limit
        limit = min(limit, 50000)  # Cap at 50k for performance
        return f"LIMIT {limit}"
    
    def _build_aggregation_columns(self, query_structure: QueryStructure) -> str:
        """Build aggregation columns for GROUP BY queries."""
        if not query_structure.aggregation_type:
            return "COUNT(*) as record_count"
        
        agg_type = query_structure.aggregation_type.upper()
        
        aggregations = []
        for param in query_structure.parameters or ['temperature', 'salinity']:
            column = self.schema.get_measurement_column(param)
            if column:
                aggregations.append(f"{agg_type}(m.{column}) as {agg_type.lower()}_{param}")
        
        if not aggregations:
            aggregations.append("COUNT(*) as record_count")
        
        return ",\n                    ".join(aggregations)
    
    def _determine_time_granularity(self, query_structure: QueryStructure) -> str:
        """Determine appropriate time granularity for temporal analysis."""
        # Check temporal constraints to determine granularity
        for temporal in query_structure.temporal_constraints:
            if temporal.season or temporal.relative_period:
                if 'month' in str(temporal.relative_period) or temporal.season:
                    return 'month'
                elif 'year' in str(temporal.relative_period):
                    return 'year'
                elif 'day' in str(temporal.relative_period):
                    return 'day'
        
        return 'month'  # Default to monthly granularity
    
    def _split_comparison_conditions(self, query_structure: QueryStructure) -> Tuple[str, str]:
        """Split conditions for comparative analysis."""
        # This is a simplified implementation - in production, would need more sophisticated splitting
        base_conditions, parameters = self._build_conditions(query_structure)
        
        # For now, return the same conditions for both datasets
        # In a real implementation, would parse comparative aspects
        return base_conditions, base_conditions
    
    def _clean_and_optimize_sql(self, sql: str) -> str:
        """Clean up and optimize generated SQL."""
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Format for readability
        sql = sql.replace('SELECT', '\nSELECT')
        sql = sql.replace('FROM', '\nFROM')
        sql = sql.replace('WHERE', '\nWHERE')
        sql = sql.replace('GROUP BY', '\nGROUP BY')
        sql = sql.replace('ORDER BY', '\nORDER BY')
        sql = sql.replace('LIMIT', '\nLIMIT')
        
        return sql.strip()
    
    def _get_season_months(self, season: str) -> List[int]:
        """Get month numbers for season."""
        season_mapping = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
            'monsoon': [6, 7, 8, 9],
            'pre-monsoon': [3, 4, 5],
            'post-monsoon': [10, 11, 12]
        }
        
        return season_mapping.get(season.lower(), [])
    
    def _estimate_query_complexity(self, sql: str, query_structure: QueryStructure) -> str:
        """Estimate query execution complexity."""
        complexity_score = 0
        
        # Base query complexity
        complexity_score += self.complexity_weights['base_query']
        
        # JOIN complexity
        join_count = len(re.findall(r'\bjoin\b', sql.lower()))
        complexity_score += join_count * self.complexity_weights['join_count']
        
        # WHERE conditions complexity
        condition_count = len(re.findall(r'\band\b|\bor\b', sql.lower()))
        complexity_score += condition_count * self.complexity_weights['where_conditions']
        
        # Aggregation complexity
        if query_structure.aggregation_type:
            complexity_score += self.complexity_weights['aggregations']
        
        # Spatial operations
        if query_structure.spatial_constraints:
            complexity_score += self.complexity_weights['spatial_operations']
        
        # Temporal operations
        if query_structure.temporal_constraints:
            complexity_score += self.complexity_weights['temporal_operations']
        
        if complexity_score < 0.3:
            return 'simple'
        elif complexity_score < 0.8:
            return 'moderate'
        else:
            return 'complex'
    
    def _estimate_execution_time(self, sql: str, query_structure: QueryStructure) -> float:
        """Estimate query execution time in seconds."""
        base_time = 0.5  # Base query time
        
        # Adjust based on complexity factors
        if query_structure.spatial_constraints:
            base_time += 1.0  # Spatial operations are expensive
        
        if query_structure.aggregation_type:
            base_time += 0.5
        
        if query_structure.limit and query_structure.limit > 10000:
            base_time += 2.0
        
        join_count = len(re.findall(r'\bjoin\b', sql.lower()))
        base_time += join_count * 0.3
        
        return base_time
    
    def _estimate_result_size(self, query_structure: QueryStructure) -> int:
        """Estimate result set size."""
        base_size = 1000
        
        # Adjust based on constraints
        if query_structure.spatial_constraints:
            base_size = int(base_size * 0.3)  # Spatial filtering reduces size
        
        if query_structure.temporal_constraints:
            base_size = int(base_size * 0.5)  # Temporal filtering reduces size
        
        if query_structure.parameter_constraints:
            base_size = int(base_size * 0.4)  # Parameter filtering reduces size
        
        if query_structure.aggregation_type:
            base_size = int(base_size * 0.1)  # Aggregation significantly reduces size
        
        if query_structure.limit:
            base_size = min(base_size, query_structure.limit)
        
        return max(1, base_size)
    
    def _generate_optimization_notes(self, sql: str, query_structure: QueryStructure) -> List[str]:
        """Generate optimization recommendations."""
        notes = []
        
        if len(query_structure.spatial_constraints) > 0:
            notes.append("Spatial index on (latitude, longitude) will improve performance")
        
        if len(query_structure.temporal_constraints) > 0:
            notes.append("Temporal index on measurement_date will improve performance")
        
        if query_structure.aggregation_type:
            notes.append("Consider materialized views for frequently used aggregations")
        
        join_count = len(re.findall(r'\bjoin\b', sql.lower()))
        if join_count > 3:
            notes.append("Complex joins detected - consider query optimization")
        
        if not query_structure.limit or query_structure.limit > 10000:
            notes.append("Consider adding LIMIT clause to improve performance")
        
        return notes
    
    def _calculate_generation_confidence(self, query_structure: QueryStructure, sql: str) -> float:
        """Calculate confidence in generated SQL."""
        base_confidence = 0.7
        
        # Boost confidence based on successful component generation
        if query_structure.parameters:
            base_confidence += 0.1
        
        if query_structure.spatial_constraints:
            base_confidence += 0.1
        
        if query_structure.temporal_constraints:
            base_confidence += 0.1
        
        # Reduce confidence for complex scenarios
        if query_structure.intent == QueryIntent.COMPARISON:
            base_confidence -= 0.1  # Comparison queries are more complex
        
        if len(re.findall(r'\bjoin\b', sql.lower())) > 5:
            base_confidence -= 0.1  # Very complex joins
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_fallback_query(self, query_structure: QueryStructure, error_msg: str) -> GeneratedQuery:
        """Generate a safe fallback query when generation fails."""
        
        fallback_sql = """
        SELECT 
            f.platform_number,
            p.cycle_number,
            p.latitude,
            p.longitude,
            p.measurement_date,
            m.depth_m,
            m.temperature_c,
            m.salinity_psu
        FROM floats f
        JOIN profiles p ON f.id = p.float_id
        JOIN measurements m ON p.id = m.profile_id
        WHERE f.is_active = true
        ORDER BY p.measurement_date DESC
        LIMIT 100
        """
        
        return GeneratedQuery(
            sql=fallback_sql,
            parameters={},
            estimated_complexity='simple',
            execution_time_estimate=1.0,
            result_size_estimate=100,
            optimization_notes=['Fallback query - limited functionality'],
            safety_checks=[f'Original query generation failed: {error_msg}'],
            confidence_score=0.3
        )