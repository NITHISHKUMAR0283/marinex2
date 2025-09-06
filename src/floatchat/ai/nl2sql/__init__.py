"""
Natural Language to SQL Engine for Oceanographic Queries.
Converts natural language questions into optimized PostgreSQL queries for ARGO float data.
"""

from .query_parser import OceanographicQueryParser, QueryStructure, SpatialConstraint, TemporalConstraint, ParameterConstraint
from .sql_generator import OceanographicSQLGenerator, QueryTemplate, SQLGenerationResult
from .oceanographic_schema import OceanographicSchema, ParameterMapping, SchemaMetadata

class NL2SQLEngine:
    """Complete Natural Language to SQL engine for oceanographic data."""
    
    def __init__(self, db_config: dict):
        """Initialize the NL2SQL engine with database configuration."""
        self.schema = OceanographicSchema()
        self.parser = OceanographicQueryParser()
        self.generator = OceanographicSQLGenerator(self.schema)
        self.db_config = db_config
        
    async def process_query(self, natural_language_query: str) -> dict:
        """
        Process a natural language query and return SQL with metadata.
        
        Args:
            natural_language_query: User's question in natural language
            
        Returns:
            Dict containing SQL query, confidence score, and execution metadata
        """
        try:
            # Parse natural language into structured query
            query_structure = await self.parser.parse_query(natural_language_query)
            
            # Generate optimized SQL
            sql_result = await self.generator.generate_sql(query_structure)
            
            # Return comprehensive result
            return {
                'sql_query': sql_result.sql_query,
                'parameters': sql_result.parameters,
                'confidence_score': query_structure.confidence_score,
                'query_type': query_structure.query_type,
                'spatial_constraints': [c.to_dict() for c in query_structure.spatial_constraints],
                'temporal_constraints': [c.to_dict() for c in query_structure.temporal_constraints],
                'parameter_constraints': [c.to_dict() for c in query_structure.parameter_constraints],
                'estimated_rows': sql_result.estimated_rows,
                'performance_notes': sql_result.performance_notes,
                'original_query': natural_language_query
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'sql_query': None,
                'confidence_score': 0.0,
                'original_query': natural_language_query
            }
    
    def get_supported_parameters(self) -> list:
        """Get list of supported oceanographic parameters."""
        return list(self.schema.parameter_mappings.keys())
    
    def get_schema_info(self) -> dict:
        """Get comprehensive schema information for query assistance."""
        return {
            'parameters': self.get_supported_parameters(),
            'regions': list(self.schema.indian_ocean_regions.keys()),
            'depth_range': self.schema.depth_ranges,
            'sample_queries': [
                "Show temperature profiles near the equator",
                "What's the salinity in the Bay of Bengal last month?",
                "Find oxygen levels below 1000m in the Arabian Sea",
                "Compare temperature and salinity at 500m depth",
                "Show all measurements from float WMO 2902746"
            ]
        }

__all__ = [
    'NL2SQLEngine',
    'OceanographicQueryParser',
    'OceanographicSQLGenerator',
    'OceanographicSchema',
    'QueryStructure',
    'SpatialConstraint',
    'TemporalConstraint',
    'ParameterConstraint',
    'QueryTemplate',
    'SQLGenerationResult',
    'ParameterMapping',
    'SchemaMetadata'
]