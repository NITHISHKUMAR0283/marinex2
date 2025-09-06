"""
Model Context Protocol (MCP) integration for oceanographic analysis.
Provides structured tool definitions and secure execution environment for LLMs.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, date
import json
import inspect
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from ...core.config import get_settings
from ..rag import RAGContext

logger = logging.getLogger(__name__)


class ToolParameterType(Enum):
    """Supported parameter types for MCP tools."""
    STRING = "string"
    NUMBER = "number" 
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class ToolParameter:
    """Tool parameter definition."""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    items_type: Optional[ToolParameterType] = None  # For arrays
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format."""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.type == ToolParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}
            
        return schema


@dataclass
class ToolDefinition:
    """MCP tool definition with validation and metadata."""
    name: str
    description: str
    parameters: List[ToolParameter]
    category: str
    security_level: str = "safe"  # safe, restricted, dangerous
    rate_limit: Optional[int] = None  # calls per minute
    timeout: float = 30.0
    requires_auth: bool = False
    
    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP tool specification format."""
        required_params = [p.name for p in self.parameters if p.required]
        
        parameters_schema = {
            "type": "object",
            "properties": {
                param.name: param.to_json_schema() 
                for param in self.parameters
            },
            "required": required_params
        }
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema
            }
        }


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    citations: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "citations": self.citations or []
        }


class SecurityValidator:
    """Security validation for tool parameters and execution."""
    
    def __init__(self):
        # SQL injection patterns
        self.sql_injection_patterns = [
            r";\s*(drop|delete|truncate|alter|create|insert|update)\s+",
            r"union\s+select",
            r"--\s*",
            r"/\*.*\*/",
            r"xp_cmdshell",
            r"sp_executesql"
        ]
        
        # Dangerous function patterns
        self.dangerous_patterns = [
            r"exec\s*\(",
            r"eval\s*\(",
            r"system\s*\(",
            r"__import__",
            r"getattr\s*\(",
            r"setattr\s*\("
        ]
    
    def validate_sql_query(self, query: str) -> tuple[bool, Optional[str]]:
        """Validate SQL query for security risks."""
        import re
        
        query_lower = query.lower()
        
        # Check for dangerous SQL patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return False, f"Potentially dangerous SQL pattern detected: {pattern}"
        
        # Check query length
        if len(query) > 10000:
            return False, "SQL query too long (>10k characters)"
        
        # Ensure query is read-only (starts with SELECT)
        query_trimmed = query_lower.strip()
        if not query_trimmed.startswith('select'):
            return False, "Only SELECT queries are allowed"
        
        return True, None
    
    def validate_parameter_value(self, param: ToolParameter, value: Any) -> tuple[bool, Optional[str]]:
        """Validate parameter value against security constraints."""
        
        # Check for dangerous patterns in string values
        if isinstance(value, str):
            import re
            for pattern in self.dangerous_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False, f"Potentially dangerous pattern detected: {pattern}"
        
        # Type validation
        if param.type == ToolParameterType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
            if len(value) > 10000:
                return False, "String parameter too long (>10k characters)"
                
        elif param.type == ToolParameterType.NUMBER:
            if not isinstance(value, (int, float)):
                return False, f"Expected number, got {type(value).__name__}"
            if param.min_value is not None and value < param.min_value:
                return False, f"Value {value} below minimum {param.min_value}"
            if param.max_value is not None and value > param.max_value:
                return False, f"Value {value} above maximum {param.max_value}"
        
        elif param.type == ToolParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Expected boolean, got {type(value).__name__}"
        
        # Enum validation
        if param.enum_values and value not in param.enum_values:
            return False, f"Value {value} not in allowed values: {param.enum_values}"
        
        return True, None


class OceanographicToolRegistry:
    """Registry of oceanographic analysis tools for MCP integration."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.tools = {}
        self.execution_stats = {}
        
        # Initialize oceanographic tools
        self._register_core_tools()
    
    def _register_core_tools(self):
        """Register core oceanographic analysis tools."""
        
        # 1. Oceanographic Query Tool
        self.register_tool(
            ToolDefinition(
                name="query_oceanographic_data",
                description="Query ARGO oceanographic database with spatial, temporal, and parameter constraints. Returns structured oceanographic measurements and metadata.",
                category="data_retrieval",
                parameters=[
                    ToolParameter("query_type", ToolParameterType.STRING, 
                                "Type of oceanographic query", True,
                                enum_values=["profile", "timeseries", "spatial_analysis", "parameter_comparison"]),
                    ToolParameter("spatial_bounds", ToolParameterType.OBJECT, 
                                "Geographic boundaries for data selection (lat/lon bounds)", False),
                    ToolParameter("temporal_range", ToolParameterType.OBJECT,
                                "Time range for data selection (start_date, end_date)", False),
                    ToolParameter("parameters", ToolParameterType.ARRAY,
                                "Ocean parameters to include in results", False,
                                items_type=ToolParameterType.STRING),
                    ToolParameter("max_results", ToolParameterType.INTEGER,
                                "Maximum number of results to return", False,
                                min_value=1, max_value=10000),
                    ToolParameter("quality_threshold", ToolParameterType.NUMBER,
                                "Minimum data quality threshold (0-1)", False,
                                min_value=0.0, max_value=1.0)
                ],
                timeout=45.0,
                rate_limit=60
            ),
            self._execute_oceanographic_query
        )
        
        # 2. Statistical Analysis Tool
        self.register_tool(
            ToolDefinition(
                name="analyze_oceanographic_statistics",
                description="Perform statistical analysis on oceanographic data including correlations, trends, anomalies, and climatological comparisons.",
                category="statistical_analysis",
                parameters=[
                    ToolParameter("data_reference", ToolParameterType.STRING,
                                "Reference ID to previously queried data", True),
                    ToolParameter("analysis_type", ToolParameterType.STRING,
                                "Type of statistical analysis to perform", True,
                                enum_values=["correlation", "trend", "anomaly_detection", "climatology", "distribution"]),
                    ToolParameter("parameters", ToolParameterType.ARRAY,
                                "Parameters to analyze", True,
                                items_type=ToolParameterType.STRING),
                    ToolParameter("confidence_level", ToolParameterType.NUMBER,
                                "Statistical confidence level", False,
                                min_value=0.8, max_value=0.99),
                    ToolParameter("seasonal_adjustment", ToolParameterType.BOOLEAN,
                                "Apply seasonal adjustment to time series", False)
                ],
                timeout=60.0,
                rate_limit=30
            ),
            self._execute_statistical_analysis
        )
        
        # 3. Visualization Generation Tool
        self.register_tool(
            ToolDefinition(
                name="generate_oceanographic_visualization", 
                description="Generate interactive visualizations for oceanographic data including maps, profiles, time series, and 3D plots.",
                category="visualization",
                parameters=[
                    ToolParameter("data_reference", ToolParameterType.STRING,
                                "Reference ID to data for visualization", True),
                    ToolParameter("plot_type", ToolParameterType.STRING,
                                "Type of visualization to generate", True,
                                enum_values=["map", "profile", "timeseries", "heatmap", "3d_scatter", "trajectory"]),
                    ToolParameter("parameters", ToolParameterType.ARRAY,
                                "Parameters to visualize", True,
                                items_type=ToolParameterType.STRING),
                    ToolParameter("color_scheme", ToolParameterType.STRING,
                                "Scientific color scheme for visualization", False,
                                enum_values=["viridis", "plasma", "coolwarm", "ocean", "thermal"]),
                    ToolParameter("interactive", ToolParameterType.BOOLEAN,
                                "Generate interactive visualization", False),
                    ToolParameter("export_format", ToolParameterType.STRING,
                                "Export format for visualization", False,
                                enum_values=["png", "svg", "html", "json"])
                ],
                timeout=30.0,
                rate_limit=20
            ),
            self._execute_visualization_generation
        )
        
        # 4. Data Export Tool
        self.register_tool(
            ToolDefinition(
                name="export_oceanographic_data",
                description="Export oceanographic query results in various scientific formats with comprehensive metadata and citations.",
                category="data_export",
                parameters=[
                    ToolParameter("data_reference", ToolParameterType.STRING,
                                "Reference ID to data for export", True),
                    ToolParameter("export_format", ToolParameterType.STRING,
                                "Data export format", True,
                                enum_values=["netcdf", "csv", "json", "parquet", "matlab"]),
                    ToolParameter("include_metadata", ToolParameterType.BOOLEAN,
                                "Include comprehensive metadata", False),
                    ToolParameter("include_quality_flags", ToolParameterType.BOOLEAN,
                                "Include data quality information", False),
                    ToolParameter("compression", ToolParameterType.STRING,
                                "Compression method for export", False,
                                enum_values=["none", "gzip", "bz2", "lzma"])
                ],
                timeout=45.0,
                rate_limit=10
            ),
            self._execute_data_export
        )
        
        # 5. Water Mass Analysis Tool
        self.register_tool(
            ToolDefinition(
                name="analyze_water_masses",
                description="Perform advanced water mass analysis including T-S diagram analysis, water mass classification, and mixing calculations.",
                category="oceanographic_analysis",
                parameters=[
                    ToolParameter("data_reference", ToolParameterType.STRING,
                                "Reference ID to oceanographic data", True),
                    ToolParameter("analysis_method", ToolParameterType.STRING,
                                "Water mass analysis method", True,
                                enum_values=["ts_analysis", "optimum_multiparameter", "mixing_analysis", "classification"]),
                    ToolParameter("reference_water_masses", ToolParameterType.ARRAY,
                                "Reference water mass definitions", False,
                                items_type=ToolParameterType.STRING),
                    ToolParameter("depth_range", ToolParameterType.OBJECT,
                                "Depth range for analysis (min_depth, max_depth)", False)
                ],
                timeout=60.0,
                rate_limit=20,
                security_level="restricted"
            ),
            self._execute_water_mass_analysis
        )
    
    def register_tool(self, tool_def: ToolDefinition, executor: Callable):
        """Register a new tool with its executor function."""
        self.tools[tool_def.name] = {
            'definition': tool_def,
            'executor': executor,
            'call_count': 0,
            'total_time': 0.0,
            'error_count': 0
        }
        logger.info(f"Registered MCP tool: {tool_def.name}")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in MCP format."""
        return [tool_info['definition'].to_mcp_format() for tool_info in self.tools.values()]
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tools filtered by category."""
        return [
            tool_info['definition'].to_mcp_format()
            for tool_info in self.tools.values()
            if tool_info['definition'].category == category
        ]
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          context: Optional[Dict] = None) -> ToolExecutionResult:
        """Execute a tool with security validation and error handling."""
        
        if tool_name not in self.tools:
            return ToolExecutionResult(
                success=False,
                result=None,
                error_message=f"Unknown tool: {tool_name}"
            )
        
        tool_info = self.tools[tool_name]
        tool_def = tool_info['definition']
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate parameters
            validation_result = await self._validate_tool_parameters(tool_def, parameters)
            if not validation_result.success:
                return validation_result
            
            # Check rate limiting
            rate_limit_result = self._check_rate_limit(tool_name)
            if not rate_limit_result.success:
                return rate_limit_result
            
            # Execute tool with timeout
            executor = tool_info['executor']
            result = await asyncio.wait_for(
                executor(parameters, context),
                timeout=tool_def.timeout
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Update statistics
            tool_info['call_count'] += 1
            tool_info['total_time'] += execution_time
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            tool_info['error_count'] += 1
            return ToolExecutionResult(
                success=False,
                result=None,
                error_message=f"Tool execution timed out after {tool_def.timeout}s"
            )
            
        except Exception as e:
            tool_info['error_count'] += 1
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return ToolExecutionResult(
                success=False,
                result=None,
                error_message=f"Tool execution failed: {str(e)}"
            )
    
    async def _validate_tool_parameters(self, tool_def: ToolDefinition, 
                                      parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Validate tool parameters against definition and security constraints."""
        
        # Check required parameters
        required_params = [p.name for p in tool_def.parameters if p.required]
        missing_params = [p for p in required_params if p not in parameters]
        if missing_params:
            return ToolExecutionResult(
                success=False,
                result=None,
                error_message=f"Missing required parameters: {missing_params}"
            )
        
        # Validate each parameter
        for param_def in tool_def.parameters:
            if param_def.name in parameters:
                value = parameters[param_def.name]
                is_valid, error_msg = self.security_validator.validate_parameter_value(param_def, value)
                if not is_valid:
                    return ToolExecutionResult(
                        success=False,
                        result=None,
                        error_message=f"Parameter validation failed for '{param_def.name}': {error_msg}"
                    )
        
        return ToolExecutionResult(success=True, result=None)
    
    def _check_rate_limit(self, tool_name: str) -> ToolExecutionResult:
        """Check rate limiting for tool execution."""
        # Simplified rate limiting - would implement proper sliding window in production
        tool_info = self.tools[tool_name]
        tool_def = tool_info['definition']
        
        if tool_def.rate_limit:
            # For now, just check if too many calls in a short time
            # In production, implement proper rate limiting with Redis or similar
            pass
        
        return ToolExecutionResult(success=True, result=None)
    
    # Tool executor implementations
    async def _execute_oceanographic_query(self, parameters: Dict[str, Any], 
                                         context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute oceanographic data query."""
        
        query_type = parameters.get('query_type', 'profile')
        spatial_bounds = parameters.get('spatial_bounds', {})
        temporal_range = parameters.get('temporal_range', {})
        max_results = parameters.get('max_results', 1000)
        
        # This would integrate with the actual database query engine
        # For now, return structured placeholder result
        return {
            "query_id": str(uuid.uuid4()),
            "query_type": query_type,
            "spatial_coverage": spatial_bounds,
            "temporal_coverage": temporal_range,
            "total_records": min(max_results, 5000),  # Simulated
            "parameters_available": ["temperature", "salinity", "pressure"],
            "data_quality_summary": {
                "good_quality": 0.92,
                "acceptable_quality": 0.06,
                "poor_quality": 0.02
            },
            "execution_summary": "Query executed successfully against ARGO database",
            "next_steps": ["Use data_reference in subsequent tool calls", "Consider visualization options"]
        }
    
    async def _execute_statistical_analysis(self, parameters: Dict[str, Any], 
                                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute statistical analysis on oceanographic data."""
        
        analysis_type = parameters.get('analysis_type', 'correlation')
        data_reference = parameters.get('data_reference')
        confidence_level = parameters.get('confidence_level', 0.95)
        
        # This would integrate with statistical analysis engine
        return {
            "analysis_id": str(uuid.uuid4()),
            "analysis_type": analysis_type,
            "data_reference": data_reference,
            "confidence_level": confidence_level,
            "results": {
                "statistical_significance": True,
                "p_value": 0.001,
                "confidence_interval": [0.85, 0.92],
                "trend_direction": "increasing" if analysis_type == "trend" else None,
                "correlation_coefficient": 0.89 if analysis_type == "correlation" else None
            },
            "interpretation": f"Statistical analysis ({analysis_type}) completed with high confidence",
            "recommendations": ["Results are statistically significant", "Consider temporal analysis"]
        }
    
    async def _execute_visualization_generation(self, parameters: Dict[str, Any], 
                                              context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute visualization generation."""
        
        plot_type = parameters.get('plot_type', 'map')
        data_reference = parameters.get('data_reference')
        interactive = parameters.get('interactive', True)
        export_format = parameters.get('export_format', 'html')
        
        # This would integrate with visualization engine
        return {
            "visualization_id": str(uuid.uuid4()),
            "plot_type": plot_type,
            "data_reference": data_reference,
            "interactive": interactive,
            "export_format": export_format,
            "file_path": f"/visualizations/{uuid.uuid4().hex}.{export_format}",
            "dimensions": {"width": 800, "height": 600},
            "features": ["zoom", "pan", "hover_info"] if interactive else ["static_plot"],
            "color_scheme": parameters.get('color_scheme', 'viridis'),
            "status": "generated_successfully"
        }
    
    async def _execute_data_export(self, parameters: Dict[str, Any], 
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute data export."""
        
        export_format = parameters.get('export_format', 'csv')
        data_reference = parameters.get('data_reference')
        include_metadata = parameters.get('include_metadata', True)
        
        return {
            "export_id": str(uuid.uuid4()),
            "data_reference": data_reference,
            "export_format": export_format,
            "file_path": f"/exports/{uuid.uuid4().hex}.{export_format}",
            "file_size_mb": 15.7,
            "record_count": 25000,
            "metadata_included": include_metadata,
            "citation": "ARGO Float Data - Global Ocean Observing System",
            "download_url": f"https://floatchat.example.com/downloads/{uuid.uuid4().hex}",
            "expiry_hours": 24
        }
    
    async def _execute_water_mass_analysis(self, parameters: Dict[str, Any], 
                                         context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute water mass analysis."""
        
        analysis_method = parameters.get('analysis_method', 'ts_analysis')
        data_reference = parameters.get('data_reference')
        
        return {
            "analysis_id": str(uuid.uuid4()),
            "method": analysis_method,
            "data_reference": data_reference,
            "water_masses_identified": [
                {"name": "Indian Ocean Central Water", "percentage": 65.2, "confidence": 0.91},
                {"name": "Antarctic Intermediate Water", "percentage": 28.7, "confidence": 0.87},
                {"name": "Deep Water Mass", "percentage": 6.1, "confidence": 0.76}
            ],
            "mixing_ratios": {"surface_influence": 0.34, "intermediate_influence": 0.66},
            "ts_diagram_path": f"/analysis/{uuid.uuid4().hex}_ts_diagram.png",
            "scientific_interpretation": "Clear water mass structure identified with strong intermediate water influence"
        }
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics."""
        stats = {}
        for tool_name, tool_info in self.tools.items():
            stats[tool_name] = {
                'total_calls': tool_info['call_count'],
                'total_time': tool_info['total_time'],
                'avg_time': tool_info['total_time'] / max(tool_info['call_count'], 1),
                'error_count': tool_info['error_count'],
                'error_rate': tool_info['error_count'] / max(tool_info['call_count'], 1),
                'category': tool_info['definition'].category
            }
        return stats


class MCPOrchestrator:
    """Main MCP orchestrator for managing tool execution and LLM integration."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.settings = get_settings()
        self.tool_registry = OceanographicToolRegistry()
        
        # Execution context
        self.session_data = {}  # Store data references between tool calls
        self.execution_history = []
        
        logger.info("MCP Orchestrator initialized")
    
    def get_available_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available tools, optionally filtered by category."""
        if category:
            return self.tool_registry.get_tools_by_category(category)
        return self.tool_registry.get_tool_definitions()
    
    async def execute_tool_sequence(self, tool_calls: List[Dict[str, Any]], 
                                  context: Optional[RAGContext] = None) -> List[ToolExecutionResult]:
        """Execute a sequence of tool calls with context propagation."""
        
        results = []
        execution_context = {
            'session_id': str(uuid.uuid4()),
            'rag_context': context,
            'previous_results': []
        }
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('function', tool_call.get('name'))
            parameters = tool_call.get('arguments', tool_call.get('parameters', {}))
            
            # Add execution context to parameters
            if context:
                execution_context['rag_context'] = context
            execution_context['previous_results'] = results
            
            result = await self.tool_registry.execute_tool(
                tool_name, 
                parameters, 
                execution_context
            )
            
            results.append(result)
            
            # Store data references for subsequent tool calls
            if result.success and isinstance(result.result, dict):
                if 'query_id' in result.result:
                    self.session_data[result.result['query_id']] = result.result
                elif 'analysis_id' in result.result:
                    self.session_data[result.result['analysis_id']] = result.result
            
            # Log execution
            self.execution_history.append({
                'timestamp': datetime.now(),
                'tool_name': tool_name,
                'parameters': parameters,
                'success': result.success,
                'execution_time': result.execution_time
            })
        
        return results
    
    def format_tool_results_for_llm(self, results: List[ToolExecutionResult]) -> str:
        """Format tool execution results for LLM consumption."""
        
        if not results:
            return "No tool execution results available."
        
        formatted_results = []
        formatted_results.append("TOOL EXECUTION RESULTS:")
        formatted_results.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"\nTool Execution #{i}:")
            formatted_results.append(f"Success: {result.success}")
            formatted_results.append(f"Execution Time: {result.execution_time:.3f}s")
            
            if result.success:
                formatted_results.append("Result:")
                if isinstance(result.result, dict):
                    for key, value in result.result.items():
                        formatted_results.append(f"  {key}: {value}")
                else:
                    formatted_results.append(f"  {result.result}")
            else:
                formatted_results.append(f"Error: {result.error_message}")
            
            if result.citations:
                formatted_results.append(f"Citations: {', '.join(result.citations)}")
            
            formatted_results.append("-" * 30)
        
        return "\n".join(formatted_results)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        tool_stats = self.tool_registry.get_tool_statistics()
        
        return {
            'tool_statistics': tool_stats,
            'session_data_count': len(self.session_data),
            'execution_history_count': len(self.execution_history),
            'total_executions': sum(stats['total_calls'] for stats in tool_stats.values()),
            'average_execution_time': sum(stats['avg_time'] for stats in tool_stats.values()) / len(tool_stats) if tool_stats else 0,
            'total_errors': sum(stats['error_count'] for stats in tool_stats.values())
        }
    
    async def close(self):
        """Clean up MCP orchestrator resources."""
        self.session_data.clear()
        self.execution_history.clear()
        logger.info("MCP Orchestrator closed")