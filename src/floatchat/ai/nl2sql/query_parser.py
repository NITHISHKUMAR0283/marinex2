"""
Advanced Natural Language Query Parser for oceanographic database queries.
Handles complex spatial, temporal, and parameter constraints with domain expertise.
"""

import re
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import json

# For advanced NLP processing
try:
    import spacy
    from dateutil import parser as date_parser
    from geopy.geocoders import Nominatim
except ImportError as e:
    print(f"Optional NLP dependencies not found: {e}")
    print("Install with: pip install spacy dateutil geopy")

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of oceanographic query intents."""
    DATA_RETRIEVAL = "data_retrieval"
    COMPARISON = "comparison" 
    TREND_ANALYSIS = "trend_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    SPATIAL_ANALYSIS = "spatial_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    WATER_MASS_ANALYSIS = "water_mass_analysis"
    EXPORT_REQUEST = "export_request"
    VISUALIZATION_REQUEST = "visualization_request"


@dataclass
class SpatialConstraint:
    """Spatial constraint for oceanographic queries."""
    constraint_type: str  # 'bounding_box', 'circle', 'polygon', 'named_region'
    coordinates: Optional[List[float]] = None  # [lat1, lon1, lat2, lon2] for bbox
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_km: Optional[float] = None
    region_name: Optional[str] = None
    polygon_coords: Optional[List[Tuple[float, float]]] = None
    depth_range: Optional[Tuple[float, float]] = None


@dataclass 
class TemporalConstraint:
    """Temporal constraint for oceanographic queries."""
    constraint_type: str  # 'range', 'before', 'after', 'season', 'year'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    season: Optional[str] = None  # 'winter', 'spring', 'summer', 'autumn'
    year: Optional[int] = None
    relative_period: Optional[str] = None  # 'last_month', 'last_year', 'recent'


@dataclass
class ParameterConstraint:
    """Parameter constraint for oceanographic data."""
    parameter_name: str
    constraint_type: str  # 'range', 'greater_than', 'less_than', 'equals', 'not_null'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    quality_threshold: Optional[str] = None  # '1', '2', '3' for QC flags


@dataclass
class QueryStructure:
    """Structured representation of parsed oceanographic query."""
    intent: QueryIntent
    parameters: List[str] = field(default_factory=list)
    spatial_constraints: List[SpatialConstraint] = field(default_factory=list)
    temporal_constraints: List[TemporalConstraint] = field(default_factory=list)
    parameter_constraints: List[ParameterConstraint] = field(default_factory=list)
    aggregation_type: Optional[str] = None  # 'avg', 'min', 'max', 'count', 'sum'
    groupby_fields: List[str] = field(default_factory=list)
    orderby_fields: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    output_format: Optional[str] = None
    confidence_score: float = 0.0


class OceanographicGazetteer:
    """Marine location and region name resolver."""
    
    def __init__(self):
        # Predefined ocean regions with boundaries
        self.ocean_regions = {
            # Indian Ocean regions
            'arabian_sea': {
                'bbox': [8.0, 50.0, 30.0, 80.0],  # [min_lat, min_lon, max_lat, max_lon]
                'center': [19.0, 65.0],
                'aliases': ['arabian sea', 'arabian basin', 'arabian ocean']
            },
            'bay_of_bengal': {
                'bbox': [2.0, 80.0, 25.0, 100.0], 
                'center': [13.5, 90.0],
                'aliases': ['bay of bengal', 'bengal bay', 'bengali sea']
            },
            'indian_ocean': {
                'bbox': [-50.0, 20.0, 30.0, 120.0],
                'center': [-10.0, 70.0], 
                'aliases': ['indian ocean', 'indian sea']
            },
            'central_indian_ocean': {
                'bbox': [-20.0, 50.0, 10.0, 90.0],
                'center': [-5.0, 70.0],
                'aliases': ['central indian ocean', 'central indian basin']
            },
            'equatorial_indian_ocean': {
                'bbox': [-10.0, 40.0, 10.0, 100.0],
                'center': [0.0, 70.0],
                'aliases': ['equatorial indian ocean', 'equatorial indian', 'equator indian ocean']
            },
            'andaman_sea': {
                'bbox': [5.0, 92.0, 20.0, 100.0],
                'center': [12.5, 96.0],
                'aliases': ['andaman sea', 'andaman basin']
            },
            
            # Global reference points
            'equator': {
                'bbox': [-2.0, -180.0, 2.0, 180.0],
                'center': [0.0, 0.0],
                'aliases': ['equator', 'equatorial region', 'equatorial waters']
            },
            'tropics': {
                'bbox': [-23.5, -180.0, 23.5, 180.0],
                'center': [0.0, 0.0],
                'aliases': ['tropics', 'tropical waters', 'tropical region']
            }
        }
        
        # Major coastal cities and their approximate ocean coordinates
        self.coastal_locations = {
            'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'ocean_offset': [0.5, 0.5]},
            'chennai': {'lat': 13.0827, 'lon': 80.2707, 'ocean_offset': [0.3, 0.3]},
            'kochi': {'lat': 9.9312, 'lon': 76.2673, 'ocean_offset': [0.2, 0.2]},
            'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'ocean_offset': [0.3, 0.3]},
            'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'ocean_offset': [0.5, 0.5]}
        }
        
        # Initialize geocoder (optional)
        self.geocoder = None
        try:
            self.geocoder = Nominatim(user_agent="floatchat_oceanographic")
        except Exception as e:
            logger.warning(f"Geocoder initialization failed: {e}")
    
    def resolve_location(self, location_text: str) -> Optional[SpatialConstraint]:
        """Resolve location name to spatial constraint."""
        location_lower = location_text.lower().strip()
        
        # Check ocean regions first
        for region_key, region_data in self.ocean_regions.items():
            if any(alias in location_lower for alias in region_data['aliases']):
                return SpatialConstraint(
                    constraint_type='bounding_box',
                    coordinates=region_data['bbox'],
                    region_name=region_key
                )
        
        # Check coastal cities
        for city_key, city_data in self.coastal_locations.items():
            if city_key in location_lower:
                # Create ocean area near the city
                lat_offset, lon_offset = city_data['ocean_offset']
                ocean_lat = city_data['lat'] + lat_offset
                ocean_lon = city_data['lon'] + lon_offset
                
                return SpatialConstraint(
                    constraint_type='circle',
                    center_lat=ocean_lat,
                    center_lon=ocean_lon,
                    radius_km=100.0,  # 100km radius around city
                    region_name=f"near_{city_key}"
                )
        
        # Try geocoding if available
        if self.geocoder:
            try:
                location = self.geocoder.geocode(location_text)
                if location:
                    return SpatialConstraint(
                        constraint_type='circle',
                        center_lat=location.latitude,
                        center_lon=location.longitude,
                        radius_km=50.0,  # Default 50km radius
                        region_name=f"geocoded_{location_text.replace(' ', '_')}"
                    )
            except Exception as e:
                logger.warning(f"Geocoding failed for '{location_text}': {e}")
        
        return None


class TemporalParser:
    """Advanced temporal expression parser for oceanographic queries."""
    
    def __init__(self):
        # Temporal patterns and their handlers
        self.temporal_patterns = {
            # Absolute dates
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})': self._parse_absolute_date,
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})': self._parse_absolute_date,
            
            # Year patterns  
            r'(?:in|during|for)\s+(\d{4})': self._parse_year,
            r'year\s+(\d{4})': self._parse_year,
            
            # Month/Season patterns
            r'(?:in|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december)': self._parse_month,
            r'(?:in|during)\s+(winter|spring|summer|autumn|fall)': self._parse_season,
            r'(?:in|during)\s+(\d{4})\s+(monsoon|pre-monsoon|post-monsoon)': self._parse_monsoon_season,
            
            # Relative periods
            r'(?:last|past|previous)\s+(\d+)\s+(days?|weeks?|months?|years?)': self._parse_relative_period,
            r'(?:recent|lately)': self._parse_recent,
            r'(?:since|from)\s+(\d{4})': self._parse_since_year,
            
            # Range patterns
            r'(?:between|from)\s+(.+?)\s+(?:and|to)\s+(.+?)(?:\s|$)': self._parse_date_range,
        }
        
        # Season definitions for oceanography
        self.season_months = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5], 
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
            'fall': [9, 10, 11],
            'monsoon': [6, 7, 8, 9],  # Indian Ocean monsoon
            'pre-monsoon': [3, 4, 5],
            'post-monsoon': [10, 11, 12]
        }
    
    def parse_temporal_expressions(self, query: str) -> List[TemporalConstraint]:
        """Parse temporal expressions from query text."""
        constraints = []
        query_lower = query.lower()
        
        for pattern, handler in self.temporal_patterns.items():
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                try:
                    constraint = handler(match)
                    if constraint:
                        constraints.append(constraint)
                except Exception as e:
                    logger.warning(f"Failed to parse temporal expression '{match.group()}': {e}")
        
        return constraints
    
    def _parse_absolute_date(self, match) -> Optional[TemporalConstraint]:
        """Parse absolute date strings."""
        date_str = match.group(1)
        try:
            parsed_date = date_parser.parse(date_str)
            return TemporalConstraint(
                constraint_type='range',
                start_date=parsed_date,
                end_date=parsed_date
            )
        except Exception:
            return None
    
    def _parse_year(self, match) -> Optional[TemporalConstraint]:
        """Parse year specifications."""
        year = int(match.group(1))
        return TemporalConstraint(
            constraint_type='year',
            year=year,
            start_date=datetime(year, 1, 1),
            end_date=datetime(year, 12, 31)
        )
    
    def _parse_month(self, match) -> Optional[TemporalConstraint]:
        """Parse month specifications."""
        month_name = match.group(1).lower()
        month_num = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }.get(month_name)
        
        if month_num:
            # Use current year or most recent year
            current_year = datetime.now().year
            return TemporalConstraint(
                constraint_type='range',
                start_date=datetime(current_year, month_num, 1),
                end_date=datetime(current_year, month_num, 28)  # Simplified
            )
        return None
    
    def _parse_season(self, match) -> Optional[TemporalConstraint]:
        """Parse seasonal specifications."""
        season = match.group(1).lower()
        months = self.season_months.get(season)
        
        if months:
            return TemporalConstraint(
                constraint_type='season',
                season=season
            )
        return None
    
    def _parse_monsoon_season(self, match) -> Optional[TemporalConstraint]:
        """Parse monsoon season with year."""
        year = int(match.group(1))
        season_type = match.group(2)
        
        months = self.season_months.get(season_type, [6, 7, 8, 9])
        start_month = min(months)
        end_month = max(months)
        
        return TemporalConstraint(
            constraint_type='range',
            start_date=datetime(year, start_month, 1),
            end_date=datetime(year, end_month, 28),
            season=f"{year}_{season_type}"
        )
    
    def _parse_relative_period(self, match) -> Optional[TemporalConstraint]:
        """Parse relative time periods."""
        amount = int(match.group(1))
        unit = match.group(2).rstrip('s')  # Remove plural 's'
        
        now = datetime.now()
        if unit == 'day':
            start_date = now - timedelta(days=amount)
        elif unit == 'week':
            start_date = now - timedelta(weeks=amount)
        elif unit == 'month':
            start_date = now - timedelta(days=amount * 30)  # Approximate
        elif unit == 'year':
            start_date = now - timedelta(days=amount * 365)  # Approximate
        else:
            return None
        
        return TemporalConstraint(
            constraint_type='range',
            start_date=start_date,
            end_date=now,
            relative_period=f"last_{amount}_{unit}s"
        )
    
    def _parse_recent(self, match) -> Optional[TemporalConstraint]:
        """Parse 'recent' time references."""
        now = datetime.now()
        start_date = now - timedelta(days=90)  # Last 3 months
        
        return TemporalConstraint(
            constraint_type='range',
            start_date=start_date,
            end_date=now,
            relative_period='recent'
        )
    
    def _parse_since_year(self, match) -> Optional[TemporalConstraint]:
        """Parse 'since year' patterns."""
        year = int(match.group(1))
        start_date = datetime(year, 1, 1)
        end_date = datetime.now()
        
        return TemporalConstraint(
            constraint_type='range',
            start_date=start_date,
            end_date=end_date
        )
    
    def _parse_date_range(self, match) -> Optional[TemporalConstraint]:
        """Parse date ranges."""
        start_str = match.group(1).strip()
        end_str = match.group(2).strip()
        
        try:
            start_date = date_parser.parse(start_str)
            end_date = date_parser.parse(end_str)
            
            return TemporalConstraint(
                constraint_type='range',
                start_date=start_date,
                end_date=end_date
            )
        except Exception:
            return None


class OceanographicQueryParser:
    """Advanced parser for oceanographic natural language queries."""
    
    def __init__(self):
        self.gazetteer = OceanographicGazetteer()
        self.temporal_parser = TemporalParser()
        
        # Load spaCy model if available
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Oceanographic parameters and their aliases
        self.parameter_aliases = {
            'temperature': ['temp', 'temperature', 'thermal', 'heat', 'warming', 'cooling'],
            'salinity': ['salt', 'salinity', 'sal', 'saltiness', 'saline'],
            'pressure': ['pressure', 'depth', 'deep', 'shallow', 'pres'],
            'oxygen': ['oxygen', 'o2', 'dissolved oxygen', 'do'],
            'density': ['density', 'sigma', 'rho'],
            'ph': ['ph', 'acidity', 'alkalinity', 'acid'],
            'chlorophyll': ['chl', 'chlorophyll', 'phytoplankton', 'algae'],
            'nitrate': ['nitrate', 'no3', 'nitrogen', 'nutrients'],
            'phosphate': ['phosphate', 'po4', 'phosphorus']
        }
        
        # Query intent patterns
        self.intent_patterns = {
            QueryIntent.DATA_RETRIEVAL: [
                r'\b(?:show|display|get|find|retrieve|list)\b',
                r'\b(?:data|measurements|records|values)\b'
            ],
            QueryIntent.COMPARISON: [
                r'\b(?:compare|contrast|difference|versus|vs|between)\b',
                r'\b(?:higher|lower|greater|less|more|fewer)\b'
            ],
            QueryIntent.TREND_ANALYSIS: [
                r'\b(?:trend|trending|change|changing|increase|decrease)\b',
                r'\b(?:over time|temporal|time series|historical)\b'
            ],
            QueryIntent.STATISTICAL_ANALYSIS: [
                r'\b(?:average|mean|median|std|statistics|correlation)\b',
                r'\b(?:analyze|analysis|statistical)\b'
            ],
            QueryIntent.SPATIAL_ANALYSIS: [
                r'\b(?:spatial|geographic|regional|distribution|pattern)\b',
                r'\b(?:map|mapping|location|where|region)\b'
            ],
            QueryIntent.ANOMALY_DETECTION: [
                r'\b(?:anomaly|anomalies|unusual|abnormal|extreme)\b',
                r'\b(?:outlier|outliers|deviation)\b'
            ],
            QueryIntent.VISUALIZATION_REQUEST: [
                r'\b(?:plot|chart|graph|visualize|visualization|map)\b',
                r'\b(?:show|display)\b.*\b(?:plot|chart|graph|map)\b'
            ],
            QueryIntent.EXPORT_REQUEST: [
                r'\b(?:export|download|save|extract)\b',
                r'\b(?:csv|json|netcdf|excel)\b'
            ]
        }
        
        # Aggregation patterns
        self.aggregation_patterns = {
            'average': [r'\b(?:average|avg|mean)\b'],
            'minimum': [r'\b(?:minimum|min|lowest)\b'],
            'maximum': [r'\b(?:maximum|max|highest|peak)\b'], 
            'count': [r'\b(?:count|number|total)\b'],
            'sum': [r'\b(?:sum|total|aggregate)\b'],
            'std': [r'\b(?:standard deviation|std|stddev|deviation)\b']
        }
    
    async def parse_query(self, query: str) -> QueryStructure:
        """Parse natural language oceanographic query into structured format."""
        
        query_lower = query.lower().strip()
        structure = QueryStructure(intent=QueryIntent.DATA_RETRIEVAL)
        
        try:
            # 1. Determine query intent
            structure.intent = self._determine_intent(query_lower)
            
            # 2. Extract oceanographic parameters
            structure.parameters = self._extract_parameters(query_lower)
            
            # 3. Parse spatial constraints
            structure.spatial_constraints = self._extract_spatial_constraints(query)
            
            # 4. Parse temporal constraints
            structure.temporal_constraints = self.temporal_parser.parse_temporal_expressions(query)
            
            # 5. Extract parameter constraints
            structure.parameter_constraints = self._extract_parameter_constraints(query_lower)
            
            # 6. Determine aggregation requirements
            structure.aggregation_type = self._extract_aggregation_type(query_lower)
            
            # 7. Extract grouping and ordering requirements
            structure.groupby_fields, structure.orderby_fields = self._extract_grouping_ordering(query_lower)
            
            # 8. Extract output format requirements
            structure.output_format = self._extract_output_format(query_lower)
            
            # 9. Determine result limit
            structure.limit = self._extract_result_limit(query_lower)
            
            # 10. Calculate confidence score
            structure.confidence_score = self._calculate_parsing_confidence(structure, query)
            
            logger.debug(f"Parsed query with intent: {structure.intent.value}, confidence: {structure.confidence_score:.3f}")
            
            return structure
            
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            # Return basic structure with low confidence
            structure.confidence_score = 0.1
            return structure
    
    def _determine_intent(self, query: str) -> QueryIntent:
        """Determine the primary intent of the query."""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            intent_scores[intent] = score
        
        # Return intent with highest score, default to data retrieval
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        return QueryIntent.DATA_RETRIEVAL
    
    def _extract_parameters(self, query: str) -> List[str]:
        """Extract oceanographic parameters from query."""
        found_parameters = []
        
        for param, aliases in self.parameter_aliases.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', query, re.IGNORECASE):
                    found_parameters.append(param)
                    break
        
        # If no specific parameters mentioned, include basic ones
        if not found_parameters:
            found_parameters = ['temperature', 'salinity', 'pressure']
        
        return list(set(found_parameters))  # Remove duplicates
    
    def _extract_spatial_constraints(self, query: str) -> List[SpatialConstraint]:
        """Extract spatial constraints from query."""
        constraints = []
        
        # Look for coordinate patterns
        coord_patterns = [
            r'(\d+\.?\d*)[°]?\s*[NS]?\s*,?\s*(\d+\.?\d*)[°]?\s*[EW]?',
            r'lat(?:itude)?[:\s]+(\d+\.?\d*)\s*,?\s*lon(?:gitude)?[:\s]+(\d+\.?\d*)',
        ]
        
        for pattern in coord_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    
                    # Basic validation
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        constraints.append(SpatialConstraint(
                            constraint_type='circle',
                            center_lat=lat,
                            center_lon=lon,
                            radius_km=25.0  # Default 25km radius
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Look for named locations
        location_patterns = [
            r'\b(?:in|near|around|at)\s+([A-Za-z][A-Za-z\s]{2,30})\b',
            r'\b([A-Za-z][A-Za-z\s]{2,15})\s+(?:sea|ocean|bay|basin|region)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                location_text = match.group(1).strip()
                if len(location_text) > 2:  # Avoid very short matches
                    spatial_constraint = self.gazetteer.resolve_location(location_text)
                    if spatial_constraint:
                        constraints.append(spatial_constraint)
        
        return constraints
    
    def _extract_parameter_constraints(self, query: str) -> List[ParameterConstraint]:
        """Extract parameter-specific constraints."""
        constraints = []
        
        # Value range patterns
        range_patterns = [
            r'(\w+)\s+(?:between|from)\s+(\d+\.?\d*)\s+(?:and|to)\s+(\d+\.?\d*)',
            r'(\w+)\s+(?:>|greater than|above)\s+(\d+\.?\d*)',
            r'(\w+)\s+(?:<|less than|below)\s+(\d+\.?\d*)',
            r'(\w+)\s+(?:=|equals?|is)\s+(\d+\.?\d*)'
        ]
        
        for pattern in range_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                param_name = match.group(1).lower()
                
                # Map to standard parameter names
                standard_param = None
                for std_param, aliases in self.parameter_aliases.items():
                    if param_name in aliases:
                        standard_param = std_param
                        break
                
                if standard_param:
                    try:
                        if 'between' in match.group(0) or 'from' in match.group(0):
                            # Range constraint
                            min_val = float(match.group(2))
                            max_val = float(match.group(3))
                            constraints.append(ParameterConstraint(
                                parameter_name=standard_param,
                                constraint_type='range',
                                min_value=min_val,
                                max_value=max_val
                            ))
                        elif '>' in match.group(0) or 'greater' in match.group(0):
                            # Greater than constraint
                            min_val = float(match.group(2))
                            constraints.append(ParameterConstraint(
                                parameter_name=standard_param,
                                constraint_type='greater_than',
                                min_value=min_val
                            ))
                        elif '<' in match.group(0) or 'less' in match.group(0):
                            # Less than constraint
                            max_val = float(match.group(2))
                            constraints.append(ParameterConstraint(
                                parameter_name=standard_param,
                                constraint_type='less_than',
                                max_value=max_val
                            ))
                        elif '=' in match.group(0) or 'equals' in match.group(0):
                            # Equals constraint
                            target_val = float(match.group(2))
                            constraints.append(ParameterConstraint(
                                parameter_name=standard_param,
                                constraint_type='equals',
                                target_value=target_val
                            ))
                    except ValueError:
                        continue
        
        return constraints
    
    def _extract_aggregation_type(self, query: str) -> Optional[str]:
        """Extract aggregation requirements."""
        for agg_type, patterns in self.aggregation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return agg_type
        return None
    
    def _extract_grouping_ordering(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract GROUP BY and ORDER BY requirements."""
        groupby_fields = []
        orderby_fields = []
        
        # GROUP BY patterns
        groupby_patterns = [
            r'(?:group|grouped)\s+by\s+(\w+)',
            r'(?:by|per)\s+(region|location|float|depth|time|year|month)'
        ]
        
        for pattern in groupby_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                field = match.group(1).lower()
                if field not in groupby_fields:
                    groupby_fields.append(field)
        
        # ORDER BY patterns
        orderby_patterns = [
            r'(?:order|sort|sorted)\s+by\s+(\w+)',
            r'(?:ascending|descending|asc|desc)\s+(\w+)',
            r'(?:latest|newest|oldest|earliest)'
        ]
        
        for pattern in orderby_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if match.lastindex and match.lastindex >= 1:
                    field = match.group(1).lower()
                    if field not in orderby_fields:
                        orderby_fields.append(field)
        
        # Default ordering for time-based queries
        if any(word in query for word in ['recent', 'latest', 'newest']):
            if 'measurement_date' not in orderby_fields:
                orderby_fields.append('measurement_date')
        
        return groupby_fields, orderby_fields
    
    def _extract_output_format(self, query: str) -> Optional[str]:
        """Extract desired output format."""
        format_patterns = {
            'csv': r'\bcsv\b',
            'json': r'\bjson\b',
            'netcdf': r'\bnetcdf\b',
            'excel': r'\bexcel\b|\bxlsx?\b',
            'parquet': r'\bparquet\b'
        }
        
        for format_name, pattern in format_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return format_name
        
        return None
    
    def _extract_result_limit(self, query: str) -> Optional[int]:
        """Extract result limit from query."""
        limit_patterns = [
            r'(?:top|first|limit)\s+(\d+)',
            r'(\d+)\s+(?:results|records|entries|floats|profiles)'
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    limit = int(match.group(1))
                    return min(limit, 10000)  # Cap at 10k for safety
                except ValueError:
                    continue
        
        return None
    
    def _calculate_parsing_confidence(self, structure: QueryStructure, original_query: str) -> float:
        """Calculate confidence score for parsing results."""
        base_confidence = 0.5
        
        # Boost confidence based on successful extractions
        if structure.parameters:
            base_confidence += 0.15
        
        if structure.spatial_constraints:
            base_confidence += 0.15
        
        if structure.temporal_constraints:
            base_confidence += 0.15
        
        if structure.parameter_constraints:
            base_confidence += 0.1
        
        # Boost for clear intent recognition
        intent_keywords = sum(1 for patterns in self.intent_patterns[structure.intent] 
                             for pattern in patterns 
                             if re.search(pattern, original_query.lower()))
        if intent_keywords > 0:
            base_confidence += min(0.2, intent_keywords * 0.1)
        
        # Reduce confidence for very short or ambiguous queries
        if len(original_query.split()) < 3:
            base_confidence -= 0.2
        
        if not structure.parameters and not structure.spatial_constraints and not structure.temporal_constraints:
            base_confidence -= 0.3
        
        return max(0.0, min(1.0, base_confidence))