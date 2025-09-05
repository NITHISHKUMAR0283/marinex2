"""
Repository pattern for ARGO oceanographic data access.

This module provides high-level data access methods optimized for LLM queries
and conversational AI applications.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from geoalchemy2 import functions as geo_func
from sqlalchemy import and_, or_, func, select, text, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from floatchat.domain.entities.argo_entities import (
    DAC, Float, Profile, Measurement, ProfileStatistic, 
    OceanRegion, MonthlyClimatology
)
from floatchat.core.logging import get_logger

logger = get_logger(__name__)


class ArgoRepository:
    """Repository for ARGO oceanographic data with LLM-optimized queries."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # ========================================
    # Float Management
    # ========================================
    
    async def get_float_by_platform_number(self, platform_number: str) -> Optional[Float]:
        """Get float by platform number."""
        query = select(Float).where(Float.platform_number == platform_number)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def create_float(self, float_data: dict) -> Float:
        """Create new ARGO float."""
        new_float = Float(**float_data)
        self.session.add(new_float)
        await self.session.flush()
        return new_float
    
    async def get_active_floats(self, limit: int = 100) -> List[Float]:
        """Get active floats with recent data."""
        query = (
            select(Float)
            .options(joinedload(Float.dac))
            .where(Float.is_active == True)
            .order_by(desc(Float.last_profile_date))
            .limit(limit)
        )
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_floats_in_region(
        self, 
        min_lat: float, 
        max_lat: float,
        min_lon: float, 
        max_lon: float
    ) -> List[Float]:
        """Get floats deployed in geographic region."""
        query = (
            select(Float)
            .where(
                and_(
                    Float.deployment_latitude.between(min_lat, max_lat),
                    Float.deployment_longitude.between(min_lon, max_lon)
                )
            )
            .options(joinedload(Float.dac))
        )
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # ========================================
    # Profile Queries for LLM
    # ========================================
    
    async def search_profiles_by_natural_language(
        self,
        region_name: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        temperature_range: Optional[Tuple[float, float]] = None,
        salinity_range: Optional[Tuple[float, float]] = None,
        data_quality: str = "good",  # "good", "any", "excellent"
        limit: int = 1000
    ) -> List[Profile]:
        """
        Search profiles using natural language parameters.
        Optimized for LLM query patterns.
        """
        query = (
            select(Profile)
            .options(
                joinedload(Profile.float).joinedload(Float.dac),
                selectinload(Profile.statistics)
            )
        )
        
        conditions = []
        
        # Geographic filtering
        if region_name:
            region_conditions = await self._build_region_conditions(region_name)
            if region_conditions:
                conditions.extend(region_conditions)
        
        # Temporal filtering
        if start_date:
            conditions.append(Profile.measurement_date >= start_date)
        if end_date:
            conditions.append(Profile.measurement_date <= end_date)
        
        # Depth filtering
        if min_depth is not None:
            conditions.append(Profile.max_depth_m >= min_depth)
        if max_depth is not None:
            conditions.append(Profile.max_depth_m <= max_depth)
        
        # Temperature filtering
        if temperature_range:
            min_temp, max_temp = temperature_range
            conditions.append(
                and_(
                    Profile.min_temperature_c <= max_temp,
                    Profile.max_temperature_c >= min_temp
                )
            )
        
        # Salinity filtering
        if salinity_range:
            min_sal, max_sal = salinity_range
            conditions.append(
                and_(
                    Profile.min_salinity_psu <= max_sal,
                    Profile.max_salinity_psu >= min_sal
                )
            )
        
        # Quality filtering
        if data_quality == "excellent":
            conditions.extend([
                Profile.profile_temp_qc == '1',
                Profile.profile_psal_qc == '1',
                Profile.data_mode == 'D'  # Delayed mode
            ])
        elif data_quality == "good":
            conditions.extend([
                Profile.profile_temp_qc.in_(['1', '2']),
                Profile.profile_psal_qc.in_(['1', '2'])
            ])
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(Profile.measurement_date)).limit(limit)
        
        result = await self.session.execute(query)
        profiles = result.scalars().all()
        
        logger.info(f"Found {len(profiles)} profiles matching criteria")
        return profiles
    
    async def get_temperature_anomalies(
        self,
        region_name: str,
        reference_period: Tuple[int, int],  # (start_year, end_year)
        analysis_period: Tuple[int, int],
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Find temperature anomalies for LLM analysis.
        
        Args:
            region_name: Geographic region name
            reference_period: Reference years for climatology
            analysis_period: Years to analyze for anomalies  
            threshold_std: Standard deviation threshold
        """
        # This would implement complex climatology analysis
        # For now, return a simplified version
        
        query = (
            select(
                Profile,
                func.avg(Profile.min_temperature_c).over().label('avg_temp'),
                func.stddev(Profile.min_temperature_c).over().label('std_temp')
            )
            .where(
                Profile.measurement_year.between(
                    analysis_period[0], analysis_period[1]
                )
            )
        )
        
        # Add region conditions
        region_conditions = await self._build_region_conditions(region_name)
        if region_conditions:
            query = query.where(and_(*region_conditions))
        
        result = await self.session.execute(query)
        rows = result.all()
        
        anomalies = []
        for row in rows:
            profile = row[0]
            avg_temp = row[1]
            std_temp = row[2]
            
            if profile.min_temperature_c and avg_temp and std_temp:
                anomaly = (profile.min_temperature_c - avg_temp) / std_temp
                if abs(anomaly) > threshold_std:
                    anomalies.append({
                        'profile_id': profile.id,
                        'date': profile.measurement_date,
                        'location': {'lat': profile.latitude, 'lon': profile.longitude},
                        'temperature': profile.min_temperature_c,
                        'anomaly_score': round(anomaly, 2),
                        'platform_number': profile.float.platform_number
                    })
        
        return sorted(anomalies, key=lambda x: abs(x['anomaly_score']), reverse=True)
    
    async def get_seasonal_patterns(
        self,
        region_name: str,
        years: Optional[List[int]] = None,
        parameter: str = "temperature"
    ) -> Dict[int, List[float]]:
        """Get seasonal patterns for time series analysis."""
        
        query_field = {
            "temperature": Profile.min_temperature_c,
            "salinity": Profile.min_salinity_psu
        }.get(parameter, Profile.min_temperature_c)
        
        query = (
            select(
                Profile.measurement_season,
                func.avg(query_field).label('avg_value'),
                func.count().label('count')
            )
            .where(Profile.data_mode == 'D')  # Use delayed-mode data
            .group_by(Profile.measurement_season)
        )
        
        # Add year filtering
        if years:
            query = query.where(Profile.measurement_year.in_(years))
        
        # Add region filtering
        region_conditions = await self._build_region_conditions(region_name)
        if region_conditions:
            query = query.where(and_(*region_conditions))
        
        result = await self.session.execute(query)
        rows = result.all()
        
        seasonal_data = {}
        for row in rows:
            season = row[0]
            avg_value = float(row[1]) if row[1] else 0.0
            seasonal_data[season] = avg_value
        
        return seasonal_data
    
    # ========================================
    # Measurement Data
    # ========================================
    
    async def get_depth_profile_data(
        self, 
        profile_id: int,
        parameter: str = "temperature"
    ) -> List[Dict[str, Any]]:
        """Get depth profile data for visualization."""
        
        param_field = {
            "temperature": Measurement.temperature_c,
            "salinity": Measurement.salinity_psu,
            "pressure": Measurement.pressure_db
        }.get(parameter, Measurement.temperature_c)
        
        query = (
            select(Measurement.depth_m, param_field)
            .where(
                and_(
                    Measurement.profile_id == profile_id,
                    Measurement.is_valid == True
                )
            )
            .order_by(Measurement.depth_m)
        )
        
        result = await self.session.execute(query)
        rows = result.all()
        
        return [
            {
                'depth_m': float(row[0]) if row[0] else 0.0,
                'value': float(row[1]) if row[1] else None
            }
            for row in rows if row[1] is not None
        ]
    
    async def bulk_insert_measurements(
        self, 
        measurements_data: List[Dict[str, Any]]
    ) -> int:
        """Bulk insert measurements for efficient NetCDF processing."""
        measurements = [Measurement(**data) for data in measurements_data]
        self.session.add_all(measurements)
        await self.session.flush()
        return len(measurements)
    
    # ========================================
    # Statistical Analysis
    # ========================================
    
    async def get_water_mass_analysis(
        self,
        region_name: str,
        depth_range: Tuple[float, float],
        date_range: Tuple[date, date]
    ) -> Dict[str, Any]:
        """Analyze water mass characteristics for LLM interpretation."""
        
        query = (
            select(
                func.avg(Measurement.temperature_c).label('avg_temp'),
                func.avg(Measurement.salinity_psu).label('avg_sal'),
                func.stddev(Measurement.temperature_c).label('std_temp'),
                func.stddev(Measurement.salinity_psu).label('std_sal'),
                func.count().label('sample_count')
            )
            .select_from(
                Measurement
                .join(Profile, Measurement.profile_id == Profile.id)
            )
            .where(
                and_(
                    Measurement.depth_m.between(depth_range[0], depth_range[1]),
                    Profile.measurement_date.between(date_range[0], date_range[1]),
                    Measurement.is_valid == True
                )
            )
        )
        
        # Add region filtering
        region_conditions = await self._build_region_conditions(region_name)
        if region_conditions:
            query = query.where(and_(*region_conditions))
        
        result = await self.session.execute(query)
        row = result.first()
        
        if not row or not row[4]:  # No samples
            return {}
        
        return {
            'region': region_name,
            'depth_range': f"{depth_range[0]}-{depth_range[1]}m",
            'date_range': f"{date_range[0]} to {date_range[1]}",
            'temperature': {
                'mean': round(float(row[0]), 2) if row[0] else None,
                'std': round(float(row[2]), 2) if row[2] else None
            },
            'salinity': {
                'mean': round(float(row[1]), 2) if row[1] else None,
                'std': round(float(row[3]), 2) if row[3] else None
            },
            'sample_count': int(row[4])
        }
    
    # ========================================
    # Helper Methods
    # ========================================
    
    async def _build_region_conditions(self, region_name: str) -> List[Any]:
        """Build spatial conditions for region queries."""
        conditions = []
        
        # Predefined region coordinates (simplified)
        region_bounds = {
            'arabian sea': (10, 25, 50, 80),  # (min_lat, max_lat, min_lon, max_lon)
            'bay of bengal': (5, 22, 80, 100),
            'indian ocean': (-60, 30, 20, 120),
            'north atlantic': (30, 70, -80, 0),
            'south atlantic': (-60, 30, -70, 20),
            'north pacific': (30, 70, 120, -100),
            'south pacific': (-60, 30, 120, -70),
            'mediterranean': (30, 46, -6, 42),
            'southern ocean': (-70, -50, -180, 180)
        }
        
        region_key = region_name.lower().strip()
        if region_key in region_bounds:
            min_lat, max_lat, min_lon, max_lon = region_bounds[region_key]
            conditions.extend([
                Profile.latitude.between(min_lat, max_lat),
                Profile.longitude.between(min_lon, max_lon)
            ])
        
        return conditions
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics for monitoring and LLM context."""
        
        stats_queries = {
            'total_floats': select(func.count(Float.id)),
            'active_floats': select(func.count(Float.id)).where(Float.is_active == True),
            'total_profiles': select(func.count(Profile.id)),
            'total_measurements': select(func.count(Measurement.id)),
            'latest_profile_date': select(func.max(Profile.measurement_date)),
            'earliest_profile_date': select(func.min(Profile.measurement_date))
        }
        
        results = {}
        for key, query in stats_queries.items():
            result = await self.session.execute(query)
            results[key] = result.scalar()
        
        # Geographic coverage
        coverage_query = select(
            func.min(Profile.latitude).label('min_lat'),
            func.max(Profile.latitude).label('max_lat'),
            func.min(Profile.longitude).label('min_lon'),
            func.max(Profile.longitude).label('max_lon')
        )
        coverage_result = await self.session.execute(coverage_query)
        coverage = coverage_result.first()
        
        results['geographic_coverage'] = {
            'latitude_range': [float(coverage[0]), float(coverage[1])],
            'longitude_range': [float(coverage[2]), float(coverage[3])]
        } if all(coverage) else None
        
        return results
    
    async def search_similar_profiles(
        self,
        reference_profile_id: int,
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find profiles similar to reference profile."""
        
        # Get reference profile statistics
        ref_query = (
            select(ProfileStatistic)
            .where(ProfileStatistic.profile_id == reference_profile_id)
        )
        ref_result = await self.session.execute(ref_query)
        ref_stats = ref_result.scalar_one_or_none()
        
        if not ref_stats:
            return []
        
        # Find similar profiles using statistical comparison
        similarity_query = (
            select(
                ProfileStatistic.profile_id,
                Profile.latitude,
                Profile.longitude,
                Profile.measurement_date,
                Float.platform_number,
                # Simple similarity score based on temperature and salinity
                (
                    1.0 / (1.0 + 
                        func.abs(ProfileStatistic.temp_mean - ref_stats.temp_mean) +
                        func.abs(ProfileStatistic.sal_mean - ref_stats.sal_mean) +
                        func.abs(ProfileStatistic.max_depth - ref_stats.max_depth) / 1000.0
                    )
                ).label('similarity_score')
            )
            .select_from(
                ProfileStatistic
                .join(Profile, ProfileStatistic.profile_id == Profile.id)
                .join(Float, Profile.float_id == Float.id)
            )
            .where(ProfileStatistic.profile_id != reference_profile_id)
            .having(text('similarity_score > :threshold'))
            .params(threshold=similarity_threshold)
            .order_by(desc(text('similarity_score')))
            .limit(limit)
        )
        
        result = await self.session.execute(similarity_query)
        rows = result.all()
        
        return [
            {
                'profile_id': row[0],
                'location': {'lat': float(row[1]), 'lon': float(row[2])},
                'date': row[3],
                'platform_number': row[4],
                'similarity_score': round(float(row[5]), 3)
            }
            for row in rows
        ]