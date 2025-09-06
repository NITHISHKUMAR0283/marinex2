"""
Database integration for NetCDF processing.

This module provides efficient database insertion methods for ARGO data
with bulk operations and transaction management.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from floatchat.core.logging import get_logger
from floatchat.infrastructure.repositories.argo_repository import ArgoRepository
from floatchat.domain.entities.argo_entities import (
    DAC, Float, Profile, Measurement, ProfileStatistic
)
from floatchat.data.processors.netcdf_processor import (
    ArgoProfileData, ArgoFloatMetadata, ProcessingStats
)

logger = get_logger(__name__)


class DatabaseIngestionService:
    """Service for efficient database ingestion of ARGO data."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = ArgoRepository(session)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache for DACs and floats to avoid repeated lookups
        self._dac_cache: Dict[str, int] = {}
        self._float_cache: Dict[str, int] = {}
    
    async def ensure_dac_exists(self, dac_name: str) -> int:
        """Ensure DAC exists in database and return ID."""
        if dac_name in self._dac_cache:
            return self._dac_cache[dac_name]
        
        # Try to find existing DAC
        dac_query = select(DAC).where(DAC.code == dac_name)
        result = await self.session.execute(dac_query)
        dac = result.scalar_one_or_none()
        
        if not dac:
            # Create new DAC
            dac_data = {
                'code': dac_name,
                'name': dac_name.upper() + ' Data Assembly Center',
                'country': 'Unknown',
                'institution': f'{dac_name.upper()} Institution'
            }
            
            dac = DAC(**dac_data)
            self.session.add(dac)
            await self.session.flush()
            
            self.logger.info(f"Created new DAC: {dac_name}")
        
        self._dac_cache[dac_name] = dac.id
        return dac.id
    
    async def ensure_float_exists(
        self, 
        platform_number: str, 
        dac_name: str,
        metadata: Optional[ArgoFloatMetadata] = None,
        stats: Optional[ProcessingStats] = None
    ) -> int:
        """Ensure float exists in database and return ID."""
        cache_key = f"{dac_name}:{platform_number}"
        
        if cache_key in self._float_cache:
            return self._float_cache[cache_key]
        
        # Try to find existing float
        float_obj = await self.repo.get_float_by_platform_number(platform_number)
        
        if not float_obj:
            # Get DAC ID
            dac_id = await self.ensure_dac_exists(dac_name)
            
            # Create new float
            float_data = {
                'platform_number': platform_number,
                'dac_id': dac_id,
                'is_active': True
            }
            
            # Add metadata if available
            if metadata:
                float_data.update({
                    'float_serial_no': metadata.float_serial_no,
                    'wmo_inst_type': metadata.wmo_inst_type,
                    'platform_type': metadata.platform_type,
                    'platform_maker': metadata.platform_maker,
                    'firmware_version': metadata.firmware_version,
                    'project_name': metadata.project_name,
                    'pi_name': metadata.pi_name,
                    'data_centre': metadata.data_centre,
                    'deployment_date': metadata.deployment_date,
                    'deployment_latitude': metadata.deployment_latitude,
                    'deployment_longitude': metadata.deployment_longitude
                })
            
            float_obj = await self.repo.create_float(float_data)
            
            if stats:
                stats.floats_created += 1
            
            self.logger.info(f"Created new float: {platform_number}")
        
        self._float_cache[cache_key] = float_obj.id
        return float_obj.id
    
    async def insert_profile_with_measurements(
        self,
        profile_data: ArgoProfileData,
        float_id: int,
        stats: Optional[ProcessingStats] = None
    ) -> Optional[int]:
        """Insert profile and all its measurements in a single transaction."""
        try:
            # Check if profile already exists
            existing_profile_query = select(Profile).where(
                Profile.float_id == float_id,
                Profile.cycle_number == profile_data.cycle_number
            )
            result = await self.session.execute(existing_profile_query)
            existing_profile = result.scalar_one_or_none()
            
            if existing_profile:
                self.logger.debug(f"Profile already exists: {profile_data.platform_number} cycle {profile_data.cycle_number}")
                return existing_profile.id
            
            # Create profile
            profile = Profile(
                float_id=float_id,
                cycle_number=profile_data.cycle_number,
                direction=profile_data.direction,
                latitude=profile_data.latitude,
                longitude=profile_data.longitude,
                juld=profile_data.julian_day,
                measurement_date=profile_data.measurement_date,
                data_mode=profile_data.data_mode,
                position_qc=profile_data.position_qc,
                profile_temp_qc=profile_data.profile_temp_qc,
                profile_psal_qc=profile_data.profile_psal_qc,
                profile_pres_qc=profile_data.profile_pres_qc
            )
            
            self.session.add(profile)
            await self.session.flush()  # Get profile ID
            
            # Prepare measurements for bulk insert
            measurements_data = []
            valid_measurements = 0
            
            for level_idx in range(len(profile_data.depths)):
                # Skip invalid measurements
                if (np.isnan(profile_data.pressures[level_idx]) and 
                    np.isnan(profile_data.temperatures[level_idx]) and 
                    np.isnan(profile_data.salinities[level_idx])):
                    continue
                
                # Extract values safely
                pressure = float(profile_data.pressures[level_idx]) if not np.isnan(profile_data.pressures[level_idx]) else None
                temperature = float(profile_data.temperatures[level_idx]) if not np.isnan(profile_data.temperatures[level_idx]) else None
                salinity = float(profile_data.salinities[level_idx]) if not np.isnan(profile_data.salinities[level_idx]) else None
                depth = float(profile_data.depths[level_idx]) if not np.isnan(profile_data.depths[level_idx]) else None
                
                # Quality flags
                pressure_qc = str(profile_data.pressure_qc[level_idx]) if level_idx < len(profile_data.pressure_qc) else '9'
                temperature_qc = str(profile_data.temperature_qc[level_idx]) if level_idx < len(profile_data.temperature_qc) else '9'
                salinity_qc = str(profile_data.salinity_qc[level_idx]) if level_idx < len(profile_data.salinity_qc) else '9'
                
                # Adjusted values
                temperature_adjusted = None
                salinity_adjusted = None
                pressure_adjusted = None
                
                if profile_data.temperatures_adjusted is not None and level_idx < len(profile_data.temperatures_adjusted):
                    temp_adj = profile_data.temperatures_adjusted[level_idx]
                    temperature_adjusted = float(temp_adj) if not np.isnan(temp_adj) else None
                
                if profile_data.salinities_adjusted is not None and level_idx < len(profile_data.salinities_adjusted):
                    sal_adj = profile_data.salinities_adjusted[level_idx]
                    salinity_adjusted = float(sal_adj) if not np.isnan(sal_adj) else None
                
                if profile_data.pressures_adjusted is not None and level_idx < len(profile_data.pressures_adjusted):
                    pres_adj = profile_data.pressures_adjusted[level_idx]
                    pressure_adjusted = float(pres_adj) if not np.isnan(pres_adj) else None
                
                # Error estimates
                temperature_error = None
                salinity_error = None
                pressure_error = None
                
                if profile_data.temperature_errors is not None and level_idx < len(profile_data.temperature_errors):
                    temp_err = profile_data.temperature_errors[level_idx]
                    temperature_error = float(temp_err) if not np.isnan(temp_err) else None
                
                if profile_data.salinity_errors is not None and level_idx < len(profile_data.salinity_errors):
                    sal_err = profile_data.salinity_errors[level_idx]
                    salinity_error = float(sal_err) if not np.isnan(sal_err) else None
                
                if profile_data.pressure_errors is not None and level_idx < len(profile_data.pressure_errors):
                    pres_err = profile_data.pressure_errors[level_idx]
                    pressure_error = float(pres_err) if not np.isnan(pres_err) else None
                
                # Check if this is a valid measurement
                is_valid = (pressure_qc in ('1', '2') and 
                           temperature_qc in ('1', '2') and 
                           salinity_qc in ('1', '2'))
                
                if is_valid:
                    valid_measurements += 1
                
                measurement_data = {
                    'profile_id': profile.id,
                    'depth_level': level_idx,
                    'pressure_db': pressure,
                    'depth_m': depth,
                    'temperature_c': temperature,
                    'salinity_psu': salinity,
                    'temperature_adjusted_c': temperature_adjusted,
                    'salinity_adjusted_c': salinity_adjusted,
                    'pressure_adjusted_db': pressure_adjusted,
                    'temperature_qc': temperature_qc,
                    'salinity_qc': salinity_qc,
                    'pressure_qc': pressure_qc,
                    'temperature_error': temperature_error,
                    'salinity_error': salinity_error,
                    'pressure_error': pressure_error,
                    'is_valid': is_valid
                }
                
                measurements_data.append(measurement_data)
            
            # Bulk insert measurements if any exist
            if measurements_data:
                stmt = insert(Measurement).values(measurements_data)
                await self.session.execute(stmt)
                
                # Update profile statistics
                profile.valid_measurements_count = valid_measurements
                profile.max_depth_m = max((m['depth_m'] for m in measurements_data if m['depth_m']), default=None)
                
                # Calculate temperature and salinity ranges
                valid_temps = [m['temperature_c'] for m in measurements_data if m['temperature_c'] and m['is_valid']]
                valid_sals = [m['salinity_psu'] for m in measurements_data if m['salinity_psu'] and m['is_valid']]
                
                if valid_temps:
                    profile.min_temperature_c = min(valid_temps)
                    profile.max_temperature_c = max(valid_temps)
                
                if valid_sals:
                    profile.min_salinity_psu = min(valid_sals)
                    profile.max_salinity_psu = max(valid_sals)
                
                if stats:
                    stats.measurements_created += len(measurements_data)
            
            if stats:
                stats.profiles_created += 1
            
            self.logger.debug(f"Inserted profile {profile_data.cycle_number} with {len(measurements_data)} measurements")
            return profile.id
            
        except Exception as e:
            self.logger.error(f"Failed to insert profile {profile_data.cycle_number}: {e}")
            raise
    
    async def update_float_metadata(
        self, 
        platform_number: str, 
        metadata: ArgoFloatMetadata
    ) -> None:
        """Update float with metadata from meta.nc file."""
        try:
            float_obj = await self.repo.get_float_by_platform_number(platform_number)
            if not float_obj:
                self.logger.warning(f"Float {platform_number} not found for metadata update")
                return
            
            # Update metadata fields
            if metadata.float_serial_no:
                float_obj.float_serial_no = metadata.float_serial_no
            if metadata.wmo_inst_type:
                float_obj.wmo_inst_type = metadata.wmo_inst_type
            if metadata.platform_type:
                float_obj.platform_type = metadata.platform_type
            if metadata.platform_maker:
                float_obj.platform_maker = metadata.platform_maker
            if metadata.firmware_version:
                float_obj.firmware_version = metadata.firmware_version
            if metadata.project_name:
                float_obj.project_name = metadata.project_name
            if metadata.pi_name:
                float_obj.pi_name = metadata.pi_name
            if metadata.data_centre:
                float_obj.data_centre = metadata.data_centre
            
            # Update deployment information
            if metadata.deployment_date and not float_obj.deployment_date:
                float_obj.deployment_date = metadata.deployment_date
            if metadata.deployment_latitude and not float_obj.deployment_latitude:
                float_obj.deployment_latitude = metadata.deployment_latitude
            if metadata.deployment_longitude and not float_obj.deployment_longitude:
                float_obj.deployment_longitude = metadata.deployment_longitude
            
            float_obj.updated_at = datetime.utcnow()
            
            self.logger.debug(f"Updated metadata for float {platform_number}")
            
        except Exception as e:
            self.logger.error(f"Failed to update float metadata for {platform_number}: {e}")
            raise
    
    async def generate_profile_statistics(self, profile_id: int) -> None:
        """Generate and store profile statistics for LLM queries."""
        try:
            # Query measurements for the profile
            measurements_query = select(Measurement).where(
                Measurement.profile_id == profile_id,
                Measurement.is_valid == True
            ).order_by(Measurement.depth_m)
            
            result = await self.session.execute(measurements_query)
            measurements = result.scalars().all()
            
            if not measurements:
                return
            
            # Extract measurement arrays
            temperatures = [m.preferred_temperature for m in measurements if m.preferred_temperature]
            salinities = [m.preferred_salinity for m in measurements if m.preferred_salinity]
            depths = [m.depth_m for m in measurements if m.depth_m]
            
            if not temperatures or not salinities:
                return
            
            # Calculate statistics
            temp_array = np.array(temperatures)
            sal_array = np.array(salinities)
            depth_array = np.array(depths)
            
            # Calculate basic statistics
            stats_data = {
                'profile_id': profile_id,
                'temp_mean': float(np.mean(temp_array)),
                'temp_std': float(np.std(temp_array)),
                'temp_min': float(np.min(temp_array)),
                'temp_max': float(np.max(temp_array)),
                'sal_mean': float(np.mean(sal_array)),
                'sal_std': float(np.std(sal_array)),
                'sal_min': float(np.min(sal_array)),
                'sal_max': float(np.max(sal_array)),
                'max_depth': float(np.max(depth_array)) if len(depth_array) > 0 else None,
                'valid_depths_count': len(depths)
            }
            
            # Surface and bottom values
            if len(temperatures) > 0:
                stats_data['temp_surface'] = temperatures[0]  # Shallowest
                stats_data['temp_bottom'] = temperatures[-1]  # Deepest
            
            if len(salinities) > 0:
                stats_data['sal_surface'] = salinities[0]
                stats_data['sal_bottom'] = salinities[-1]
            
            # Simple quality score based on number of valid measurements
            total_measurements = len(measurements)
            quality_ratio = len([m for m in measurements if m.is_good_quality]) / total_measurements
            stats_data['data_quality_score'] = quality_ratio
            
            # Insert or update statistics
            existing_stats_query = select(ProfileStatistic).where(
                ProfileStatistic.profile_id == profile_id
            )
            result = await self.session.execute(existing_stats_query)
            existing_stats = result.scalar_one_or_none()
            
            if existing_stats:
                # Update existing
                for key, value in stats_data.items():
                    if key != 'profile_id':
                        setattr(existing_stats, key, value)
            else:
                # Create new
                profile_stats = ProfileStatistic(**stats_data)
                self.session.add(profile_stats)
            
            self.logger.debug(f"Generated statistics for profile {profile_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate statistics for profile {profile_id}: {e}")
            # Don't raise - statistics are optional
    
    async def bulk_insert_profiles(
        self,
        profiles_data: List[ArgoProfileData],
        float_id: int,
        stats: Optional[ProcessingStats] = None
    ) -> List[int]:
        """Bulk insert multiple profiles with their measurements."""
        profile_ids = []
        
        for profile_data in profiles_data:
            try:
                profile_id = await self.insert_profile_with_measurements(
                    profile_data, float_id, stats
                )
                if profile_id:
                    profile_ids.append(profile_id)
                    
                    # Generate statistics in background (don't wait)
                    asyncio.create_task(
                        self.generate_profile_statistics(profile_id)
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to insert profile {profile_data.cycle_number}: {e}")
                if stats:
                    stats.errors.append(
                        f"Profile {profile_data.cycle_number}: {str(e)}"
                    )
                continue
        
        return profile_ids
    
    async def optimize_after_bulk_insert(self) -> None:
        """Run optimization queries after bulk data insertion."""
        try:
            # Update materialized views if they exist
            # Refresh any cached statistics
            # This is a placeholder for future optimizations
            
            await self.session.execute(text("ANALYZE floats, profiles, measurements"))
            self.logger.debug("Ran ANALYZE on main tables")
            
        except Exception as e:
            self.logger.warning(f"Post-insert optimization failed: {e}")
            # Don't raise - optimization is optional