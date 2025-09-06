"""
High-performance NetCDF processing engine for ARGO oceanographic data.

This module provides efficient parsing and validation of ARGO NetCDF files
with optimized database ingestion for conversational AI applications.
"""

import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
import numpy as np
import netCDF4 as nc
from dataclasses import dataclass, field

from floatchat.core.config import settings
from floatchat.core.logging import get_logger
from floatchat.infrastructure.database.connection import get_session
from floatchat.infrastructure.repositories.argo_repository import ArgoRepository
from floatchat.domain.entities.argo_entities import DAC, Float, Profile, Measurement
from floatchat.data.processors.database_integration import DatabaseIngestionService

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for NetCDF processing operation."""
    files_processed: int = 0
    files_failed: int = 0
    floats_created: int = 0
    profiles_created: int = 0
    measurements_created: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        total = self.files_processed + self.files_failed
        return (self.files_processed / total * 100) if total > 0 else 0.0


@dataclass
class ArgoFileInfo:
    """Parsed information from ARGO NetCDF file."""
    platform_number: str
    file_type: str  # 'prof', 'meta', 'traj', 'tech'
    file_path: Path
    dac_name: str
    file_size_bytes: int
    last_modified: datetime


@dataclass
class ArgoProfileData:
    """Extracted profile data from NetCDF."""
    platform_number: str
    cycle_number: int
    latitude: float
    longitude: float
    measurement_date: datetime
    julian_day: float
    direction: str
    data_mode: str
    
    # Quality control flags
    position_qc: str
    profile_temp_qc: str
    profile_psal_qc: str
    profile_pres_qc: str
    
    # Measurements (depth-indexed arrays)
    depths: np.ndarray
    pressures: np.ndarray
    temperatures: np.ndarray
    salinities: np.ndarray
    
    # Quality flags for measurements
    pressure_qc: np.ndarray
    temperature_qc: np.ndarray
    salinity_qc: np.ndarray
    
    # Adjusted values (if available)
    temperatures_adjusted: Optional[np.ndarray] = None
    salinities_adjusted: Optional[np.ndarray] = None
    pressures_adjusted: Optional[np.ndarray] = None
    
    # Error estimates
    temperature_errors: Optional[np.ndarray] = None
    salinity_errors: Optional[np.ndarray] = None
    pressure_errors: Optional[np.ndarray] = None


@dataclass  
class ArgoFloatMetadata:
    """Extracted float metadata from NetCDF."""
    platform_number: str
    float_serial_no: Optional[str]
    wmo_inst_type: Optional[str]
    platform_type: Optional[str]
    platform_maker: Optional[str]
    firmware_version: Optional[str]
    project_name: Optional[str]
    pi_name: Optional[str]
    data_centre: Optional[str]
    deployment_date: Optional[date]
    deployment_latitude: Optional[float]
    deployment_longitude: Optional[float]


class NetCDFProcessor:
    """High-performance ARGO NetCDF processor with async capabilities."""
    
    def __init__(self, max_concurrent: int = None):
        self.max_concurrent = max_concurrent or settings.data.max_concurrent_files
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    async def process_directory(
        self, 
        directory_path: Path,
        pattern: str = "*.nc",
        recursive: bool = True
    ) -> ProcessingStats:
        """
        Process all NetCDF files in directory with concurrent processing.
        
        Args:
            directory_path: Directory containing NetCDF files
            pattern: File pattern to match (default: *.nc)
            recursive: Process subdirectories recursively
            
        Returns:
            Processing statistics
        """
        start_time = datetime.utcnow()
        stats = ProcessingStats()
        
        self.logger.info(f"Starting NetCDF processing in {directory_path}")
        
        try:
            # Find all NetCDF files
            if recursive:
                files = list(directory_path.rglob(pattern))
            else:
                files = list(directory_path.glob(pattern))
            
            self.logger.info(f"Found {len(files)} NetCDF files to process")
            
            if not files:
                return stats
            
            # Process files concurrently
            tasks = [self._process_file_with_semaphore(file_path, stats) for file_path in files]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate total processing time
            stats.processing_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.info(
                f"Processing completed: {stats.files_processed} files processed, "
                f"{stats.files_failed} failed, {stats.success_rate:.1f}% success rate"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {e}")
            stats.errors.append(f"Directory processing error: {str(e)}")
            stats.processing_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            return stats
    
    async def _process_file_with_semaphore(self, file_path: Path, stats: ProcessingStats) -> None:
        """Process single file with semaphore control."""
        async with self.semaphore:
            await self._process_single_file(file_path, stats)
    
    async def _process_single_file(self, file_path: Path, stats: ProcessingStats) -> None:
        """Process a single NetCDF file."""
        try:
            self.logger.debug(f"Processing file: {file_path}")
            
            # Parse file info
            file_info = self._parse_file_info(file_path)
            
            # Process based on file type
            if file_info.file_type == 'prof':
                await self._process_profile_file(file_path, file_info, stats)
            elif file_info.file_type == 'meta':
                await self._process_metadata_file(file_path, file_info, stats)
            else:
                # Skip trajectory and technical files for now
                self.logger.debug(f"Skipping {file_info.file_type} file: {file_path}")
                return
            
            stats.files_processed += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            stats.files_failed += 1
            stats.errors.append(f"{file_path}: {str(e)}")
    
    def _parse_file_info(self, file_path: Path) -> ArgoFileInfo:
        """Extract file information from path and metadata."""
        filename = file_path.stem
        
        # Parse ARGO filename convention: PLATFORM_NUMBER_TYPE.nc
        if '_' in filename:
            parts = filename.split('_')
            platform_number = parts[0].replace('D', '')  # Remove 'D' prefix from individual profiles
            file_type = parts[-1].lower()  # prof, meta, traj, tech, or number for individual profiles
            
            # Handle individual profile files (D2900226_001.nc)
            if file_type.isdigit():
                file_type = 'prof'
                platform_number = parts[0][1:]  # Remove 'D' prefix
        else:
            # Fallback parsing
            platform_number = filename[:7] if len(filename) >= 7 else filename
            file_type = 'prof'  # Default assumption
        
        # Extract DAC from path
        dac_name = 'unknown'
        path_parts = file_path.parts
        if 'dac' in path_parts:
            dac_index = path_parts.index('dac')
            if dac_index + 1 < len(path_parts):
                dac_name = path_parts[dac_index + 1]
        
        return ArgoFileInfo(
            platform_number=platform_number,
            file_type=file_type,
            file_path=file_path,
            dac_name=dac_name,
            file_size_bytes=file_path.stat().st_size,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
    
    async def _process_profile_file(
        self, 
        file_path: Path, 
        file_info: ArgoFileInfo, 
        stats: ProcessingStats
    ) -> None:
        """Process ARGO profile NetCDF file."""
        try:
            # Extract profile data
            profiles_data = self._extract_profile_data(file_path)
            
            if not profiles_data:
                self.logger.warning(f"No valid profiles found in {file_path}")
                return
            
            async with get_session() as session:
                db_service = DatabaseIngestionService(session)
                
                # Ensure float exists
                float_id = await db_service.ensure_float_exists(
                    file_info.platform_number, file_info.dac_name, None, stats
                )
                
                # Process all profiles in bulk
                await db_service.bulk_insert_profiles(profiles_data, float_id, stats)
                
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Profile file processing failed for {file_path}: {e}")
            raise
    
    async def _process_metadata_file(
        self, 
        file_path: Path, 
        file_info: ArgoFileInfo, 
        stats: ProcessingStats
    ) -> None:
        """Process ARGO metadata NetCDF file."""
        try:
            # Extract metadata
            metadata = self._extract_float_metadata(file_path)
            
            if not metadata:
                self.logger.warning(f"No valid metadata found in {file_path}")
                return
            
            async with get_session() as session:
                db_service = DatabaseIngestionService(session)
                
                # Ensure float exists with metadata
                await db_service.ensure_float_exists(
                    metadata.platform_number, file_info.dac_name, metadata, stats
                )
                
                # Update with additional metadata
                await db_service.update_float_metadata(metadata.platform_number, metadata)
                
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Metadata file processing failed for {file_path}: {e}")
            raise
    
    def _extract_profile_data(self, file_path: Path) -> List[ArgoProfileData]:
        """Extract profile data from NetCDF file."""
        profiles = []
        
        try:
            with nc.Dataset(file_path, 'r') as ds:
                n_prof = ds.dimensions['N_PROF'].size
                n_levels = ds.dimensions['N_LEVELS'].size
                
                for prof_idx in range(n_prof):
                    try:
                        # Extract basic profile information
                        platform_number = self._extract_string_var(ds, 'PLATFORM_NUMBER', prof_idx)
                        if not platform_number:
                            continue
                        
                        # Spatial-temporal coordinates
                        latitude = float(ds.variables['LATITUDE'][prof_idx])
                        longitude = float(ds.variables['LONGITUDE'][prof_idx])
                        julian_day = float(ds.variables['JULD'][prof_idx])
                        
                        # Skip invalid coordinates
                        if abs(latitude) > 90 or abs(longitude) > 180:
                            continue
                        
                        # Convert Julian day to datetime
                        measurement_date = self._julian_to_datetime(julian_day)
                        
                        # Profile metadata
                        cycle_number = int(ds.variables['CYCLE_NUMBER'][prof_idx])
                        direction = self._extract_string_var(ds, 'DIRECTION', prof_idx) or 'A'
                        data_mode = self._extract_string_var(ds, 'DATA_MODE', prof_idx) or 'R'
                        
                        # Quality control flags
                        position_qc = self._extract_qc_flag(ds, 'POSITION_QC', prof_idx)
                        profile_temp_qc = self._extract_qc_flag(ds, 'PROFILE_TEMP_QC', prof_idx)
                        profile_psal_qc = self._extract_qc_flag(ds, 'PROFILE_PSAL_QC', prof_idx)
                        profile_pres_qc = self._extract_qc_flag(ds, 'PROFILE_PRES_QC', prof_idx)
                        
                        # Extract measurement arrays
                        pressures = self._extract_measurement_array(ds, 'PRES', prof_idx, n_levels)
                        temperatures = self._extract_measurement_array(ds, 'TEMP', prof_idx, n_levels)
                        salinities = self._extract_measurement_array(ds, 'PSAL', prof_idx, n_levels)
                        
                        # Quality flags
                        pressure_qc = self._extract_qc_array(ds, 'PRES_QC', prof_idx, n_levels)
                        temperature_qc = self._extract_qc_array(ds, 'TEMP_QC', prof_idx, n_levels)
                        salinity_qc = self._extract_qc_array(ds, 'PSAL_QC', prof_idx, n_levels)
                        
                        # Calculate approximate depths from pressure
                        depths = self._pressure_to_depth(pressures, latitude)
                        
                        # Extract adjusted values if available
                        temperatures_adjusted = self._extract_measurement_array(
                            ds, 'TEMP_ADJUSTED', prof_idx, n_levels, optional=True
                        )
                        salinities_adjusted = self._extract_measurement_array(
                            ds, 'PSAL_ADJUSTED', prof_idx, n_levels, optional=True
                        )
                        pressures_adjusted = self._extract_measurement_array(
                            ds, 'PRES_ADJUSTED', prof_idx, n_levels, optional=True
                        )
                        
                        # Extract error estimates if available
                        temperature_errors = self._extract_measurement_array(
                            ds, 'TEMP_ADJUSTED_ERROR', prof_idx, n_levels, optional=True
                        )
                        salinity_errors = self._extract_measurement_array(
                            ds, 'PSAL_ADJUSTED_ERROR', prof_idx, n_levels, optional=True
                        )
                        pressure_errors = self._extract_measurement_array(
                            ds, 'PRES_ADJUSTED_ERROR', prof_idx, n_levels, optional=True
                        )
                        
                        profile = ArgoProfileData(
                            platform_number=platform_number,
                            cycle_number=cycle_number,
                            latitude=latitude,
                            longitude=longitude,
                            measurement_date=measurement_date,
                            julian_day=julian_day,
                            direction=direction,
                            data_mode=data_mode,
                            position_qc=position_qc,
                            profile_temp_qc=profile_temp_qc,
                            profile_psal_qc=profile_psal_qc,
                            profile_pres_qc=profile_pres_qc,
                            depths=depths,
                            pressures=pressures,
                            temperatures=temperatures,
                            salinities=salinities,
                            pressure_qc=pressure_qc,
                            temperature_qc=temperature_qc,
                            salinity_qc=salinity_qc,
                            temperatures_adjusted=temperatures_adjusted,
                            salinities_adjusted=salinities_adjusted,
                            pressures_adjusted=pressures_adjusted,
                            temperature_errors=temperature_errors,
                            salinity_errors=salinity_errors,
                            pressure_errors=pressure_errors
                        )
                        
                        # Validate profile before adding
                        if self._validate_profile_data(profile):
                            profiles.append(profile)
                        else:
                            self.logger.warning(f"Skipping invalid profile {prof_idx} from {file_path}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to extract profile {prof_idx} from {file_path}: {e}")
                        continue
                
        except Exception as e:
            self.logger.error(f"Failed to read NetCDF file {file_path}: {e}")
            raise
        
        return profiles
    
    def _extract_float_metadata(self, file_path: Path) -> Optional[ArgoFloatMetadata]:
        """Extract float metadata from NetCDF file."""
        try:
            with nc.Dataset(file_path, 'r') as ds:
                platform_number = self._extract_string_var(ds, 'PLATFORM_NUMBER', 0)
                if not platform_number:
                    return None
                
                metadata = ArgoFloatMetadata(
                    platform_number=platform_number,
                    float_serial_no=self._extract_string_var(ds, 'FLOAT_SERIAL_NO', 0),
                    wmo_inst_type=self._extract_string_var(ds, 'WMO_INST_TYPE', 0),
                    platform_type=self._extract_string_var(ds, 'PLATFORM_TYPE', 0),
                    platform_maker=self._extract_string_var(ds, 'PLATFORM_MAKER', 0),
                    firmware_version=self._extract_string_var(ds, 'FIRMWARE_VERSION', 0),
                    project_name=self._extract_string_var(ds, 'PROJECT_NAME', 0),
                    pi_name=self._extract_string_var(ds, 'PI_NAME', 0),
                    data_centre=self._extract_string_var(ds, 'DATA_CENTRE', 0)
                )
                
                # Extract deployment information
                if 'LAUNCH_DATE' in ds.variables:
                    launch_date_str = self._extract_string_var(ds, 'LAUNCH_DATE', 0)
                    if launch_date_str and len(launch_date_str) >= 8:
                        try:
                            metadata.deployment_date = datetime.strptime(
                                launch_date_str[:8], '%Y%m%d'
                            ).date()
                        except ValueError:
                            pass
                
                if 'LAUNCH_LATITUDE' in ds.variables:
                    lat = ds.variables['LAUNCH_LATITUDE'][0]
                    if not np.ma.is_masked(lat) and abs(lat) <= 90:
                        metadata.deployment_latitude = float(lat)
                
                if 'LAUNCH_LONGITUDE' in ds.variables:
                    lon = ds.variables['LAUNCH_LONGITUDE'][0]
                    if not np.ma.is_masked(lon) and abs(lon) <= 180:
                        metadata.deployment_longitude = float(lon)
                
                return metadata
                
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {file_path}: {e}")
            raise
    
    # Helper methods for data extraction
    def _extract_string_var(self, ds: nc.Dataset, var_name: str, index: int) -> Optional[str]:
        """Extract string variable from NetCDF dataset."""
        if var_name not in ds.variables:
            return None
        
        try:
            var_data = ds.variables[var_name]
            if len(var_data.shape) == 1:
                # 1D string array
                value = var_data[:]
            else:
                # 2D string array (profile index, character index)
                value = var_data[index, :]
            
            # Convert to string and clean
            if hasattr(value, 'tostring'):
                result = value.tostring().decode('utf-8', errors='ignore').strip()
            else:
                result = ''.join(char.decode('utf-8', errors='ignore') for char in value).strip()
            
            return result if result else None
            
        except (IndexError, KeyError, UnicodeDecodeError):
            return None
    
    def _extract_qc_flag(self, ds: nc.Dataset, var_name: str, index: int) -> str:
        """Extract quality control flag."""
        try:
            if var_name in ds.variables:
                qc_value = ds.variables[var_name][index]
                if hasattr(qc_value, 'decode'):
                    return qc_value.decode('utf-8', errors='ignore').strip()
                else:
                    return str(qc_value).strip()
        except:
            pass
        return '9'  # Default to missing/unknown
    
    def _extract_measurement_array(
        self, 
        ds: nc.Dataset, 
        var_name: str, 
        prof_idx: int, 
        n_levels: int,
        optional: bool = False
    ) -> Optional[np.ndarray]:
        """Extract measurement array from NetCDF."""
        if var_name not in ds.variables:
            return None if optional else np.full(n_levels, np.nan)
        
        try:
            data = ds.variables[var_name][prof_idx, :]
            # Handle masked arrays
            if np.ma.is_masked(data):
                data = np.ma.filled(data, np.nan)
            return data.astype(np.float32)
        except:
            return None if optional else np.full(n_levels, np.nan)
    
    def _extract_qc_array(self, ds: nc.Dataset, var_name: str, prof_idx: int, n_levels: int) -> np.ndarray:
        """Extract quality control flag array."""
        if var_name not in ds.variables:
            return np.full(n_levels, '9', dtype='U1')
        
        try:
            qc_data = ds.variables[var_name][prof_idx, :]
            # Convert to string array
            qc_strings = []
            for qc_val in qc_data:
                if hasattr(qc_val, 'decode'):
                    qc_strings.append(qc_val.decode('utf-8', errors='ignore').strip() or '9')
                else:
                    qc_strings.append(str(qc_val).strip() or '9')
            return np.array(qc_strings, dtype='U1')
        except:
            return np.full(n_levels, '9', dtype='U1')
    
    def _julian_to_datetime(self, julian_day: float) -> datetime:
        """Convert Julian day (since 1950-01-01) to datetime."""
        reference_date = datetime(1950, 1, 1)
        return reference_date + np.timedelta64(int(julian_day), 'D')
    
    def _pressure_to_depth(self, pressure: np.ndarray, latitude: float) -> np.ndarray:
        """
        Convert pressure to approximate depth using UNESCO formula.
        
        Args:
            pressure: Pressure in decibars
            latitude: Latitude in degrees
            
        Returns:
            Depth in meters
        """
        # Simplified UNESCO formula for depth calculation
        # More accurate formula would consider water density variations
        g = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * latitude**2) * np.sin(np.radians(latitude))**2)
        depth = pressure * 1.019716 / g * 10000
        return depth.astype(np.float32)
    
    # Utility methods for data quality and validation
    def _validate_profile_data(self, profile_data: ArgoProfileData) -> bool:
        """Validate profile data quality and completeness."""
        # Basic validation checks
        if not profile_data.platform_number or len(profile_data.platform_number) < 5:
            return False
        
        # Check coordinate validity
        if abs(profile_data.latitude) > 90 or abs(profile_data.longitude) > 180:
            return False
        
        # Check if we have any valid measurements
        valid_measurements = 0
        for i in range(len(profile_data.depths)):
            if (not np.isnan(profile_data.pressures[i]) or 
                not np.isnan(profile_data.temperatures[i]) or 
                not np.isnan(profile_data.salinities[i])):
                valid_measurements += 1
        
        return valid_measurements > 0
    
    def _get_processing_summary(self, stats: ProcessingStats) -> Dict[str, Any]:
        """Get formatted processing summary for logging."""
        return {
            'files_processed': stats.files_processed,
            'files_failed': stats.files_failed,
            'success_rate': f"{stats.success_rate:.1f}%",
            'floats_created': stats.floats_created,
            'profiles_created': stats.profiles_created,
            'measurements_created': stats.measurements_created,
            'processing_time': f"{stats.processing_time_seconds:.2f}s",
            'error_count': len(stats.errors)
        }