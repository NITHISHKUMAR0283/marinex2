"""
Domain entities for ARGO oceanographic data.

This module defines SQLAlchemy models for the ARGO float database schema
optimized for LLM queries and conversational AI applications.
"""

from datetime import datetime, date
from typing import Optional, List
from uuid import UUID

from geoalchemy2 import Geometry
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Date, 
    ForeignKey, Text, ARRAY, CheckConstraint, UniqueConstraint,
    BigInteger, REAL, CHAR, Index, func, text
)
from sqlalchemy.types import Float as FloatType
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, NUMRANGE
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import expression
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class DAC(Base):
    """Data Assembly Center - organizations that collect and distribute ARGO data."""
    
    __tablename__ = "dacs"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    country = Column(String(100))
    institution = Column(String(255))
    contact_email = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    floats = relationship("Float", back_populates="dac")
    
    def __repr__(self) -> str:
        return f"<DAC(code='{self.code}', name='{self.name}')>"


class OceanRegion(Base):
    """Ocean regions for geographic classification of profiles."""
    
    __tablename__ = "ocean_regions"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    aliases = Column(ARRAY(Text))
    boundary = Column(Geometry('POLYGON', srid=4326))
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    monthly_climatology = relationship("MonthlyClimatology", back_populates="region")
    
    def __repr__(self) -> str:
        return f"<OceanRegion(name='{self.name}')>"
        
    @hybrid_property
    def alias_list(self) -> List[str]:
        """Get aliases as a Python list."""
        return self.aliases or []


class Float(Base):
    """ARGO float with comprehensive metadata."""
    
    __tablename__ = "floats"
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), unique=True, nullable=False, index=True)
    dac_id = Column(Integer, ForeignKey("dacs.id"))
    
    # Physical characteristics
    float_serial_no = Column(String(50))
    wmo_inst_type = Column(String(10))
    platform_type = Column(String(50))
    platform_maker = Column(String(100))
    firmware_version = Column(String(50))
    
    # Deployment information
    deployment_date = Column(Date)
    deployment_latitude = Column(FloatType)
    deployment_longitude = Column(FloatType)
    deployment_geom = Column(Geometry('POINT', srid=4326))
    
    # Mission details
    project_name = Column(String(255))
    pi_name = Column(String(255))
    data_centre = Column(String(50))
    
    # Status tracking
    is_active = Column(Boolean, default=True)
    last_profile_date = Column(Date)
    total_profiles = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    dac = relationship("DAC", back_populates="floats")
    profiles = relationship("Profile", back_populates="float", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            'deployment_latitude BETWEEN -90 AND 90 AND '
            'deployment_longitude BETWEEN -180 AND 180',
            name='valid_deployment_coords'
        ),
        Index('idx_floats_deployment_geom', 'deployment_geom', postgresql_using='gist'),
        Index('idx_floats_text_search', 
              func.to_tsvector('english', 
                  func.coalesce(text('project_name'), '') + ' ' + 
                  func.coalesce(text('pi_name'), '')), 
              postgresql_using='gin'),
    )
    
    @validates('deployment_latitude')
    def validate_latitude(self, key, value):
        if value is not None and not -90 <= value <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return value
    
    @validates('deployment_longitude') 
    def validate_longitude(self, key, value):
        if value is not None and not -180 <= value <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return value
    
    def __repr__(self) -> str:
        return f"<Float(platform_number='{self.platform_number}', active={self.is_active})>"


class Profile(Base):
    """Individual ARGO profile measurement."""
    
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True)
    profile_uuid = Column(PostgresUUID(as_uuid=True), server_default=func.gen_random_uuid())
    float_id = Column(Integer, ForeignKey("floats.id", ondelete="CASCADE"), nullable=False)
    
    # Profile identification
    cycle_number = Column(Integer, nullable=False)
    direction = Column(String(1))
    profile_filename = Column(String(255))
    
    # Spatial-temporal information
    latitude = Column(FloatType, nullable=False)
    longitude = Column(FloatType, nullable=False)
    geom = Column(Geometry('POINT', srid=4326))
    
    # Temporal information
    juld = Column(FloatType)
    measurement_date = Column(DateTime(timezone=True), nullable=False)
    measurement_year = Column(Integer)
    measurement_month = Column(Integer)
    measurement_season = Column(Integer)
    
    # Data quality and processing
    data_mode = Column(String(1))
    position_qc = Column(CHAR(1))
    vertical_sampling_scheme = Column(Text)
    
    # Profile summary statistics
    max_depth_m = Column(FloatType)
    min_temperature_c = Column(FloatType)
    max_temperature_c = Column(FloatType)
    min_salinity_psu = Column(FloatType)
    max_salinity_psu = Column(FloatType)
    valid_measurements_count = Column(Integer, default=0)
    
    # Quality flags for entire profile
    profile_temp_qc = Column(CHAR(1))
    profile_psal_qc = Column(CHAR(1))
    profile_pres_qc = Column(CHAR(1))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    float = relationship("Float", back_populates="profiles")
    measurements = relationship("Measurement", back_populates="profile", cascade="all, delete-orphan")
    statistics = relationship("ProfileStatistic", back_populates="profile", uselist=False)
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('float_id', 'cycle_number', name='uq_float_cycle'),
        CheckConstraint(
            'latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180',
            name='valid_coordinates'
        ),
        CheckConstraint(
            "position_qc IN ('1', '2', '3', '4', '5', '8', '9') AND "
            "profile_temp_qc IN ('1', '2', '3', '4', '5', '8', '9') AND "
            "profile_psal_qc IN ('1', '2', '3', '4', '5', '8', '9') AND "
            "profile_pres_qc IN ('1', '2', '3', '4', '5', '8', '9')",
            name='valid_qc_flags'
        ),
        Index('idx_profiles_geom', 'geom', postgresql_using='gist'),
        Index('idx_profiles_date', 'measurement_date'),
        Index('idx_profiles_year_month', 'measurement_year', 'measurement_month'),
        Index('idx_profiles_season', 'measurement_season'),
        Index('idx_profiles_float_id', 'float_id'),
        Index('idx_profiles_cycle', 'float_id', 'cycle_number'),
        Index('idx_spatial_temporal_quality', 'geom', 'measurement_date', 'data_mode',
              postgresql_using='gist'),
    )
    
    @hybrid_property
    def season_name(self) -> str:
        """Get season name from season number."""
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
        return seasons.get(self.measurement_season, 'Unknown')
    
    @validates('latitude')
    def validate_latitude(self, key, value):
        if not -90 <= value <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return value
        
    @validates('longitude')
    def validate_longitude(self, key, value):
        if not -180 <= value <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return value
    
    def __repr__(self) -> str:
        return f"<Profile(float_id={self.float_id}, cycle={self.cycle_number}, date={self.measurement_date})>"


class Measurement(Base):
    """Individual depth measurement from ARGO profile."""
    
    __tablename__ = "measurements"
    
    id = Column(BigInteger, primary_key=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    
    # Depth information
    depth_level = Column(Integer, nullable=False)
    pressure_db = Column(REAL)
    depth_m = Column(REAL)
    
    # Oceanographic parameters
    temperature_c = Column(REAL)
    salinity_psu = Column(REAL)
    
    # Adjusted values (delayed-mode processing)
    temperature_adjusted_c = Column(REAL)
    salinity_adjusted_c = Column(REAL)
    pressure_adjusted_db = Column(REAL)
    
    # Quality control flags
    temperature_qc = Column(CHAR(1))
    salinity_qc = Column(CHAR(1))
    pressure_qc = Column(CHAR(1))
    temperature_adjusted_qc = Column(CHAR(1))
    salinity_adjusted_qc = Column(CHAR(1))
    pressure_adjusted_qc = Column(CHAR(1))
    
    # Error estimates
    temperature_error = Column(REAL)
    salinity_error = Column(REAL)
    pressure_error = Column(REAL)
    
    # Data validity (computed column)
    is_valid = Column(Boolean)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    profile = relationship("Profile", back_populates="measurements")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            'pressure_db >= 0 AND '
            'temperature_c >= -3 AND temperature_c <= 50 AND '
            'salinity_psu >= 0 AND salinity_psu <= 50',
            name='valid_measurements'
        ),
        CheckConstraint(
            "temperature_qc IN ('1', '2', '3', '4', '5', '8', '9') AND "
            "salinity_qc IN ('1', '2', '3', '4', '5', '8', '9') AND "
            "pressure_qc IN ('1', '2', '3', '4', '5', '8', '9')",
            name='valid_qc_flags'
        ),
        Index('idx_measurements_profile', 'profile_id'),
        Index('idx_measurements_depth', 'depth_m', postgresql_where=text('is_valid = true')),
        Index('idx_measurements_temp', 'temperature_c', postgresql_where=text('is_valid = true')),
        Index('idx_measurements_sal', 'salinity_psu', postgresql_where=text('is_valid = true')),
        Index('idx_measurements_valid', 'is_valid', 'pressure_qc', 'temperature_qc', 'salinity_qc'),
    )
    
    @hybrid_property
    def is_good_quality(self) -> bool:
        """Check if measurement has good quality flags."""
        return (self.temperature_qc in ('1', '2') and 
                self.salinity_qc in ('1', '2') and 
                self.pressure_qc in ('1', '2'))
    
    @hybrid_property
    def preferred_temperature(self) -> Optional[float]:
        """Get preferred temperature value (adjusted if available and valid)."""
        if (self.temperature_adjusted_c is not None and 
            self.temperature_adjusted_qc in ('1', '2')):
            return self.temperature_adjusted_c
        elif self.temperature_qc in ('1', '2'):
            return self.temperature_c
        return None
    
    @hybrid_property
    def preferred_salinity(self) -> Optional[float]:
        """Get preferred salinity value (adjusted if available and valid)."""
        if (self.salinity_adjusted_c is not None and 
            self.salinity_adjusted_qc in ('1', '2')):
            return self.salinity_adjusted_c
        elif self.salinity_qc in ('1', '2'):
            return self.salinity_psu
        return None
    
    def __repr__(self) -> str:
        return f"<Measurement(profile_id={self.profile_id}, depth={self.depth_m}m, temp={self.temperature_c}°C)>"


class ProfileStatistic(Base):
    """Pre-computed statistics for ARGO profiles to optimize LLM queries."""
    
    __tablename__ = "profile_statistics"
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    
    # Temperature statistics
    temp_mean = Column(REAL)
    temp_std = Column(REAL)
    temp_min = Column(REAL)
    temp_max = Column(REAL)
    temp_surface = Column(REAL)
    temp_bottom = Column(REAL)
    
    # Salinity statistics
    sal_mean = Column(REAL)
    sal_std = Column(REAL)
    sal_min = Column(REAL)
    sal_max = Column(REAL)
    sal_surface = Column(REAL)
    sal_bottom = Column(REAL)
    
    # Depth statistics
    max_depth = Column(REAL)
    valid_depths_count = Column(Integer)
    
    # Derived parameters
    mixed_layer_depth = Column(REAL)
    thermocline_depth = Column(REAL)
    
    # Quality metrics
    data_quality_score = Column(REAL)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    profile = relationship("Profile", back_populates="statistics")
    
    def __repr__(self) -> str:
        return f"<ProfileStatistic(profile_id={self.profile_id}, quality_score={self.data_quality_score})>"


class MonthlyClimatology(Base):
    """Monthly aggregated oceanographic data for trend analysis."""
    
    __tablename__ = "monthly_climatology"
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey("ocean_regions.id"))
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    depth_range_start = Column(REAL, nullable=False)
    depth_range_end = Column(REAL, nullable=False)
    
    # Aggregated measurements
    avg_temperature = Column(REAL)
    std_temperature = Column(REAL)
    avg_salinity = Column(REAL)
    std_salinity = Column(REAL)
    measurement_count = Column(Integer)
    
    # Spatial coverage
    latitude_range = Column(NUMRANGE)
    longitude_range = Column(NUMRANGE)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    region = relationship("OceanRegion", back_populates="monthly_climatology")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('region_id', 'year', 'month', 'depth_range_start', 'depth_range_end',
                        name='uq_monthly_climatology'),
    )
    
    @hybrid_property
    def date_range(self) -> str:
        """Get readable date range."""
        return f"{self.year}-{self.month:02d}"
    
    @hybrid_property
    def depth_range(self) -> str:
        """Get readable depth range."""
        return f"{self.depth_range_start}-{self.depth_range_end}m"
    
    def __repr__(self) -> str:
        return f"<MonthlyClimatology(region_id={self.region_id}, {self.date_range}, {self.depth_range})>"


# Additional indexes for cross-table queries
Index('idx_measurements_profile_depth', Measurement.profile_id, Measurement.depth_m)
Index('idx_profile_stats_temp', ProfileStatistic.temp_mean, ProfileStatistic.temp_std)
Index('idx_profile_stats_sal', ProfileStatistic.sal_mean, ProfileStatistic.sal_std)