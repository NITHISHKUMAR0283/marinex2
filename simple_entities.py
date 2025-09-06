"""
Simplified entities for SQLite database setup.
"""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, DateTime, Date, ForeignKey, Text, Boolean, REAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DAC(Base):
    """Data Assembly Center - simplified version."""
    
    __tablename__ = "dacs"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    country = Column(String(100))
    institution = Column(String(255))
    
    # Relationships
    floats = relationship("Float", back_populates="dac")
    
    def __repr__(self) -> str:
        return f"<DAC(code='{self.code}', name='{self.name}')>"


class Float(Base):
    """ARGO float - simplified version."""
    
    __tablename__ = "floats"
    
    id = Column(Integer, primary_key=True)
    platform_number = Column(String(20), unique=True, nullable=False, index=True)
    dac_id = Column(Integer, ForeignKey("dacs.id"))
    
    # Physical characteristics
    platform_type = Column(String(50))
    
    # Deployment information
    deployment_date = Column(Date)
    deployment_latitude = Column(REAL)
    deployment_longitude = Column(REAL)
    
    # Mission details
    project_name = Column(String(255))
    pi_name = Column(String(255))
    
    # Status tracking
    is_active = Column(Boolean, default=True)
    total_profiles = Column(Integer, default=0)
    
    # Relationships
    dac = relationship("DAC", back_populates="floats")
    profiles = relationship("Profile", back_populates="float")
    
    def __repr__(self) -> str:
        return f"<Float(platform_number='{self.platform_number}', active={self.is_active})>"


class Profile(Base):
    """Individual ARGO profile - simplified version."""
    
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True)
    float_id = Column(Integer, ForeignKey("floats.id"), nullable=False)
    
    # Profile identification
    cycle_number = Column(Integer, nullable=False)
    direction = Column(String(1))
    
    # Spatial-temporal information
    latitude = Column(REAL, nullable=False)
    longitude = Column(REAL, nullable=False)
    measurement_date = Column(DateTime, nullable=False)
    
    # Data quality
    data_mode = Column(String(1))  # 'R' real-time, 'D' delayed-mode
    
    # Profile summary statistics
    max_depth_m = Column(REAL)
    min_temperature_c = Column(REAL)
    max_temperature_c = Column(REAL)
    min_salinity_psu = Column(REAL)
    max_salinity_psu = Column(REAL)
    valid_measurements_count = Column(Integer, default=0)
    
    # Relationships
    float = relationship("Float", back_populates="profiles")
    measurements = relationship("Measurement", back_populates="profile")
    
    def __repr__(self) -> str:
        return f"<Profile(float_id={self.float_id}, cycle={self.cycle_number}, date={self.measurement_date})>"


class Measurement(Base):
    """Individual depth measurement - simplified version."""
    
    __tablename__ = "measurements"
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    
    # Depth information
    depth_level = Column(Integer, nullable=False)
    pressure_db = Column(REAL)
    depth_m = Column(REAL)
    
    # Oceanographic parameters
    temperature_c = Column(REAL)
    salinity_psu = Column(REAL)
    
    # Quality control flags
    temperature_qc = Column(String(1))
    salinity_qc = Column(String(1))
    pressure_qc = Column(String(1))
    
    # Data validity
    is_valid = Column(Boolean, default=True)
    
    # Relationships
    profile = relationship("Profile", back_populates="measurements")
    
    def __repr__(self) -> str:
        return f"<Measurement(profile_id={self.profile_id}, depth={self.depth_m}m, temp={self.temperature_c}°C)>"