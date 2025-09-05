"""
Domain entities for FloatChat.

This package contains the core domain entities and data models
for the FloatChat application.
"""

from .argo_entities import (
    Base,
    DAC,
    OceanRegion, 
    Float,
    Profile,
    Measurement,
    ProfileStatistic,
    MonthlyClimatology
)

__all__ = [
    "Base",
    "DAC",
    "OceanRegion",
    "Float", 
    "Profile",
    "Measurement",
    "ProfileStatistic",
    "MonthlyClimatology"
]