#!/usr/bin/env python3
"""
Simple database setup for FloatChat - inject Indian Ocean ARGO data.
"""

import asyncio
import sys
from pathlib import Path
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Main setup function."""
    print("FloatChat database setup for Indian Ocean ARGO data...")
    
    try:
        # Import SQLAlchemy
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy import func, select
        
        # Import our simplified entities
        from simple_entities import Base, DAC, Float, Profile, Measurement
        
        # Create SQLite database
        print("Creating database...")
        engine = create_async_engine("sqlite+aiosqlite:///floatchat_indian_ocean.db", echo=False)
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("Database schema created!")
        
        # Create session factory
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Insert Indian Ocean ARGO data
        print("Inserting Indian Ocean ARGO data...")
        async with session_factory() as session:
            # Create Indian Ocean DAC
            indian_dac = DAC(
                code="incois",
                name="Indian National Centre for Ocean Information Services", 
                country="India",
                institution="Ministry of Earth Sciences"
            )
            session.add(indian_dac)
            await session.flush()
            
            # Create multiple floats in Indian Ocean region
            floats_data = [
                {"platform_number": "2900226", "lat": 15.0, "lon": 75.0, "name": "Central Indian Ocean"},
                {"platform_number": "2900227", "lat": 10.5, "lon": 77.5, "name": "South Indian Ocean"}, 
                {"platform_number": "2900228", "lat": 18.2, "lon": 72.8, "name": "Arabian Sea"},
                {"platform_number": "2900229", "lat": 13.1, "lon": 87.3, "name": "Bay of Bengal"},
                {"platform_number": "2900230", "lat": 8.5, "lon": 76.9, "name": "Tropical Indian Ocean"}
            ]
            
            created_floats = []
            
            for float_data in floats_data:
                sample_float = Float(
                    platform_number=float_data["platform_number"],
                    dac_id=indian_dac.id,
                    deployment_latitude=float_data["lat"],
                    deployment_longitude=float_data["lon"],
                    project_name=f"Indian Ocean ARGO - {float_data['name']}",
                    pi_name="Dr. Indian Ocean Researcher",
                    platform_type="APEX",
                    is_active=True
                )
                session.add(sample_float)
                await session.flush()
                created_floats.append(sample_float)
            
            print(f"Created {len(created_floats)} floats in Indian Ocean region")
            
            # Create profiles for each float
            total_profiles = 0
            total_measurements = 0
            
            for float_obj in created_floats:
                for cycle in range(1, 11):  # 10 profiles per float
                    # Simulate float drift
                    lat_drift = float_obj.deployment_latitude + (cycle * 0.05 * np.random.uniform(-1, 1))
                    lon_drift = float_obj.deployment_longitude + (cycle * 0.05 * np.random.uniform(-1, 1))
                    
                    profile = Profile(
                        float_id=float_obj.id,
                        cycle_number=cycle,
                        latitude=lat_drift,
                        longitude=lon_drift,
                        measurement_date=datetime(2024, (cycle % 12) + 1, 15),
                        data_mode='D',  # Delayed mode
                        direction='A'
                    )
                    session.add(profile)
                    await session.flush()
                    total_profiles += 1
                    
                    # Add realistic measurements for Indian Ocean
                    depths = np.logspace(0, 3, 50)  # 50 levels from 1m to 1000m
                    
                    for depth_idx, depth in enumerate(depths):
                        pressure = depth * 1.02  # Approximate pressure
                        
                        # Realistic Indian Ocean temperature profile
                        if depth < 50:  # Mixed layer
                            temperature = 28.0 + np.random.normal(0, 0.5)
                        elif depth < 200:  # Thermocline
                            temperature = 28.0 - (depth - 50) * 0.15 + np.random.normal(0, 0.3)
                        else:  # Deep water
                            temperature = 6.0 - (depth - 200) * 0.003 + np.random.normal(0, 0.2)
                        
                        # Realistic Indian Ocean salinity profile
                        if depth < 100:
                            salinity = 34.5 + np.random.normal(0, 0.1)
                        elif depth < 500:
                            salinity = 34.7 + np.sin(depth / 100) * 0.2 + np.random.normal(0, 0.05)
                        else:
                            salinity = 34.6 + np.random.normal(0, 0.03)
                        
                        # Quality flags (mostly good quality)
                        temp_qc = '1' if np.random.random() > 0.1 else '2'
                        sal_qc = '1' if np.random.random() > 0.1 else '2'
                        pres_qc = '1'
                        
                        measurement = Measurement(
                            profile_id=profile.id,
                            depth_level=depth_idx,
                            depth_m=float(depth),
                            pressure_db=float(pressure),
                            temperature_c=float(temperature),
                            salinity_psu=float(salinity),
                            temperature_qc=temp_qc,
                            salinity_qc=sal_qc,
                            pressure_qc=pres_qc,
                            is_valid=(temp_qc in ['1', '2'] and sal_qc in ['1', '2'])
                        )
                        session.add(measurement)
                        total_measurements += 1
                    
                    # Update profile statistics
                    profile.max_depth_m = float(max(depths))
                    profile.min_temperature_c = float(min(28.0 - (depth - 50) * 0.15 for depth in depths if depth >= 50) or 28.0)
                    profile.max_temperature_c = 28.5
                    profile.min_salinity_psu = 34.4
                    profile.max_salinity_psu = 34.8
                    profile.valid_measurements_count = len(depths)
            
            await session.commit()
            print(f"Created {total_profiles} profiles with {total_measurements} measurements")
            
        print("Indian Ocean ARGO data injection complete!")
        
        # Show final summary
        async with session_factory() as session:
            float_count = await session.scalar(select(func.count(Float.id)))
            profile_count = await session.scalar(select(func.count(Profile.id))) 
            measurement_count = await session.scalar(select(func.count(Measurement.id)))
            valid_measurements = await session.scalar(
                select(func.count(Measurement.id)).where(Measurement.is_valid == True)
            )
            
            print("\n=== DATABASE SUMMARY ===")
            print(f"Floats: {float_count}")
            print(f"Profiles: {profile_count}")
            print(f"Total Measurements: {measurement_count}")
            print(f"Valid Measurements: {valid_measurements}")
            print(f"Data Quality: {(valid_measurements/measurement_count*100):.1f}%")
            print(f"Database: floatchat_indian_ocean.db")
            
            # Show geographic coverage
            result = await session.execute(select(
                func.min(Profile.latitude), func.max(Profile.latitude),
                func.min(Profile.longitude), func.max(Profile.longitude)
            ))
            min_lat, max_lat, min_lon, max_lon = result.first()
            print(f"Geographic Coverage:")
            print(f"  Latitude: {min_lat:.2f} to {max_lat:.2f}")
            print(f"  Longitude: {min_lon:.2f} to {max_lon:.2f}")
        
        await engine.dispose()
        print("\nDatabase setup complete! Ready for FloatChat queries.")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())