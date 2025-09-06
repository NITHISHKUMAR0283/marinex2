#!/usr/bin/env python3
"""
Extended database setup for FloatChat - inject comprehensive ARGO float data.
Creates a realistic global ARGO network with multiple ocean regions and extensive data.
"""

import asyncio
import sys
from pathlib import Path
import os
import numpy as np
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Main setup function with comprehensive ARGO data."""
    print("FloatChat Extended Database Setup - Comprehensive ARGO Network...")
    
    try:
        # Import SQLAlchemy
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy import func, select
        
        # Import our simplified entities
        from simple_entities import Base, DAC, Float, Profile, Measurement
        
        # Create SQLite database
        print("Creating extended database...")
        engine = create_async_engine("sqlite+aiosqlite:///floatchat_global_argo.db", echo=False)
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("Database schema created!")
        
        # Create session factory
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Define comprehensive DAC data
        dacs_data = [
            {"code": "incois", "name": "Indian National Centre for Ocean Information Services", "country": "India", "institution": "Ministry of Earth Sciences"},
            {"code": "csio", "name": "CSIR-National Institute of Oceanography", "country": "India", "institution": "Council of Scientific and Industrial Research"},
            {"code": "aoml", "name": "Atlantic Oceanographic and Meteorological Laboratory", "country": "USA", "institution": "NOAA"},
            {"code": "coriolis", "name": "Coriolis Operational Oceanography", "country": "France", "institution": "Ifremer"},
            {"code": "jma", "name": "Japan Meteorological Agency", "country": "Japan", "institution": "JMA"},
            {"code": "kma", "name": "Korea Meteorological Administration", "country": "South Korea", "institution": "KMA"},
            {"code": "csiro", "name": "Commonwealth Scientific and Industrial Research", "country": "Australia", "institution": "CSIRO"},
            {"code": "ukmo", "name": "UK Met Office", "country": "United Kingdom", "institution": "Met Office"}
        ]
        
        # Insert comprehensive ARGO data
        print("Inserting comprehensive global ARGO data...")
        async with session_factory() as session:
            # Create multiple DACs
            created_dacs = {}
            for dac_data in dacs_data:
                dac = DAC(**dac_data)
                session.add(dac)
                await session.flush()
                created_dacs[dac_data["code"]] = dac
                print(f"Created DAC: {dac.name}")
            
            # Define comprehensive float data across global oceans
            floats_data = [
                # Indian Ocean - Extended Coverage
                {"platform_number": "2900226", "lat": 15.0, "lon": 75.0, "region": "Central Indian Ocean", "dac": "incois", "project": "Indian Ocean ARGO Network"},
                {"platform_number": "2900227", "lat": 10.5, "lon": 77.5, "region": "South Indian Ocean", "dac": "incois", "project": "Indian Ocean ARGO Network"}, 
                {"platform_number": "2900228", "lat": 18.2, "lon": 72.8, "region": "Arabian Sea", "dac": "incois", "project": "Arabian Sea Monitoring"},
                {"platform_number": "2900229", "lat": 13.1, "lon": 87.3, "region": "Bay of Bengal", "dac": "incois", "project": "Bay of Bengal Studies"},
                {"platform_number": "2900230", "lat": 8.5, "lon": 76.9, "region": "Tropical Indian Ocean", "dac": "incois", "project": "Tropical Ocean Dynamics"},
                {"platform_number": "2900231", "lat": 5.2, "lon": 73.4, "region": "Equatorial Indian Ocean", "dac": "incois", "project": "Equatorial Currents"},
                {"platform_number": "2900232", "lat": 20.1, "lon": 70.5, "region": "Northern Arabian Sea", "dac": "incois", "project": "Monsoon Impact Study"},
                {"platform_number": "2900233", "lat": 12.8, "lon": 93.2, "region": "Andaman Sea", "dac": "csio", "project": "Southeast Asian Waters"},
                
                # Pacific Ocean
                {"platform_number": "2900300", "lat": 35.5, "lon": 140.2, "region": "North Pacific", "dac": "jma", "project": "Kuroshio Current Study"},
                {"platform_number": "2900301", "lat": 25.8, "lon": 160.1, "region": "Central Pacific", "dac": "aoml", "project": "Pacific Climate Monitoring"},
                {"platform_number": "2900302", "lat": 10.2, "lon": -170.5, "region": "Tropical Pacific", "dac": "aoml", "project": "ENSO Monitoring"},
                {"platform_number": "2900303", "lat": -15.3, "lon": -150.8, "region": "South Pacific", "dac": "csiro", "project": "Southern Ocean Dynamics"},
                {"platform_number": "2900304", "lat": 45.1, "lon": -140.2, "region": "North Pacific Gyre", "dac": "aoml", "project": "Ocean Gyre Study"},
                {"platform_number": "2900305", "lat": 0.5, "lon": 180.0, "region": "Equatorial Pacific", "dac": "jma", "project": "Equatorial Undercurrent"},
                {"platform_number": "2900306", "lat": 38.2, "lon": 125.8, "region": "East China Sea", "dac": "kma", "project": "East Asian Marginal Seas"},
                
                # Atlantic Ocean
                {"platform_number": "2900400", "lat": 40.2, "lon": -30.5, "region": "North Atlantic", "dac": "coriolis", "project": "North Atlantic Current"},
                {"platform_number": "2900401", "lat": 25.1, "lon": -40.8, "region": "Central Atlantic", "dac": "coriolis", "project": "Atlantic Meridional Circulation"},
                {"platform_number": "2900402", "lat": 10.8, "lon": -25.2, "region": "Tropical Atlantic", "dac": "coriolis", "project": "Tropical Atlantic Variability"},
                {"platform_number": "2900403", "lat": -20.5, "lon": -10.1, "region": "South Atlantic", "dac": "coriolis", "project": "South Atlantic Gyre"},
                {"platform_number": "2900404", "lat": 60.2, "lon": -20.8, "region": "Nordic Seas", "dac": "ukmo", "project": "Arctic-Atlantic Exchange"},
                {"platform_number": "2900405", "lat": 35.8, "lon": -65.2, "region": "Sargasso Sea", "dac": "aoml", "project": "Subtropical Gyre Dynamics"},
                
                # Southern Ocean
                {"platform_number": "2900500", "lat": -45.2, "lon": 45.8, "region": "Southern Indian Ocean", "dac": "csiro", "project": "Southern Ocean Carbon"},
                {"platform_number": "2900501", "lat": -55.1, "lon": 80.5, "region": "Subantarctic Zone", "dac": "csiro", "project": "Antarctic Circumpolar Current"},
                {"platform_number": "2900502", "lat": -40.8, "lon": 140.2, "region": "Southern Pacific", "dac": "csiro", "project": "Pacific Sector Antarctica"},
                {"platform_number": "2900503", "lat": -50.5, "lon": -60.1, "region": "Drake Passage", "dac": "aoml", "project": "Antarctic Frontal Systems"},
                
                # Arctic Ocean
                {"platform_number": "2900600", "lat": 75.2, "lon": 15.8, "region": "Arctic Ocean", "dac": "ukmo", "project": "Arctic Climate Change"},
                {"platform_number": "2900601", "lat": 70.5, "lon": -150.2, "region": "Beaufort Sea", "dac": "aoml", "project": "Arctic Sea Ice Dynamics"},
                
                # Mediterranean Sea
                {"platform_number": "2900700", "lat": 40.5, "lon": 15.2, "region": "Mediterranean Sea", "dac": "coriolis", "project": "Mediterranean Water Masses"},
                {"platform_number": "2900701", "lat": 35.8, "lon": 25.1, "region": "Eastern Mediterranean", "dac": "coriolis", "project": "Deep Water Formation"},
            ]
            
            created_floats = []
            
            print(f"Creating {len(floats_data)} floats across global oceans...")
            for float_data in floats_data:
                dac = created_dacs[float_data["dac"]]
                
                sample_float = Float(
                    platform_number=float_data["platform_number"],
                    dac_id=dac.id,
                    deployment_latitude=float_data["lat"],
                    deployment_longitude=float_data["lon"],
                    project_name=float_data["project"],
                    pi_name=f"Dr. {float_data['region']} Researcher",
                    platform_type=random.choice(["APEX", "NOVA", "ARVOR", "PROVOR"]),
                    is_active=random.choice([True, True, True, False])  # 75% active
                )
                session.add(sample_float)
                await session.flush()
                created_floats.append((sample_float, float_data))
            
            print(f"Created {len(created_floats)} floats across global ocean regions")
            
            # Create profiles for each float with varying amounts
            total_profiles = 0
            total_measurements = 0
            
            for float_obj, float_data in created_floats:
                # Vary number of profiles based on float age and activity
                if float_obj.is_active:
                    profile_count = random.randint(15, 35)  # Active floats have more profiles
                else:
                    profile_count = random.randint(5, 15)   # Inactive floats have fewer
                
                print(f"Creating {profile_count} profiles for {float_data['region']} float...")
                
                for cycle in range(1, profile_count + 1):
                    # Simulate realistic float drift
                    days_since_deployment = cycle * random.randint(8, 12)  # 8-12 days between cycles
                    measurement_date = datetime(2023, 1, 1) + timedelta(days=days_since_deployment)
                    
                    # Realistic drift patterns
                    drift_factor = cycle * 0.02 * random.uniform(0.5, 2.0)
                    lat_drift = float_obj.deployment_latitude + drift_factor * np.random.normal(0, 1)
                    lon_drift = float_obj.deployment_longitude + drift_factor * np.random.normal(0, 1)
                    
                    # Keep within realistic bounds
                    lat_drift = max(-85, min(85, lat_drift))
                    lon_drift = (lon_drift + 180) % 360 - 180  # Wrap longitude
                    
                    profile = Profile(
                        float_id=float_obj.id,
                        cycle_number=cycle,
                        latitude=lat_drift,
                        longitude=lon_drift,
                        measurement_date=measurement_date,
                        data_mode=random.choice(['R', 'D', 'D', 'D']),  # Mostly delayed mode
                        direction=random.choice(['A', 'D'])
                    )
                    session.add(profile)
                    await session.flush()
                    total_profiles += 1
                    
                    # Create realistic measurements based on ocean region
                    depth_levels = random.randint(40, 80)  # Variable depth sampling
                    depths = np.logspace(0, 3, depth_levels)  # 1m to 1000m
                    
                    # Region-specific oceanographic characteristics
                    region = float_data["region"].lower()
                    
                    for depth_idx, depth in enumerate(depths):
                        pressure = depth * 1.02  # Approximate pressure
                        
                        # Region-specific temperature and salinity profiles
                        if "indian" in region or "arabian" in region or "bay of bengal" in region:
                            # Warm tropical/subtropical waters
                            if depth < 50:
                                temperature = 28.0 + np.random.normal(0, 1.0)
                            elif depth < 200:
                                temperature = 28.0 - (depth - 50) * 0.15 + np.random.normal(0, 0.5)
                            else:
                                temperature = 6.0 - (depth - 200) * 0.003 + np.random.normal(0, 0.3)
                            
                            salinity = 34.5 + np.sin(depth / 100) * 0.3 + np.random.normal(0, 0.1)
                            
                        elif "pacific" in region:
                            # Pacific characteristics
                            if depth < 100:
                                temperature = 24.0 + np.random.normal(0, 2.0)
                            elif depth < 300:
                                temperature = 20.0 - (depth - 100) * 0.08 + np.random.normal(0, 0.5)
                            else:
                                temperature = 4.0 - (depth - 300) * 0.002 + np.random.normal(0, 0.2)
                                
                            salinity = 34.2 + np.cos(depth / 150) * 0.4 + np.random.normal(0, 0.1)
                            
                        elif "atlantic" in region:
                            # Atlantic characteristics  
                            if depth < 80:
                                temperature = 22.0 + np.random.normal(0, 1.5)
                            elif depth < 250:
                                temperature = 18.0 - (depth - 80) * 0.1 + np.random.normal(0, 0.4)
                            else:
                                temperature = 4.5 - (depth - 250) * 0.0025 + np.random.normal(0, 0.2)
                                
                            salinity = 35.0 + np.sin(depth / 120) * 0.25 + np.random.normal(0, 0.08)
                            
                        elif "southern" in region or "antarctic" in region:
                            # Cold southern waters
                            if depth < 60:
                                temperature = 8.0 + np.random.normal(0, 1.0)
                            elif depth < 200:
                                temperature = 5.0 - (depth - 60) * 0.02 + np.random.normal(0, 0.3)
                            else:
                                temperature = 2.0 - (depth - 200) * 0.001 + np.random.normal(0, 0.1)
                                
                            salinity = 34.0 + np.random.normal(0, 0.05)
                            
                        elif "arctic" in region:
                            # Arctic waters
                            temperature = max(-1.8, 2.0 - depth * 0.002 + np.random.normal(0, 0.5))
                            salinity = 32.0 + depth * 0.002 + np.random.normal(0, 0.1)
                            
                        else:
                            # Default temperate waters
                            temperature = 20.0 - depth * 0.015 + np.random.normal(0, 1.0)
                            salinity = 34.0 + np.random.normal(0, 0.2)
                        
                        # Ensure realistic bounds
                        temperature = max(-2.0, min(35.0, temperature))
                        salinity = max(30.0, min(37.0, salinity))
                        
                        # Quality flags (mostly good quality)
                        temp_qc = '1' if np.random.random() > 0.15 else '2'
                        sal_qc = '1' if np.random.random() > 0.12 else '2'
                        pres_qc = '1' if np.random.random() > 0.05 else '2'
                        
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
                            is_valid=(temp_qc in ['1', '2'] and sal_qc in ['1', '2'] and pres_qc in ['1', '2'])
                        )
                        session.add(measurement)
                        total_measurements += 1
                    
                    # Update profile statistics
                    profile.max_depth_m = float(max(depths))
                    profile.min_temperature_c = float(min(temperature - 2, temperature))
                    profile.max_temperature_c = float(max(temperature + 2, temperature))
                    profile.min_salinity_psu = float(min(salinity - 0.5, salinity))
                    profile.max_salinity_psu = float(max(salinity + 0.5, salinity))
                    profile.valid_measurements_count = depth_levels
                
                # Commit every 5 floats to avoid memory issues
                if (len([f for f, _ in created_floats if f.id <= float_obj.id]) % 5) == 0:
                    await session.commit()
                    print(f"Committed batch... {total_profiles} profiles, {total_measurements} measurements so far")
            
            # Final commit
            await session.commit()
            print(f"Created {total_profiles} profiles with {total_measurements} measurements")
            
        print("Comprehensive global ARGO data injection complete!")
        
        # Show comprehensive summary
        async with session_factory() as session:
            dac_count = await session.scalar(select(func.count(DAC.id)))
            float_count = await session.scalar(select(func.count(Float.id)))
            profile_count = await session.scalar(select(func.count(Profile.id))) 
            measurement_count = await session.scalar(select(func.count(Measurement.id)))
            valid_measurements = await session.scalar(
                select(func.count(Measurement.id)).where(Measurement.is_valid == True)
            )
            active_floats = await session.scalar(
                select(func.count(Float.id)).where(Float.is_active == True)
            )
            
            print("\n=== COMPREHENSIVE DATABASE SUMMARY ===")
            print(f"Data Assembly Centers: {dac_count}")
            print(f"Total Floats: {float_count} ({active_floats} active)")
            print(f"Total Profiles: {profile_count}")
            print(f"Total Measurements: {measurement_count}")
            print(f"Valid Measurements: {valid_measurements}")
            print(f"Data Quality: {(valid_measurements/measurement_count*100):.1f}%")
            print(f"Database: floatchat_global_argo.db")
            
            # Show geographic coverage
            result = await session.execute(select(
                func.min(Profile.latitude), func.max(Profile.latitude),
                func.min(Profile.longitude), func.max(Profile.longitude)
            ))
            min_lat, max_lat, min_lon, max_lon = result.first()
            print(f"Global Geographic Coverage:")
            print(f"  Latitude: {min_lat:.2f} to {max_lat:.2f}")
            print(f"  Longitude: {min_lon:.2f} to {max_lon:.2f}")
            
            # Show temporal coverage
            result = await session.execute(select(
                func.min(Profile.measurement_date), func.max(Profile.measurement_date)
            ))
            min_date, max_date = result.first()
            print(f"Temporal Coverage: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            
            # Show regional distribution
            print("\nRegional Distribution:")
            result = await session.execute(select(Float.project_name, func.count(Float.id)).group_by(Float.project_name).order_by(func.count(Float.id).desc()))
            for project, count in result:
                print(f"  {project}: {count} floats")
        
        await engine.dispose()
        print("\nExtended database setup complete! Ready for comprehensive FloatChat queries.")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())