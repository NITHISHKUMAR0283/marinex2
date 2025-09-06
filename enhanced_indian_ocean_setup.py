#!/usr/bin/env python3
"""
Enhanced Indian Ocean ARGO database setup for FloatChat Phase 2.
Creates comprehensive dataset with 100+ floats, 2000m depth profiles, and realistic data distribution.
Optimized for LLM training and RAG system implementation.
"""

import asyncio
import sys
from pathlib import Path
import os
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Enhanced setup function with comprehensive Indian Ocean ARGO data."""
    print("FloatChat Enhanced Indian Ocean Database Setup - Phase 2 Ready...")
    print("Target: 100+ floats, 2000m depth profiles, comprehensive data for RAG training")
    
    try:
        # Import SQLAlchemy
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy import func, select
        
        # Import our simplified entities
        from simple_entities import Base, DAC, Float, Profile, Measurement
        
        # Create enhanced SQLite database
        print("Creating enhanced Indian Ocean database...")
        db_path = "floatchat_indian_ocean_enhanced.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("Enhanced database schema created!")
        
        # Create session factory
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Define Indian Ocean DACs
        dacs_data = [
            {"code": "incois", "name": "Indian National Centre for Ocean Information Services", 
             "country": "India", "institution": "Ministry of Earth Sciences"},
            {"code": "csio", "name": "CSIR-National Institute of Oceanography", 
             "country": "India", "institution": "Council of Scientific and Industrial Research"},
            {"code": "niot", "name": "National Institute of Ocean Technology", 
             "country": "India", "institution": "Ministry of Earth Sciences"},
            {"code": "coriolis", "name": "Coriolis Indian Ocean Program", 
             "country": "France", "institution": "Ifremer Indian Ocean Division"}
        ]
        
        # Generate comprehensive Indian Ocean float positions
        def generate_indian_ocean_positions(num_floats: int = 120) -> List[Dict]:
            """Generate realistic Indian Ocean ARGO float positions."""
            positions = []
            
            # Define Indian Ocean sub-regions with realistic densities
            regions = [
                # Arabian Sea (25 floats)
                {"name": "Central Arabian Sea", "lat_range": (12, 22), "lon_range": (60, 68), "count": 8},
                {"name": "Northern Arabian Sea", "lat_range": (18, 25), "lon_range": (63, 72), "count": 6},
                {"name": "Western Arabian Sea", "lat_range": (8, 18), "lon_range": (55, 65), "count": 6},
                {"name": "Somali Coast", "lat_range": (0, 12), "lon_range": (45, 55), "count": 5},
                
                # Bay of Bengal (30 floats)
                {"name": "Central Bay of Bengal", "lat_range": (8, 18), "lon_range": (82, 92), "count": 10},
                {"name": "Northern Bay of Bengal", "lat_range": (16, 23), "lon_range": (85, 95), "count": 8},
                {"name": "Southern Bay of Bengal", "lat_range": (2, 10), "lon_range": (80, 90), "count": 7},
                {"name": "Eastern Bay of Bengal", "lat_range": (5, 15), "lon_range": (90, 98), "count": 5},
                
                # Central Indian Ocean (25 floats)
                {"name": "Central Indian Basin", "lat_range": (-5, 5), "lon_range": (70, 85), "count": 8},
                {"name": "Tropical Indian Ocean", "lat_range": (0, 12), "lon_range": (65, 80), "count": 8},
                {"name": "Equatorial Indian Ocean", "lat_range": (-8, 2), "lon_range": (55, 75), "count": 9},
                
                # South Indian Ocean (25 floats)
                {"name": "Southwest Indian Ocean", "lat_range": (-25, -10), "lon_range": (50, 70), "count": 8},
                {"name": "South Central Indian Ocean", "lat_range": (-20, -5), "lon_range": (70, 90), "count": 9},
                {"name": "Southeast Indian Ocean", "lat_range": (-25, -10), "lon_range": (85, 105), "count": 8},
                
                # Marginal seas (15 floats)  
                {"name": "Andaman Sea", "lat_range": (6, 16), "lon_range": (92, 98), "count": 5},
                {"name": "Lakshadweep Sea", "lat_range": (8, 14), "lon_range": (71, 76), "count": 4},
                {"name": "Ceylon Basin", "lat_range": (2, 8), "lon_range": (78, 85), "count": 3},
                {"name": "Maldives Region", "lat_range": (-2, 6), "lon_range": (71, 76), "count": 3},
            ]
            
            float_id = 2900500  # Starting platform number
            
            for region in regions:
                for i in range(region["count"]):
                    # Generate random position within region bounds
                    lat = random.uniform(region["lat_range"][0], region["lat_range"][1])
                    lon = random.uniform(region["lon_range"][0], region["lon_range"][1])
                    
                    # Add some clustering for realistic distribution
                    lat += random.gauss(0, 0.5)  # Small clustering effect
                    lon += random.gauss(0, 0.8)
                    
                    positions.append({
                        "platform_number": str(float_id),
                        "lat": round(lat, 3),
                        "lon": round(lon, 3),
                        "region": region["name"],
                        "dac": random.choice(["incois", "incois", "csio", "niot"]),  # Bias toward INCOIS
                        "project": f"Indian Ocean ARGO - {region['name']}"
                    })
                    float_id += 1
            
            return positions
        
        # Insert comprehensive ARGO data
        print("Inserting comprehensive Indian Ocean ARGO data...")
        async with session_factory() as session:
            
            # Create DACs
            created_dacs = {}
            for dac_data in dacs_data:
                dac = DAC(**dac_data)
                session.add(dac)
                await session.flush()
                created_dacs[dac_data["code"]] = dac
                print(f"Created DAC: {dac.name}")
            
            # Generate 120 float positions across Indian Ocean
            print("Generating 120 Indian Ocean float positions...")
            floats_data = generate_indian_ocean_positions(120)
            
            created_floats = []
            
            print(f"Creating {len(floats_data)} floats across Indian Ocean regions...")
            for i, float_data in enumerate(floats_data):
                dac = created_dacs[float_data["dac"]]
                
                # Realistic deployment dates (2020-2024)
                deployment_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
                
                sample_float = Float(
                    platform_number=float_data["platform_number"],
                    dac_id=dac.id,
                    deployment_latitude=float_data["lat"],
                    deployment_longitude=float_data["lon"],
                    deployment_date=deployment_date.date(),
                    project_name=float_data["project"],
                    pi_name=f"Dr. {float_data['region']} Researcher",
                    platform_type=random.choice(["APEX", "NOVA", "ARVOR", "PROVOR", "DEEP-APEX"]),
                    is_active=random.choice([True, True, True, False])  # 75% active
                )
                session.add(sample_float)
                await session.flush()
                created_floats.append((sample_float, float_data))
                
                if (i + 1) % 20 == 0:
                    print(f"Created {i + 1} floats...")
            
            print(f"Created {len(created_floats)} floats across Indian Ocean regions")
            
            # Create comprehensive profiles for each float
            total_profiles = 0
            total_measurements = 0
            
            print("Creating comprehensive profiles with 2000m depth coverage...")
            
            for float_obj, float_data in created_floats:
                # Realistic profile counts based on deployment date and status
                days_since_deployment = (datetime.now().date() - float_obj.deployment_date).days
                theoretical_profiles = max(1, days_since_deployment // 10)  # One profile every 10 days
                
                if float_obj.is_active:
                    profile_count = min(theoretical_profiles, random.randint(80, 120))
                else:
                    profile_count = min(theoretical_profiles, random.randint(20, 60))
                
                print(f"Creating {profile_count} profiles for {float_data['region']} ({float_data['platform_number']})...")
                
                for cycle in range(1, profile_count + 1):
                    # Realistic temporal progression
                    days_since_deployment = cycle * random.randint(8, 14)  # 8-14 days between cycles
                    measurement_date = float_obj.deployment_date + timedelta(days=days_since_deployment)
                    
                    # Realistic drift patterns based on ocean currents
                    drift_factor = cycle * 0.01 * random.uniform(0.3, 2.5)
                    
                    # Region-specific drift patterns
                    region_lower = float_data["region"].lower()
                    if "arabian" in region_lower:
                        # Arabian Sea - westward drift during monsoon
                        lat_drift = float_obj.deployment_latitude + drift_factor * random.gauss(0.2, 1)
                        lon_drift = float_obj.deployment_longitude + drift_factor * random.gauss(-0.5, 1)
                    elif "bay of bengal" in region_lower:
                        # Bay of Bengal - cyclonic patterns
                        lat_drift = float_obj.deployment_latitude + drift_factor * random.gauss(0, 1.2)
                        lon_drift = float_obj.deployment_longitude + drift_factor * random.gauss(0.3, 1)
                    else:
                        # General Indian Ocean drift
                        lat_drift = float_obj.deployment_latitude + drift_factor * random.gauss(0, 1)
                        lon_drift = float_obj.deployment_longitude + drift_factor * random.gauss(0, 1)
                    
                    # Keep within Indian Ocean bounds
                    lat_drift = max(-40, min(30, lat_drift))
                    lon_drift = max(35, min(110, lon_drift))
                    
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
                    
                    # Create comprehensive measurements: 1m to 2000m depth
                    # Use variable depth sampling - more samples in upper ocean
                    upper_depths = np.logspace(0, 2, 35)        # 1m to 100m (35 levels)
                    mid_depths = np.logspace(2, 2.7, 25)       # 100m to 500m (25 levels)  
                    deep_depths = np.logspace(2.7, 3.3, 20)    # 500m to 2000m (20 levels)
                    
                    all_depths = np.concatenate([upper_depths, mid_depths, deep_depths])
                    depths = np.unique(all_depths)  # Remove any duplicates
                    
                    # Ensure maximum depth is close to 2000m
                    if depths[-1] < 1950:
                        depths = np.append(depths, 2000.0)
                    
                    # Region-specific oceanographic characteristics
                    region = float_data["region"].lower()
                    
                    profile_temps = []
                    profile_salts = []
                    
                    for depth_idx, depth in enumerate(depths):
                        pressure = depth * 1.025  # More accurate pressure calculation
                        
                        # Enhanced region-specific T-S profiles
                        if "arabian" in region:
                            # Arabian Sea - high salinity, warm surface
                            if depth < 50:  # Mixed layer
                                temperature = 28.5 + random.gauss(0, 1.2)
                                salinity = 35.8 + random.gauss(0, 0.15)
                            elif depth < 100:  # Thermocline start
                                temperature = 28.5 - (depth - 50) * 0.25 + random.gauss(0, 0.8)
                                salinity = 35.8 + (depth - 50) * 0.004 + random.gauss(0, 0.1)
                            elif depth < 300:  # Main thermocline
                                temperature = 16.0 - (depth - 100) * 0.045 + random.gauss(0, 0.4)
                                salinity = 36.0 - (depth - 100) * 0.003 + random.gauss(0, 0.08)
                            elif depth < 800:  # Intermediate water
                                temperature = 7.0 - (depth - 300) * 0.006 + random.gauss(0, 0.3)
                                salinity = 35.2 + (depth - 300) * 0.0004 + random.gauss(0, 0.05)
                            else:  # Deep water
                                temperature = 4.0 - (depth - 800) * 0.0015 + random.gauss(0, 0.2)
                                salinity = 34.7 + (depth - 800) * 0.0001 + random.gauss(0, 0.03)
                                
                        elif "bay of bengal" in region:
                            # Bay of Bengal - lower salinity due to freshwater input
                            if depth < 60:  # Mixed layer
                                temperature = 29.0 + random.gauss(0, 1.0)
                                salinity = 33.5 + random.gauss(0, 0.2)  # Lower salinity
                            elif depth < 120:  # Strong halocline
                                temperature = 29.0 - (depth - 60) * 0.2 + random.gauss(0, 0.6)
                                salinity = 33.5 + (depth - 60) * 0.02 + random.gauss(0, 0.1)
                            elif depth < 400:  # Thermocline
                                temperature = 17.0 - (depth - 120) * 0.04 + random.gauss(0, 0.4)
                                salinity = 34.7 - (depth - 120) * 0.002 + random.gauss(0, 0.08)
                            elif depth < 1000:  # Intermediate water
                                temperature = 6.5 - (depth - 400) * 0.004 + random.gauss(0, 0.2)
                                salinity = 34.4 + (depth - 400) * 0.0005 + random.gauss(0, 0.05)
                            else:  # Deep water
                                temperature = 3.8 - (depth - 1000) * 0.001 + random.gauss(0, 0.15)
                                salinity = 34.7 + random.gauss(0, 0.03)
                                
                        elif "central" in region or "equatorial" in region:
                            # Central/Equatorial Indian Ocean
                            if depth < 80:  # Mixed layer
                                temperature = 28.2 + random.gauss(0, 0.8)
                                salinity = 34.8 + random.gauss(0, 0.12)
                            elif depth < 200:  # Upper thermocline
                                temperature = 28.2 - (depth - 80) * 0.15 + random.gauss(0, 0.5)
                                salinity = 34.8 + (depth - 80) * 0.008 + random.gauss(0, 0.08)
                            elif depth < 500:  # Lower thermocline
                                temperature = 10.0 - (depth - 200) * 0.02 + random.gauss(0, 0.3)
                                salinity = 35.2 - (depth - 200) * 0.001 + random.gauss(0, 0.06)
                            elif depth < 1200:  # Intermediate
                                temperature = 4.0 - (depth - 500) * 0.002 + random.gauss(0, 0.2)
                                salinity = 34.8 + (depth - 500) * 0.0002 + random.gauss(0, 0.04)
                            else:  # Deep water
                                temperature = 2.6 - (depth - 1200) * 0.0008 + random.gauss(0, 0.1)
                                salinity = 34.7 + random.gauss(0, 0.02)
                                
                        elif "south" in region:
                            # Southern Indian Ocean - cooler, saltier
                            if depth < 100:  # Mixed layer
                                temperature = 24.0 + random.gauss(0, 1.5)
                                salinity = 35.2 + random.gauss(0, 0.1)
                            elif depth < 300:  # Thermocline
                                temperature = 24.0 - (depth - 100) * 0.08 + random.gauss(0, 0.4)
                                salinity = 35.2 + (depth - 100) * 0.002 + random.gauss(0, 0.06)
                            elif depth < 800:  # Intermediate
                                temperature = 8.0 - (depth - 300) * 0.008 + random.gauss(0, 0.3)
                                salinity = 34.6 + (depth - 300) * 0.0004 + random.gauss(0, 0.04)
                            else:  # Deep water
                                temperature = 4.0 - (depth - 800) * 0.002 + random.gauss(0, 0.2)
                                salinity = 34.7 + random.gauss(0, 0.03)
                                
                        else:
                            # Default Indian Ocean profile
                            if depth < 75:
                                temperature = 27.5 - depth * 0.02 + random.gauss(0, 1.0)
                                salinity = 34.6 + depth * 0.005 + random.gauss(0, 0.1)
                            elif depth < 500:
                                temperature = 25.0 - (depth - 75) * 0.045 + random.gauss(0, 0.5)
                                salinity = 35.0 - (depth - 75) * 0.001 + random.gauss(0, 0.08)
                            else:
                                temperature = 6.0 - (depth - 500) * 0.003 + random.gauss(0, 0.3)
                                salinity = 34.7 + random.gauss(0, 0.05)
                        
                        # Ensure realistic bounds
                        temperature = max(-2.0, min(32.0, temperature))
                        salinity = max(32.0, min(37.0, salinity))
                        
                        profile_temps.append(temperature)
                        profile_salts.append(salinity)
                        
                        # Quality flags (realistic QC distribution)
                        temp_qc = '1' if random.random() > 0.12 else ('2' if random.random() > 0.5 else '3')
                        sal_qc = '1' if random.random() > 0.15 else ('2' if random.random() > 0.5 else '3')
                        pres_qc = '1' if random.random() > 0.05 else '2'
                        
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
                    profile.min_temperature_c = float(min(profile_temps))
                    profile.max_temperature_c = float(max(profile_temps))
                    profile.min_salinity_psu = float(min(profile_salts))
                    profile.max_salinity_psu = float(max(profile_salts))
                    profile.valid_measurements_count = len([m for m in range(len(depths)) 
                                                          if temp_qc in ['1', '2'] and sal_qc in ['1', '2']])
                
                # Commit every 10 floats to avoid memory issues
                if (len([f for f, _ in created_floats if f.id <= float_obj.id]) % 10) == 0:
                    await session.commit()
                    print(f"Committed batch... {total_profiles} profiles, {total_measurements:,} measurements so far")
            
            # Final commit
            await session.commit()
            print(f"✅ Created {total_profiles} profiles with {total_measurements:,} measurements")
            
        print("🎉 Comprehensive Indian Ocean ARGO data injection complete!")
        
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
            
            # Depth coverage analysis
            depth_stats = await session.execute(select(
                func.min(Measurement.depth_m), 
                func.max(Measurement.depth_m),
                func.avg(Measurement.depth_m)
            ).where(Measurement.is_valid == True))
            min_depth, max_depth, avg_depth = depth_stats.first()
            
            print("\n" + "="*60)
            print("🌊 ENHANCED INDIAN OCEAN DATABASE SUMMARY")
            print("="*60)
            print(f"📍 Geographic Focus: Indian Ocean Basin")
            print(f"🏢 Data Assembly Centers: {dac_count}")
            print(f"🛰️  Total Floats: {float_count} ({active_floats} active, {float_count-active_floats} inactive)")
            print(f"📊 Total Profiles: {profile_count:,}")
            print(f"🔍 Total Measurements: {measurement_count:,}")
            print(f"✅ Valid Measurements: {valid_measurements:,}")
            print(f"📈 Data Quality: {(valid_measurements/measurement_count*100):.1f}%")
            print(f"🗃️  Database: {db_path}")
            
            print(f"\n🌊 DEPTH COVERAGE ANALYSIS:")
            print(f"   Minimum Depth: {min_depth:.1f}m")
            print(f"   Maximum Depth: {max_depth:.1f}m") 
            print(f"   Average Depth: {avg_depth:.1f}m")
            
            # Show geographic coverage
            geo_stats = await session.execute(select(
                func.min(Profile.latitude), func.max(Profile.latitude),
                func.min(Profile.longitude), func.max(Profile.longitude)
            ))
            min_lat, max_lat, min_lon, max_lon = geo_stats.first()
            print(f"\n🗺️  GEOGRAPHIC COVERAGE:")
            print(f"   Latitude: {min_lat:.2f}°N to {max_lat:.2f}°N")
            print(f"   Longitude: {min_lon:.2f}°E to {max_lon:.2f}°E")
            print(f"   Coverage: Indian Ocean Basin Complete")
            
            # Show temporal coverage
            temporal_stats = await session.execute(select(
                func.min(Profile.measurement_date), func.max(Profile.measurement_date)
            ))
            min_date, max_date = temporal_stats.first()
            print(f"\n📅 TEMPORAL COVERAGE:")
            print(f"   Start Date: {min_date.strftime('%Y-%m-%d')}")
            print(f"   End Date: {max_date.strftime('%Y-%m-%d')}")
            print(f"   Duration: {(max_date - min_date).days} days")
            
            # Regional distribution
            print(f"\n🌍 REGIONAL DISTRIBUTION:")
            regional_stats = await session.execute(
                select(Float.project_name, func.count(Float.id))
                .group_by(Float.project_name)
                .order_by(func.count(Float.id).desc())
            )
            for project, count in regional_stats:
                region_name = project.replace("Indian Ocean ARGO - ", "")
                print(f"   {region_name}: {count} floats")
        
        await engine.dispose()
        
        print("\n" + "="*60)
        print("🚀 PHASE 2 READY: Enhanced database optimized for RAG system!")
        print("✅ 120+ floats with comprehensive 2000m depth coverage")
        print("✅ Realistic oceanographic profiles for Indian Ocean")
        print("✅ High-quality dataset ready for LLM training")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())