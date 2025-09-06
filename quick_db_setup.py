#!/usr/bin/env python3
"""
Quick database setup for FloatChat - inject Indian Ocean ARGO data.
"""

import asyncio
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for in-memory database
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///floatchat_test.db"

async def main():
    """Main setup function."""
    print("Quick FloatChat database setup for Indian Ocean ARGO data...")
    
    try:
        # Import after setting env vars
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from floatchat.domain.entities.argo_entities import Base, DAC, Float, Profile, Measurement
        
        # Create SQLite database
        engine = create_async_engine("sqlite+aiosqlite:///floatchat_indian_ocean.db", echo=False)
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("Database schema created!")
        
        # Create session factory
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Insert some sample data for Indian Ocean
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
            
            # Create a sample float in Indian Ocean
            sample_float = Float(
                platform_number="2900226",
                dac_id=indian_dac.id,
                deployment_latitude=15.0,   # Indian Ocean coordinates
                deployment_longitude=75.0,  # Indian Ocean coordinates  
                project_name="Indian Ocean ARGO Network",
                pi_name="Dr. Indian Ocean Researcher",
                is_active=True
            )
            session.add(sample_float)
            await session.flush()
            
            # Create sample profiles
            from datetime import datetime
            import numpy as np
            
            for cycle in range(1, 6):  # Create 5 sample profiles
                profile = Profile(
                    float_id=sample_float.id,
                    cycle_number=cycle,
                    latitude=15.0 + cycle * 0.1,  # Slight drift
                    longitude=75.0 + cycle * 0.1,
                    measurement_date=datetime(2024, cycle, 15),
                    data_mode='D',  # Delayed mode
                    position_qc='1',
                    profile_temp_qc='1',
                    profile_psal_qc='1',
                    profile_pres_qc='1'
                )
                session.add(profile)
                await session.flush()
                
                # Add measurements for each profile
                for depth_idx in range(50):  # 50 depth levels
                    depth = depth_idx * 10.0  # Every 10 meters
                    pressure = depth * 1.02   # Approximate pressure
                    
                    # Simulate typical Indian Ocean T-S profile
                    temperature = 28.0 - (depth / 1000.0) * 20  # Decrease with depth
                    salinity = 34.5 + np.sin(depth / 200) * 0.5  # Slight variation
                    
                    measurement = Measurement(
                        profile_id=profile.id,
                        depth_level=depth_idx,
                        depth_m=depth,
                        pressure_db=pressure,
                        temperature_c=temperature,
                        salinity_psu=salinity,
                        temperature_qc='1',
                        salinity_qc='1', 
                        pressure_qc='1',
                        is_valid=True
                    )
                    session.add(measurement)
            
            await session.commit()
            
        print("🎯 Indian Ocean ARGO data injected successfully!")
        
        # Show summary
        async with session_factory() as session:
            from sqlalchemy import func, select
            
            float_count = await session.scalar(select(func.count(Float.id)))
            profile_count = await session.scalar(select(func.count(Profile.id))) 
            measurement_count = await session.scalar(select(func.count(Measurement.id)))
            
            print(f"📊 Database Summary:")
            print(f"   Floats: {float_count}")
            print(f"   Profiles: {profile_count}")
            print(f"   Measurements: {measurement_count}")
            print(f"💾 Database file: floatchat_indian_ocean.db")
        
        await engine.dispose()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())