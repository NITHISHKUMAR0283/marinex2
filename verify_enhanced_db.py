#!/usr/bin/env python3
"""
Quick verification script for enhanced Indian Ocean database.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def verify_database():
    """Verify the enhanced database has proper data."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy import func, select
        from simple_entities import Base, DAC, Float, Profile, Measurement
        
        # Connect to enhanced database
        engine = create_async_engine("sqlite+aiosqlite:///floatchat_indian_ocean_enhanced.db", echo=False)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with session_factory() as session:
            # Basic counts
            float_count = await session.scalar(select(func.count(Float.id)))
            profile_count = await session.scalar(select(func.count(Profile.id)))
            measurement_count = await session.scalar(select(func.count(Measurement.id)))
            
            # Depth analysis
            max_depth = await session.scalar(select(func.max(Measurement.depth_m)))
            min_depth = await session.scalar(select(func.min(Measurement.depth_m)))
            
            # Regional distribution
            regions = await session.execute(select(Float.project_name, func.count(Float.id)).group_by(Float.project_name))
            
            print("🔍 DATABASE VERIFICATION RESULTS")
            print("="*40)
            print(f"✅ Floats: {float_count}")
            print(f"✅ Profiles: {profile_count:,}")  
            print(f"✅ Measurements: {measurement_count:,}")
            print(f"✅ Depth Range: {min_depth:.1f}m - {max_depth:.1f}m")
            
            print(f"\n🌍 Regional Coverage:")
            for project, count in regions:
                region = project.replace("Indian Ocean ARGO - ", "")
                print(f"   {region}: {count} floats")
            
            # Check if Phase 2 ready
            if float_count >= 100 and max_depth >= 1900 and measurement_count >= 500000:
                print(f"\n🚀 STATUS: PHASE 2 READY!")
                print(f"✅ Sufficient floats: {float_count} >= 100")
                print(f"✅ Full depth coverage: {max_depth:.0f}m >= 1900m")
                print(f"✅ Rich dataset: {measurement_count:,} measurements")
            else:
                print(f"\n⚠️  STATUS: NEEDS MORE DATA")
                print(f"{'❌' if float_count < 100 else '✅'} Floats: {float_count}/100")
                print(f"{'❌' if max_depth < 1900 else '✅'} Depth: {max_depth:.0f}m/1900m")
                print(f"{'❌' if measurement_count < 500000 else '✅'} Measurements: {measurement_count:,}/500,000")
                
        await engine.dispose()
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_database())