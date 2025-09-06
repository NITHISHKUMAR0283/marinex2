#!/usr/bin/env python3
"""
Setup script to create database and inject ARGO data.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from floatchat.infrastructure.database.service import db_service
from floatchat.data.services.ingestion_service import ingestion_service

async def main():
    """Main setup function."""
    print("🌊 Setting up FloatChat database and injecting ARGO data...")
    
    try:
        # Initialize database with schema
        print("📊 Creating database schema...")
        await db_service.initialize_database()
        print("✅ Database schema created successfully!")
        
        # Check if we have sample files
        sample_path = Path("data/argo/samples")
        if sample_path.exists() and any(sample_path.glob("*.nc")):
            print(f"📁 Found sample NetCDF files in {sample_path}")
            
            # Process sample files
            print("🔄 Processing sample ARGO data...")
            job_id = await ingestion_service.start_ingestion(
                source_path=sample_path,
                file_pattern="*.nc",
                recursive=False,
                max_concurrent=3
            )
            
            print(f"✅ Started ingestion job: {job_id}")
            
            # Monitor progress
            print("⏳ Monitoring ingestion progress...")
            while True:
                await asyncio.sleep(2)
                status = await ingestion_service.get_job_status(job_id)
                
                if not status:
                    break
                
                if status["status"] in ["completed", "completed_with_errors", "failed", "cancelled"]:
                    print(f"🎉 Ingestion {status['status']}!")
                    
                    if "progress" in status and status["progress"]:
                        prog = status["progress"]
                        print(f"📊 Final Stats: {prog['processed_files']} processed, {prog['failed_files']} failed")
                    
                    # Show database statistics
                    print("📈 Database statistics:")
                    stats = await ingestion_service.get_ingestion_statistics()
                    if "error" not in stats:
                        print(f"   Floats: {stats.get('total_floats', 0)}")
                        print(f"   Profiles: {stats.get('total_profiles', 0)}")  
                        print(f"   Measurements: {stats.get('total_measurements', 0)}")
                    
                    break
                
                if status.get("progress"):
                    prog = status["progress"]
                    print(f"   Progress: {prog['completion_percentage']:.1f}% ({prog['processed_files']}/{prog['total_files']})")
        
        else:
            print("⚠️ No sample NetCDF files found in data/argo/samples/")
            print("   You can download ARGO data and place .nc files there, then run:")
            print("   python setup_database.py")
        
        print("🎯 Database setup complete! FloatChat is ready.")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up
        await db_service.close()

if __name__ == "__main__":
    asyncio.run(main())