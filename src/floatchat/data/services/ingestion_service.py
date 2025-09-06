"""
ARGO data ingestion service for high-performance NetCDF processing.

This service orchestrates the ingestion of ARGO oceanographic data
with progress tracking, error handling, and database optimization.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import aiofiles
import json

from floatchat.core.config import settings
from floatchat.core.logging import get_logger
from floatchat.data.processors.netcdf_processor import NetCDFProcessor, ProcessingStats
from floatchat.infrastructure.database.connection import get_session
from floatchat.infrastructure.repositories.argo_repository import ArgoRepository

logger = get_logger(__name__)


@dataclass
class IngestionJob:
    """Data ingestion job configuration."""
    job_id: str
    source_path: Path
    file_pattern: str = "*.nc"
    recursive: bool = True
    max_concurrent: int = 10
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    progress_callback: Optional[Callable] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class IngestionProgress:
    """Progress tracking for ingestion jobs."""
    job_id: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    current_file: str = ""
    estimated_completion: Optional[datetime] = None
    processing_rate_files_per_second: float = 0.0
    
    @property
    def completion_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def is_complete(self) -> bool:
        return self.processed_files + self.failed_files >= self.total_files


class IngestionService:
    """High-performance ARGO data ingestion service."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.active_jobs: Dict[str, IngestionJob] = {}
        self.job_progress: Dict[str, IngestionProgress] = {}
        self._job_counter = 0
    
    async def start_ingestion(
        self,
        source_path: Path,
        file_pattern: str = "*.nc",
        recursive: bool = True,
        max_concurrent: int = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Start a new data ingestion job.
        
        Args:
            source_path: Directory or file path to process
            file_pattern: File pattern to match
            recursive: Process subdirectories recursively
            max_concurrent: Maximum concurrent file processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Job ID for tracking progress
        """
        # Generate unique job ID
        self._job_counter += 1
        job_id = f"ingestion_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._job_counter}"
        
        # Create job configuration
        job = IngestionJob(
            job_id=job_id,
            source_path=source_path,
            file_pattern=file_pattern,
            recursive=recursive,
            max_concurrent=max_concurrent or settings.data.max_concurrent_files,
            progress_callback=progress_callback
        )
        
        self.active_jobs[job_id] = job
        
        # Initialize progress tracking
        self.job_progress[job_id] = IngestionProgress(job_id=job_id)
        
        # Start processing asynchronously
        asyncio.create_task(self._run_ingestion_job(job))
        
        self.logger.info(f"Started ingestion job {job_id} for {source_path}")
        return job_id
    
    async def _run_ingestion_job(self, job: IngestionJob) -> None:
        """Execute ingestion job with full error handling."""
        job.status = "running"
        job.started_at = datetime.utcnow()
        progress = self.job_progress[job.job_id]
        
        try:
            self.logger.info(f"Running ingestion job {job.job_id}")
            
            # Validate source path
            if not job.source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {job.source_path}")
            
            # Count total files for progress tracking
            if job.source_path.is_file():
                total_files = 1
            else:
                if job.recursive:
                    files = list(job.source_path.rglob(job.file_pattern))
                else:
                    files = list(job.source_path.glob(job.file_pattern))
                total_files = len(files)
            
            progress.total_files = total_files
            self.logger.info(f"Job {job.job_id}: Found {total_files} files to process")
            
            if total_files == 0:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                self.logger.warning(f"Job {job.job_id}: No files found matching pattern {job.file_pattern}")
                return
            
            # Create processor with job-specific concurrency
            processor = NetCDFProcessor(max_concurrent=job.max_concurrent)
            
            # Process files with progress tracking
            processing_stats = await self._process_with_progress_tracking(
                processor, job, progress
            )
            
            # Update job status
            job.status = "completed" if processing_stats.files_failed == 0 else "completed_with_errors"
            job.completed_at = datetime.utcnow()
            
            # Log final results
            self.logger.info(
                f"Job {job.job_id} completed: "
                f"{processing_stats.files_processed} processed, "
                f"{processing_stats.files_failed} failed, "
                f"{processing_stats.floats_created} floats created, "
                f"{processing_stats.profiles_created} profiles created, "
                f"{processing_stats.measurements_created} measurements created"
            )
            
            # Save job results
            await self._save_job_results(job, processing_stats)
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            self.logger.error(f"Job {job.job_id} failed: {e}")
            
            # Save error information
            await self._save_job_error(job, str(e))
        
        finally:
            # Clean up progress tracking after some time
            asyncio.create_task(self._cleanup_job_after_delay(job.job_id, delay_hours=24))
    
    async def _process_with_progress_tracking(
        self,
        processor: NetCDFProcessor,
        job: IngestionJob,
        progress: IngestionProgress
    ) -> ProcessingStats:
        """Process files with real-time progress tracking."""
        start_time = datetime.utcnow()
        
        # Create custom processor that reports progress
        class ProgressTrackingProcessor(NetCDFProcessor):
            def __init__(self, base_processor: NetCDFProcessor, progress_obj: IngestionProgress):
                super().__init__(base_processor.max_concurrent)
                self.progress = progress_obj
                self.processed_count = 0
                self.start_time = start_time
                
            async def _process_single_file(self, file_path: Path, stats: ProcessingStats) -> None:
                # Update current file
                self.progress.current_file = str(file_path)
                
                # Call parent processing
                await super()._process_single_file(file_path, stats)
                
                # Update progress
                self.processed_count += 1
                self.progress.processed_files = stats.files_processed
                self.progress.failed_files = stats.files_failed
                
                # Calculate processing rate
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                if elapsed > 0:
                    self.progress.processing_rate_files_per_second = self.processed_count / elapsed
                    
                    # Estimate completion time
                    remaining_files = self.progress.total_files - self.processed_count
                    if self.progress.processing_rate_files_per_second > 0:
                        remaining_seconds = remaining_files / self.progress.processing_rate_files_per_second
                        self.progress.estimated_completion = datetime.utcnow() + \
                            asyncio.get_event_loop().time() + remaining_seconds
                
                # Call progress callback if provided
                if job.progress_callback:
                    try:
                        if asyncio.iscoroutinefunction(job.progress_callback):
                            await job.progress_callback(self.progress)
                        else:
                            job.progress_callback(self.progress)
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed: {e}")
        
        # Use progress tracking processor
        tracking_processor = ProgressTrackingProcessor(processor, progress)
        
        # Process directory or single file
        if job.source_path.is_file():
            stats = ProcessingStats()
            await tracking_processor._process_single_file(job.source_path, stats)
            return stats
        else:
            return await tracking_processor.process_directory(
                job.source_path, job.file_pattern, job.recursive
            )
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of ingestion job."""
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        progress = self.job_progress.get(job_id)
        
        return {
            "job_id": job_id,
            "status": job.status,
            "source_path": str(job.source_path),
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "progress": {
                "total_files": progress.total_files if progress else 0,
                "processed_files": progress.processed_files if progress else 0,
                "failed_files": progress.failed_files if progress else 0,
                "completion_percentage": progress.completion_percentage if progress else 0,
                "current_file": progress.current_file if progress else "",
                "processing_rate": progress.processing_rate_files_per_second if progress else 0,
                "estimated_completion": progress.estimated_completion.isoformat() 
                    if progress and progress.estimated_completion else None
            } if progress else None
        }
    
    async def list_active_jobs(self) -> List[Dict]:
        """List all active ingestion jobs."""
        jobs = []
        for job_id in self.active_jobs:
            job_status = await self.get_job_status(job_id)
            if job_status:
                jobs.append(job_status)
        return jobs
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active ingestion job."""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        if job.status in ["completed", "failed"]:
            return False
        
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        
        self.logger.info(f"Cancelled ingestion job {job_id}")
        return True
    
    async def _save_job_results(self, job: IngestionJob, stats: ProcessingStats) -> None:
        """Save job results to file for auditing."""
        try:
            results_dir = settings.data.argo_data_path / "ingestion_logs"
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"{job.job_id}_results.json"
            
            results_data = {
                "job_id": job.job_id,
                "source_path": str(job.source_path),
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "processing_time_seconds": stats.processing_time_seconds,
                "statistics": {
                    "files_processed": stats.files_processed,
                    "files_failed": stats.files_failed,
                    "floats_created": stats.floats_created,
                    "profiles_created": stats.profiles_created,
                    "measurements_created": stats.measurements_created,
                    "success_rate": stats.success_rate,
                    "errors": stats.errors
                }
            }
            
            async with aiofiles.open(results_file, 'w') as f:
                await f.write(json.dumps(results_data, indent=2))
                
        except Exception as e:
            self.logger.error(f"Failed to save job results for {job.job_id}: {e}")
    
    async def _save_job_error(self, job: IngestionJob, error_message: str) -> None:
        """Save job error information."""
        try:
            results_dir = settings.data.argo_data_path / "ingestion_logs"
            results_dir.mkdir(exist_ok=True)
            
            error_file = results_dir / f"{job.job_id}_error.json"
            
            error_data = {
                "job_id": job.job_id,
                "source_path": str(job.source_path),
                "status": "failed",
                "error_message": error_message,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "failed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            async with aiofiles.open(error_file, 'w') as f:
                await f.write(json.dumps(error_data, indent=2))
                
        except Exception as e:
            self.logger.error(f"Failed to save job error for {job.job_id}: {e}")
    
    async def _cleanup_job_after_delay(self, job_id: str, delay_hours: int = 24) -> None:
        """Clean up job data after specified delay."""
        await asyncio.sleep(delay_hours * 3600)  # Convert hours to seconds
        
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        if job_id in self.job_progress:
            del self.job_progress[job_id]
            
        self.logger.debug(f"Cleaned up job data for {job_id}")
    
    async def get_ingestion_statistics(self) -> Dict:
        """Get overall ingestion statistics."""
        try:
            async with get_session() as session:
                repo = ArgoRepository(session)
                db_stats = await repo.get_database_statistics()
                
                # Add ingestion-specific statistics
                active_jobs_count = len([j for j in self.active_jobs.values() if j.status == "running"])
                completed_jobs_count = len([j for j in self.active_jobs.values() if j.status in ["completed", "completed_with_errors"]])
                
                return {
                    **db_stats,
                    "ingestion_jobs": {
                        "active": active_jobs_count,
                        "completed_recent": completed_jobs_count,
                        "total_tracked": len(self.active_jobs)
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get ingestion statistics: {e}")
            return {"error": str(e)}
    
    async def process_sample_files(self) -> str:
        """Process sample ARGO files for testing."""
        sample_path = settings.data.argo_data_path / "samples"
        
        if not sample_path.exists() or not any(sample_path.glob("*.nc")):
            raise FileNotFoundError(f"No sample files found in {sample_path}")
        
        return await self.start_ingestion(
            source_path=sample_path,
            file_pattern="*.nc",
            recursive=False,
            max_concurrent=3  # Limit for testing
        )


# Global service instance
ingestion_service = IngestionService()


# Convenience functions for common operations
async def ingest_argo_directory(directory_path: Path, **kwargs) -> str:
    """Convenience function to ingest ARGO files from directory."""
    return await ingestion_service.start_ingestion(directory_path, **kwargs)


async def get_ingestion_progress(job_id: str) -> Optional[Dict]:
    """Get progress of specific ingestion job."""
    return await ingestion_service.get_job_status(job_id)