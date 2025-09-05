"""
Configuration management for FloatChat.

This module provides comprehensive configuration management using Pydantic settings
with environment variable support, validation, and type safety.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, PostgresDsn, RedisDsn
from pydantic.networks import AnyHttpUrl


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL Configuration
    postgres_server: str = Field(default="localhost", env="POSTGRES_SERVER")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="floatchat", env="POSTGRES_USER")
    postgres_password: str = Field(default="floatchat123", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="floatchat", env="POSTGRES_DB")
    postgres_schema: str = Field(default="public", env="POSTGRES_SCHEMA")
    
    # Connection Pool Settings
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def database_url_sync(self) -> str:
        """Generate synchronous PostgreSQL connection URL for migrations."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")
    
    # Connection Pool Settings
    redis_pool_size: int = Field(default=20, env="REDIS_POOL_SIZE")
    redis_socket_timeout: float = Field(default=5.0, env="REDIS_SOCKET_TIMEOUT")
    redis_socket_connect_timeout: float = Field(default=5.0, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        protocol = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class AISettings(BaseSettings):
    """AI and ML configuration settings."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-1106-preview", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    
    # Anthropic Configuration  
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=4000, env="ANTHROPIC_MAX_TOKENS")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-mpnet-base-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    
    # Vector Database Configuration
    faiss_index_type: str = Field(default="HNSW", env="FAISS_INDEX_TYPE")
    faiss_hnsw_m: int = Field(default=32, env="FAISS_HNSW_M") 
    faiss_hnsw_ef: int = Field(default=200, env="FAISS_HNSW_EF")
    
    # RAG Configuration
    rag_chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=20, env="RAG_TOP_K")
    rag_score_threshold: float = Field(default=0.7, env="RAG_SCORE_THRESHOLD")


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # Security
    secret_key: str = Field(default="floatchat-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Configuration
    backend_cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000", 
            "http://localhost:8501",
            "https://localhost:8501"
        ],
        env="BACKEND_CORS_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


class DataProcessingSettings(BaseSettings):
    """Data processing configuration settings."""
    
    # ARGO Data Configuration
    argo_data_path: Path = Field(default=Path("data/argo"), env="ARGO_DATA_PATH")
    argo_ftp_server: str = Field(default="ftp.ifremer.fr", env="ARGO_FTP_SERVER")
    argo_ftp_path: str = Field(default="/ifremer/argo", env="ARGO_FTP_PATH")
    
    # Processing Configuration
    max_concurrent_files: int = Field(default=10, env="MAX_CONCURRENT_FILES")
    chunk_size: int = Field(default=1000, env="NETCDF_CHUNK_SIZE")
    memory_limit_gb: float = Field(default=4.0, env="MEMORY_LIMIT_GB")
    
    # Data Quality Configuration
    quality_flag_threshold: int = Field(default=3, env="QUALITY_FLAG_THRESHOLD")
    missing_data_threshold: float = Field(default=0.8, env="MISSING_DATA_THRESHOLD")
    
    # Export Configuration
    export_formats: List[str] = Field(
        default=["parquet", "csv", "netcdf"],
        env="EXPORT_FORMATS"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration settings."""
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Metrics Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Sentry Configuration
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")
    
    # Health Check Configuration
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Application Metadata
    app_name: str = Field(default="FloatChat", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(
        default="AI-Powered Conversational Interface for ARGO Ocean Data Discovery",
        env="APP_DESCRIPTION"
    )
    
    # Component Settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    ai: AISettings = AISettings()
    api: APISettings = APISettings()
    data: DataProcessingSettings = DataProcessingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        """Pydantic configuration."""
        
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed_environments = ["development", "testing", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.testing or self.environment == "testing"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()