# Multi-stage Dockerfile for FloatChat
# Optimized for production deployment with minimal image size

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    pkg-config \
    libhdf5-dev \
    libnetcdf-dev \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim as production

# Set metadata labels
LABEL maintainer="floatchat@sih2025.dev" \
      org.label-schema.name="FloatChat" \
      org.label-schema.description="AI-Powered Conversational Interface for ARGO Ocean Data" \
      org.label-schema.version=$VERSION \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libhdf5-103 \
    libnetcdf19 \
    libgdal32 \
    libproj25 \
    libgeos-c1v5 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r floatchat && useradd -r -g floatchat floatchat

# Create application directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ src/
COPY migrations/ migrations/
COPY alembic.ini ./

# Create necessary directories
RUN mkdir -p data/argo data/cache data/exports logs \
    && chown -R floatchat:floatchat /app

# Switch to non-root user
USER floatchat

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "floatchat.api.main"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir ".[dev,performance]"

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true \
    API_RELOAD=true

# Create development user (with sudo access)
RUN apt-get update && apt-get install -y sudo \
    && adduser --disabled-password --gecos '' developer \
    && adduser developer sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Working directory
WORKDIR /app

# Development server command
CMD ["python", "-m", "floatchat.api.main"]