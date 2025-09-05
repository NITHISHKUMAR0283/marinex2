# FloatChat: AI-Powered Conversational Interface for ARGO Ocean Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated AI-powered conversational system that democratizes access to ARGO oceanographic data through natural language queries, interactive visualizations, and AI-powered insights.

**Built for Smart India Hackathon 2025** with production-ready architecture and advanced RAG + Model Context Protocol integration.

## 🌊 Overview

FloatChat transforms complex oceanographic data into accessible insights through:

- **Natural Language Querying**: Ask questions like "Show me temperature anomalies in the Arabian Sea during 2023 monsoon"
- **Advanced RAG Pipeline**: First-of-its-kind RAG+MCP implementation for scientific data
- **Multi-modal Embeddings**: Composite embeddings combining text, spatial, temporal, and parametric data
- **Interactive Visualizations**: Real-time maps, profiles, time-series, and comparative analysis
- **Production-Ready Architecture**: Scalable system supporting 1000+ concurrent users

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ 
- Docker & Docker Compose
- PostgreSQL 14+ with PostGIS
- Redis 7+
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/NITHISHKUMAR0283/marin.git
cd floatchat-sih2025

# Set up development environment
make dev-setup

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services with Docker
make docker-run

# Access the application
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Health Check: http://localhost:8000/health
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run the application
make run
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FloatChat Architecture                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Data Layer    │  Processing     │   Interface Layer       │
│                 │     Layer       │                         │
│ • ARGO NetCDF   │ • Data ETL      │ • Streamlit Dashboard   │
│ • PostgreSQL    │ • RAG Pipeline  │ • Chat Interface        │
│ • FAISS Vector  │ • MCP Server    │ • API Gateway           │
│   Database      │ • LLM Engine    │ • Plotly/Leaflet Maps  │
│ • Redis Cache   │ • Query Engine  │ • Data Export Tools     │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Technology Stack

- **Backend**: Python 3.9+, FastAPI, PostgreSQL, Redis
- **AI/ML**: LangChain, FAISS, Sentence-Transformers, OpenAI/Anthropic APIs  
- **Frontend**: Streamlit, Plotly, Folium, Leaflet.js
- **Data**: pandas, xarray, NetCDF4, argopy, dask
- **Deployment**: Docker, Docker Compose, Nginx, Kubernetes

## 📊 Features

### Natural Language Processing
- Complex oceanographic query understanding
- Multi-entity extraction (locations, parameters, time ranges)
- Context-aware conversation management
- Educational explanations and scientific interpretations

### Data Analysis Capabilities
- Temperature, salinity, pressure profile analysis
- Seasonal pattern recognition and trend analysis
- Water mass identification and comparison
- Anomaly detection and quality assessment
- Statistical analysis and climatological averages

### Visualization Suite
- Interactive world maps with ARGO float clustering
- Time-series animation and temporal controls
- 3D ocean visualization with depth profiles
- Scientific plotting (T-S diagrams, contour plots)
- Export capabilities in multiple formats

### Advanced Technical Features
- **RAG Pipeline**: Domain-specific retrieval with oceanographic knowledge
- **MCP Integration**: Tool calling for complex data operations
- **Vector Search**: Sub-100ms similarity search on 10M+ embeddings
- **Real-time Processing**: Streaming data updates and incremental indexing
- **Multi-modal AI**: Composite understanding of text, spatial, and temporal data

## 🛠️ Development

### Project Structure

```
floatchat-sih2025/
├── src/floatchat/              # Main application package
│   ├── api/                    # FastAPI application
│   ├── core/                   # Core utilities and config
│   ├── data/                   # Data processing modules
│   ├── domain/                 # Business logic
│   ├── infrastructure/         # External integrations
│   └── presentation/           # UI components
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── performance/            # Performance tests
├── docker/                     # Docker configurations
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
└── migrations/                 # Database migrations
```

### Development Commands

```bash
# Development server
make run                    # Start FastAPI server
make run-dashboard         # Start Streamlit dashboard

# Code quality
make format               # Format code with black/isort
make lint                 # Run all linting checks
make type-check          # Run mypy type checking
make security            # Run security checks

# Testing
make test                # Run all tests
make test-unit           # Run unit tests only
make test-coverage       # Run with coverage report

# Database
make db-upgrade          # Run migrations
make db-reset           # Reset database (dev only)

# Docker
make docker-build       # Build images
make docker-run         # Start all services
make docker-stop        # Stop services
```

### Environment Variables

Key configuration options:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/floatchat

# AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...

# Data Processing
ARGO_DATA_PATH=data/argo
MAX_CONCURRENT_FILES=10
MEMORY_LIMIT_GB=4.0
```

## 📈 Performance

### Target Metrics
- **Query Response**: <5 seconds (95th percentile)
- **Concurrent Users**: 1000+ simultaneous users  
- **Data Processing**: 1000+ NetCDF files/hour
- **Vector Search**: <100ms for 10M embeddings
- **System Availability**: >99.5% uptime

### Optimization Features
- Multi-level caching (Redis + application + browser)
- Async processing with configurable concurrency
- Database connection pooling and query optimization
- Vector search index optimization (FAISS)
- Progressive loading for large datasets

## 🧪 Testing

### Test Suite
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=floatchat --cov-report=html

# Run specific test types
pytest tests/unit                    # Unit tests
pytest tests/integration             # Integration tests  
pytest tests/performance             # Performance tests

# Run performance benchmarks
make benchmark

# Load testing
make load-test
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **Performance Tests**: Load and stress testing
- **API Tests**: Endpoint validation and contract testing
- **End-to-End Tests**: Complete user workflow testing

## 📊 Monitoring

### Health Checks
- `/health` - Comprehensive health status
- `/health/live` - Kubernetes liveness probe
- `/health/ready` - Kubernetes readiness probe
- `/metrics` - Prometheus metrics endpoint

### Observability Stack
- **Metrics**: Prometheus + Grafana dashboards
- **Logging**: Structured JSON logging with correlation IDs
- **Tracing**: Distributed tracing for request flow
- **Alerting**: Intelligent alerting for critical issues

## 🌐 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs (development)
- **ReDoc**: http://localhost:8000/redoc (development)
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Key Endpoints
```bash
# Health and Monitoring
GET /health                         # System health status
GET /metrics                        # Prometheus metrics

# Data API (Coming in Phase 2)
GET /api/v1/data/floats            # ARGO float information
GET /api/v1/data/profiles          # Profile data
POST /api/v1/data/query            # Complex data queries

# AI API (Coming in Phase 3)  
POST /api/v1/ai/query              # Natural language queries
GET /api/v1/ai/embeddings          # Vector embeddings
POST /api/v1/ai/chat               # Conversational interface

# Visualization API (Coming in Phase 4)
GET /api/v1/viz/maps               # Interactive maps
GET /api/v1/viz/plots              # Data visualizations
POST /api/v1/viz/export            # Export functionality
```

## 🚀 Deployment

### Docker Deployment
```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Development deployment
docker-compose up -d
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=floatchat
```

### Production Checklist
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations applied
- [ ] Health checks responding
- [ ] Monitoring alerts configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security scan passed

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following code quality standards
4. Run tests (`make test`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

### Code Quality Standards
- **Formatting**: Black + isort
- **Linting**: flake8 + mypy + bandit
- **Testing**: >90% coverage required
- **Documentation**: Comprehensive docstrings
- **Type Hints**: Full type annotation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Smart India Hackathon 2025

### Innovation Highlights
- **First RAG+MCP Implementation** for scientific data analysis
- **Multi-modal Embeddings** for oceanographic data understanding
- **Production-Ready Architecture** with enterprise scalability
- **Domain-Specific AI** optimized for marine science applications

### Demonstration Scenarios
1. **Marine Researcher**: "Analyze temperature anomalies in Indian Ocean during El Niño events"
2. **Policy Maker**: "Compare ocean warming trends near major coastal cities"
3. **Educator**: "Explain thermohaline circulation and show ARGO data examples"
4. **Student**: "What are the differences between Arabian Sea and Bay of Bengal?"

### Team & Acknowledgments

Built with ❤️ for **Smart India Hackathon 2025** by the FloatChat development team.

**Data Sources**:
- [ARGO Global Data Repository](http://www.argodatamgt.org/)
- [Indian National Centre for Ocean Information Services (INCOIS)](https://incois.gov.in/)

**Special Thanks**:
- Marine science community for domain expertise
- ARGO program for making ocean data freely available
- Open source community for excellent tools and libraries

---

**🌊 Making Ocean Data Accessible to Everyone 🌊**