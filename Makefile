# FloatChat Development Makefile
# Production-ready commands for Smart India Hackathon 2025

.PHONY: help install dev-setup test lint format type-check security clean docker-build docker-run

# Default target
help:
	@echo "FloatChat Development Commands"
	@echo "============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install       Install all dependencies"
	@echo "  dev-setup     Complete development environment setup"
	@echo "  pre-commit    Install pre-commit hooks"
	@echo ""
	@echo "Development Commands:"
	@echo "  run           Run the FastAPI development server"
	@echo "  run-dashboard Run the Streamlit dashboard"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-coverage Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint          Run all linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run type checking with mypy"
	@echo "  security      Run security checks with bandit"
	@echo "  pre-commit-run  Run pre-commit on all files"
	@echo ""
	@echo "Database Commands:"
	@echo "  db-upgrade    Run database migrations"
	@echo "  db-downgrade  Rollback last database migration"
	@echo "  db-reset      Reset database (development only)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-run    Run application with Docker Compose"
	@echo "  docker-stop   Stop Docker services"
	@echo "  docker-logs   View Docker logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean         Clean build artifacts and cache"
	@echo "  deps-update   Update dependencies"
	@echo "  docs          Build documentation"

# Setup Commands
install:
	python -m pip install --upgrade pip
	pip install -e ".[dev,performance]"

dev-setup: install pre-commit
	@echo "Setting up development environment..."
	mkdir -p data/argo data/cache data/exports logs
	cp .env.example .env 2>/dev/null || echo "# FloatChat Environment Variables" > .env
	@echo "Development environment setup complete!"
	@echo "Please configure your .env file with appropriate values."

pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Development Commands
run:
	python -m floatchat.api.main

run-dashboard:
	streamlit run src/floatchat/presentation/dashboard.py --server.port 8501

test:
	pytest

test-unit:
	pytest tests/unit

test-integration:
	pytest tests/integration

test-performance:
	pytest tests/performance

test-coverage:
	pytest --cov=floatchat --cov-report=html --cov-report=term-missing

test-watch:
	pytest-watch

# Code Quality Commands
lint:
	@echo "Running flake8..."
	flake8 src tests
	@echo "Running mypy..."
	mypy src/floatchat
	@echo "Running bandit..."
	bandit -r src/floatchat
	@echo "All linting checks passed!"

format:
	@echo "Formatting with black..."
	black src tests
	@echo "Sorting imports with isort..."
	isort src tests
	@echo "Code formatting complete!"

type-check:
	mypy src/floatchat

security:
	bandit -r src/floatchat
	safety check

pre-commit-run:
	pre-commit run --all-files

# Database Commands
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	@echo "WARNING: This will destroy all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	alembic downgrade base
	alembic upgrade head

db-revision:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

# Docker Commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# Data Commands
data-download:
	python -m floatchat.data.downloaders.argo_downloader --limit 100

data-process:
	python -m floatchat.data.processors.netcdf_processor --input data/argo --output data/processed

data-index:
	python -m floatchat.infrastructure.vector_db.indexer --rebuild

# Utility Commands
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	@echo "Clean complete!"

deps-update:
	pip install --upgrade pip
	pip-compile pyproject.toml --upgrade
	pip install -e ".[dev,performance]"

docs:
	sphinx-build -b html docs docs/_build/html

docs-serve:
	python -m http.server 8080 --directory docs/_build/html

# Performance Commands
benchmark:
	python -m floatchat.performance.benchmarks

load-test:
	locust -f tests/performance/locustfile.py --web-host 0.0.0.0 --web-port 8089

profile:
	python -m cProfile -o profile.stats -m floatchat.api.main
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Monitoring Commands
metrics:
	@echo "Application metrics available at: http://localhost:8000/metrics"
	@echo "Health check available at: http://localhost:8000/health"

logs:
	tail -f logs/floatchat.log

# CI/CD Commands
ci-test:
	pytest --cov=floatchat --cov-report=xml --junit-xml=test-results.xml

ci-lint:
	flake8 --format=junit-xml --output-file=lint-results.xml src tests || true
	mypy --junit-xml mypy-results.xml src/floatchat || true

ci-security:
	bandit -r src/floatchat -f json -o security-results.json || true
	safety check --json --output security-deps.json || true

# Release Commands
version:
	@python -c "from floatchat import __version__; print(__version__)"

build:
	python -m build

release: clean build
	@echo "Built package ready for release"
	@echo "Version: $$(python -c 'from floatchat import __version__; print(__version__)')"

# Development Helpers
shell:
	python -i -c "from floatchat.core.config import settings; from floatchat.core.logging import get_logger; logger = get_logger('shell')"

jupyter:
	jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

# Environment specific commands
dev: dev-setup run

prod-check:
	@echo "Production readiness checklist:"
	@echo "- Environment variables configured: $$(test -f .env && echo "✓" || echo "✗")"
	@echo "- Dependencies installed: $$(pip show floatchat >/dev/null 2>&1 && echo "✓" || echo "✗")"
	@echo "- Tests passing: $$(pytest --tb=no -q && echo "✓" || echo "✗")"
	@echo "- Security checks: $$(bandit -r src/floatchat -q && echo "✓" || echo "✗")"
	@echo "- Type checks: $$(mypy src/floatchat --no-error-summary && echo "✓" || echo "✗")"