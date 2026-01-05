# ML Module Makefile
# Provides standard commands for testing, linting, and development

.PHONY: test test-unit test-integration lint format typecheck clean help

# Default target
help:
	@echo "ML Module Commands:"
	@echo ""
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-verbose    Run all tests with verbose output"
	@echo "  make test-cov        Run tests with coverage report"
	@echo ""
	@echo "  make lint            Run linter (ruff)"
	@echo "  make format          Auto-format code (ruff)"
	@echo "  make typecheck       Run type checker (mypy)"
	@echo ""
	@echo "  make clean           Remove cache files"
	@echo "  make server          Start prediction server"
	@echo ""

# =============================================================================
# TESTING
# =============================================================================

test:
	python3 -m pytest tests/ -q

test-unit:
	python3 -m pytest tests/unit/ -q

test-integration:
	python3 -m pytest tests/integration/ -q

test-verbose:
	python3 -m pytest tests/ -v

test-cov:
	python3 -m pytest tests/ --cov=server --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-quick:
	python3 -m pytest tests/ -q -x --tb=short

# =============================================================================
# LINTING & FORMATTING
# =============================================================================

lint:
	@command -v ruff >/dev/null 2>&1 || pip3 install ruff --quiet
	ruff check server/ tests/

format:
	@command -v ruff >/dev/null 2>&1 || pip3 install ruff --quiet
	ruff format server/ tests/
	ruff check --fix server/ tests/

typecheck:
	@command -v mypy >/dev/null 2>&1 || pip3 install mypy --quiet
	mypy server/ --ignore-missing-imports

# =============================================================================
# DEVELOPMENT
# =============================================================================

server:
	POLYGON_API_KEY=$(POLYGON_API_KEY) python3 predict_server.py

server-dev:
	FLASK_ENV=development POLYGON_API_KEY=$(POLYGON_API_KEY) python3 predict_server.py

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

# =============================================================================
# CI PIPELINE (all checks)
# =============================================================================

ci: lint test
	@echo "✅ CI checks passed"

ci-full: lint typecheck test-cov
	@echo "✅ Full CI checks passed"
