# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship

# Run the main program
run:
  @uv run python -m tools.main --env mountaincar --steps 200000

# Show available commands
help:
  @just --list

# Run all checks
check: lint typecheck

# Check and format code
lint:
  @ruff format src tests
  @ruff check --fix src tests

# Run type checking
typecheck:
  @ty check src tests

# Run tests
test:
  echo
  # @uv run pytest

# Run tests with HTML coverage report
test-cov:
  @uv run pytest --cov-report=html

# Sync dependencies
sync:
  @uv sync --dev

# Remove generated files
clean:
  @rm -rf .ruff_cache .pytest_cache htmlcov .coverage .ty_cache
  @find . -name "*.pyc" -delete
  @find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Reset virtual environment
reset: clean
  @rm -rf .venv
  @uv sync --dev
