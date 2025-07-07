# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship

# Run the main program
run:
  @just train --env mountaincar --algo tqc

# Run main training script
train *ARGS:
  @uv run python -m tools.train {{ ARGS }}

# Run training curve visualization script
visualize-history *ARGS:
  @uv run python -m tools.visualize_history {{ ARGS }}

# Run critic and policy visualization script for mountaincar
visualize-critic-policy *ARGS:
  @uv run python -m tools.visualize_critic_policy {{ ARGS }}

# Run action-critic visualization script for mountaincar
visualize-action-critic *ARGS:
  @uv run python -m tools.visualize_action_critic {{ ARGS }}

# Compute monte carlo for each step of the learning history of an agent
compute-monte-carlo *ARGS:
  @uv run python -m tools.compute_monte_carlo {{ ARGS }}


compute-tqc-figure *ARGS:
  @uv run python -m tools.compute_tqc_figure {{ ARGS }}

visualize-tqc-figure *ARGS:
  @uv run python -m tools.visualize_tqc_figure {{ ARGS }}

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

# Used to make gif out of images
makegif:
  find . -maxdepth 1 ! -name "scaled_*" -regex '^\./[0-9]+\.png$' -exec sh -c 'magick "$1" -resize 1280x720 "scaled_$(basename "$1")"' _ {} \; && magick -limit memory 16GB -limit map 32GB scaled_*.png output.gif


