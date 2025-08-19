# Copyright (C) 2025 Paul Chambaz
# This file is part of isir-internship

# Run the main program
run:
  @just train --env mountaincar --algo afutqc

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

compute-tqc-figure2 *ARGS:
  @uv run python -m tools.compute_tqc_figure2 {{ ARGS }}

compute-tqc-monte-carlo *ARGS:
  @uv run python -m tools.compute_tqc_monte_carlo {{ ARGS }}

visualize-tqc-figure *ARGS:
  @uv run python -m tools.visualize_tqc_figure {{ ARGS }}

visualize-figure-critics *ARGS:
  @uv run python -m tools.visualize_figure_critics {{ ARGS }}

visualize-bias *ARGS:
  @uv run python -m tools.visualize_bias {{ ARGS }}

visualize-bias-parameter *ARGS:
  @uv run python -m tools.visualize_bias_parameter {{ ARGS }}

organize ENV ALGO:
  @mv ~/down/history.pk ./outputs/figure/{{ ENV }}/{{ ALGO }}_history.pk
  @uv run python -m tools.merge --input ~/down/agent_history_0.pk ~/down/agent_history_1.pk --output ./outputs/figure/{{ ENV }}/{{ ALGO }}_agent.pt
  @ls -lh outputs/figure/{{ ENV }}

training ENV ALGO *ARGS:
  just train --env {{ ENV }} --algo {{ ALGO }} {{ ARGS }}
  mkdir -p ./outputs/{{ ENV }}/{{ ALGO }}/$(echo {{ ARGS }} | sed 's/--//g' | sed 's/ //g')
  mv ./outputs/agent_history_*.pk  ./outputs/history.pk ./outputs/{{ ENV }}/{{ ALGO }}/$(echo {{ ARGS }} | sed 's/--//g' | sed 's/ //g')

train-mountaincar:
  just training-msac mountaincar
  just training-sac mountaincar
  just training-ttqc mountaincar
  just training-tqc mountaincar
  just training-top mountaincar
  just training-ndtop mountaincar
  just training-afu mountaincar
  just training-tafu mountaincar

train-pendulum:
  just training-msac pendulum
  just training-sac pendulum
  just training-ttqc pendulum
  just training-tqc pendulum
  just training-top pendulum
  just training-ndtop pendulum
  just training-afu pendulum
  just training-tafu pendulum

train-lunarlander:
  just training-msac lunarlander
  just training-sac lunarlander
  just training-ttqc lunarlander
  just training-tqc lunarlander
  just training-top lunarlander
  just training-ndtop lunarlander
  just training-afu lunarlander
  just training-tafu lunarlander

train-swimmer:
  just training-msac swimmer
  just training-sac swimmer
  just training-ttqc swimmer
  just training-tqc swimmer
  just training-top swimmer
  just training-ndtop swimmer
  just training-afu swimmer
  just training-tafu swimmer

training-msac ENV:
  @just training {{ ENV }} msac --n 1
  @just training {{ ENV }} msac --n 3
  @just training {{ ENV }} msac --n 5
  @just training {{ ENV }} msac --n 10

training-sac ENV:
  @just training {{ ENV }} sac --n 2
  @just training {{ ENV }} sac --n 3
  @just training {{ ENV }} sac --n 5
  @just training {{ ENV }} sac --n 8

training-ttqc ENV:
  @just training {{ ENV }} tqc --n 1 --m 25 --d 1
  @just training {{ ENV }} tqc --n 1 --m 25 --d 2
  @just training {{ ENV }} tqc --n 1 --m 25 --d 3
  @just training {{ ENV }} tqc --n 1 --m 25 --d 5

training-tqc ENV:
  @just training {{ ENV }} tqc --n 2 --m 25 --d 1
  @just training {{ ENV }} tqc --n 2 --m 25 --d 2
  @just training {{ ENV }} tqc --n 2 --m 25 --d 3
  @just training {{ ENV }} tqc --n 2 --m 25 --d 5

training-top ENV:
  @just training {{ ENV }} top --n 2 --m 25 --b -1.0
  @just training {{ ENV }} top --n 2 --m 25 --b -0.5
  @just training {{ ENV }} top --n 2 --m 25 --b 0.0
  @just training {{ ENV }} top --n 2 --m 25 --b 0.5

training-ndtop ENV:
  @just training {{ ENV }} ndtop --n 2 --b -1.0
  @just training {{ ENV }} ndtop --n 2 --b -0.5
  @just training {{ ENV }} ndtop --n 2 --b 0.0
  @just training {{ ENV }} ndtop --n 2 --b 0.5

training-afu ENV:
  @just training {{ ENV }} afu --n 2 --r 0.2
  @just training {{ ENV }} afu --n 2 --r 0.4
  @just training {{ ENV }} afu --n 2 --r 0.6
  @just training {{ ENV }} afu --n 2 --r 0.8

training-tafu ENV:
  @just training {{ ENV }} afu --n 1 --r 0.2
  @just training {{ ENV }} afu --n 1 --r 0.4
  @just training {{ ENV }} afu --n 1 --r 0.6
  @just training {{ ENV }} afu --n 1 --r 0.8

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


