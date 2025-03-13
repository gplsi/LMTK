# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run -v "$(CURDIR)":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only

.PHONY: help build validate tokenize train dev-shell clean test test-unit test-integration test-all test-grid

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build       - Build Docker image"
	@echo   "container   - Start interactive development shell"
	@echo   "clean       - Clean build artifacts"
	@echo   "test        - Run all tests"
	@echo   "test-unit   - Run unit tests"
	@echo   "test-integration - Run integration tests"
	@echo   "test-grid   - Run parameterized tests using a test grid"

build:
	docker build -t $(PROJECT_NAME) --network=host -f docker/Dockerfile .
	
container:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_framework -it --network=host --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) bash

clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
	-@rm -rf .pytest_cache 2> /dev/null
	-@rm -rf .coverage htmlcov 2> /dev/null

# Testing targets
test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit -v

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration -v

test-all:
	@echo "Running all tests with coverage report..."
	pytest --cov=src --cov-report=term-missing tests/

test-grid:
	@echo "Running parameterized tests with grid config..."
	@if [ -z "$(GRID)" ]; then \
		echo "Usage: make test-grid GRID=config/test_grids/minimal_test_grid.yaml"; \
		exit 1; \
	fi
	python scripts/run_parameterized_tests.py --test-type=$(TEST_TYPE) --param-file=$(GRID)

# Extra test targets for specific scenarios
test-minimal:
	@echo "Running minimal test grid..."
	python scripts/run_parameterized_tests.py --test-type=unit --param-file=config/test_grids/minimal_test_grid.yaml

test-comprehensive:
	@echo "Running comprehensive test grid (this may take a while)..."
	python scripts/run_parameterized_tests.py --test-type=all --param-file=config/test_grids/comprehensive_test_grid.yaml
