# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run --rm -v "$(CURDIR)":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only

.PHONY: help build validate tokenize train dev-shell clean test test-unit test-integration test-all test-grid test-gpu test-distributed-gpu test-perf

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build           - Build Docker image"
	@echo   "container       - Start interactive development shell"
	@echo   "clean           - Clean build artifacts"
	@echo   "test           - Run CPU-only tests"
	@echo   "test-unit      - Run unit tests (CPU only)"
	@echo   "test-integration - Run integration tests (CPU only)"
	@echo   "test-gpu       - Run tests requiring GPU"
	@echo   "test-distributed-gpu - Run distributed tests with real GPUs"
	@echo   "test-perf      - Run performance benchmark tests"

build:
	docker build -t $(PROJECT_NAME) --network=host -f docker/Dockerfile .
	
container:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_framework -it --network=host --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) bash

clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
	-@rm -rf .pytest_cache 2> /dev/null
	-@rm -rf .coverage htmlcov 2> /dev/null

# Testing targets - CPU only
test: test-unit test-integration

test-unit:
	@echo "Running unit tests (CPU only)..."
	$(DOCKER_RUN) $(PROJECT_NAME) pytest -v -m "not requires_gpu" tests/unit

test-integration:
	@echo "Running integration tests (CPU only)..."
	$(DOCKER_RUN) $(PROJECT_NAME) pytest -v -m "not requires_gpu" tests/integration

test-all:
	@echo "Running all CPU tests with coverage report..."
	$(DOCKER_RUN) $(PROJECT_NAME) pytest --cov=src --cov-report=term-missing -v -m "not requires_gpu" tests/

test-grid:
	@echo "Running parameterized tests with grid config..."
	@if [ -z "$(GRID)" ]; then \
		echo "Usage: make test-grid GRID=config/test_grids/minimal_test_grid.yaml"; \
		exit 1; \
	fi
	$(DOCKER_RUN) $(PROJECT_NAME) python scripts/run_parameterized_tests.py --test-type=$(TEST_TYPE) --param-file=$(GRID)

# GPU testing targets
test-gpu:
	@echo "Running GPU tests in Docker container..."
	$(DOCKER_RUN) --gpus all $(PROJECT_NAME) pytest -v -m "requires_gpu" tests/

test-distributed-gpu:
	@echo "Running distributed GPU tests..."
	$(DOCKER_RUN) --gpus all $(PROJECT_NAME) python scripts/run_parameterized_tests.py \
		--test-type=all \
		--param-file=config/test_grids/gpu_test_grid.yaml \
		--test-pattern="test_pretraining_with_fsdp or test_pretraining_with_ddp" \
		--gpu-only

# Performance testing target
test-perf:
	@echo "Running performance benchmark tests..."
	@if [ -z "$(GPUS)" ]; then \
		echo "Error: GPUS environment variable must be set (e.g., GPUS=2)"; \
		exit 1; \
	fi
	$(DOCKER_RUN) --gpus all \
		-e CUDA_VISIBLE_DEVICES=0,$$(seq -s, 1 $$(($(GPUS)-1))) \
		$(PROJECT_NAME) python scripts/run_parameterized_tests.py \
		--test-type=all \
		--param-file=config/test_grids/performance_benchmark.yaml \
		--gpu-only

# Extra test targets for specific scenarios
test-minimal:
	@echo "Running minimal test grid..."
	$(DOCKER_RUN) $(PROJECT_NAME) python scripts/run_parameterized_tests.py --test-type=unit --param-file=config/test_grids/minimal_test_grid.yaml

test-comprehensive:
	@echo "Running comprehensive test grid..."
	$(DOCKER_RUN) $(PROJECT_NAME) python scripts/run_parameterized_tests.py --test-type=all --param-file=config/test_grids/comprehensive_test_grid.yaml
