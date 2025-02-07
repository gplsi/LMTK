# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run --rm -v "$(CURDIR)":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only

.PHONY: help build validate tokenize train dev-shell clean

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build       - Build Docker image"
	@echo   "container   - Start interactive development shell"
	@echo   "clean       - Clean build artifacts"

build:
	docker build -t $(PROJECT_NAME) -f docker/Dockerfile .
	
container:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_framework -it --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) bash

clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
