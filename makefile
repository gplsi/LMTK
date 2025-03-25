# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run -v /raid/gplsi/ernesto/continual-pretraining-framework:/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only

.PHONY: help build validate tokenize train dev-shell clean run

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build      - Build Docker image"
	@echo   "container  - Start interactive development shell"
	@echo   "run        - Run a specific experiment"
	@echo   "            Usage: make run CONFIG=config/experiments/your_config.yaml"
	@echo   "clean      - Clean build artifacts"
	@echo   ""
	@echo   "Note: Once inside the container, use 'setup-wandb YOUR_API_KEY' to configure WandB"

build:
	docker build -t $(PROJECT_NAME) --network=host -f docker/Dockerfile .
	
container:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_framework -it --network=host --gpus '"device=3,4,0"' $(PROJECT_NAME) bash

run:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Please provide CONFIG. Usage: make run CONFIG=path/to/config.yaml"; \
		exit 1; \
	fi
	$(DOCKER_RUN) \
		-e PYTHONPATH=/workspace/src:$$PYTHONPATH \
		--network=host --gpus '"device=3,4,0"' $(PROJECT_NAME) \
		"python src/main.py $(CONFIG)"

clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
