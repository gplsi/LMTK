# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = lmtk
CONFIG_PATH = config
DOCKER_RUN = docker run -v ".":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only

# Get current user and group IDs for Docker user mapping
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)
USERNAME := $(shell whoami)

.PHONY: help build validate tokenize train dev-shell clean

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build       - Build Docker image"
	@echo   "container   - Start interactive development shell"
	@echo   "clean       - Clean build artifacts"

build:
	@echo "Building Docker image with user mapping: USER_ID=$(USER_ID), GROUP_ID=$(GROUP_ID), USERNAME=$(USERNAME)"
	docker build \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg USERNAME=$(USERNAME) \
		-t $(PROJECT_NAME) \
		--network=host \
		-f docker/Dockerfile .
	
container:
	@echo "Starting container with user: $(USERNAME) (UID: $(USER_ID), GID: $(GROUP_ID))"
	$(DOCKER_RUN) \
		--name gplsi_lmtk \
		-it \
		--network=host \
		--gpus all \
		--user "$(USER_ID):$(GROUP_ID)" \
		$(PROJECT_NAME) bash

clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
