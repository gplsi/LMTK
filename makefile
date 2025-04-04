# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run -v "$(CURDIR)":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only
POETRY = poetry
PYTHON = $(POETRY) run python

# Poetry configuration
POETRY_VENV = .venv
POETRY_INSTALL_OPTS ?= 

.PHONY: help build build-dev build-prod container validate tokenize train clean poetry-install poetry-update poetry-shell poetry-export poetry-test poetry-lint poetry-docs poetry-run

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build            - Build Docker image with Poetry (development)"
	@echo   "build-dev        - Build Docker image for development"
	@echo   "build-prod       - Build Docker image for production (smaller size)"
	@echo   "container        - Start interactive development shell"
	@echo   ""
	@echo   "Poetry commands:"
	@echo   "poetry-install   - Install dependencies using Poetry"
	@echo   "poetry-update    - Update dependencies to latest versions"
	@echo   "poetry-shell     - Activate Poetry virtual environment"
	@echo   "poetry-export    - Export dependencies to requirements.txt"
	@echo   "poetry-test      - Run tests using pytest through Poetry"
	@echo   "poetry-lint      - Run linting through Poetry"
	@echo   "poetry-run       - Run a command in the Poetry environment (usage: make poetry-run CMD='python src/main.py')"

# Docker build commands
build: build-dev

build-dev:
	docker build -t $(PROJECT_NAME):dev --network=host -f docker/Dockerfile .

build-prod:
	@echo "Building production image (multi-stage build)..."
	docker build --target production -t $(PROJECT_NAME):prod --network=host -f docker/Dockerfile.prod .

# Container execution
container:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_framework -it --network=host --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME):dev bash

container-prod:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_prod -it --network=host --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME):prod bash

# Poetry utility commands (for local development)
poetry-install:
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) install $(POETRY_INSTALL_OPTS)

poetry-update:
	$(POETRY) update

poetry-shell:
	$(POETRY) shell

poetry-export:
	$(POETRY) export -f requirements.txt --output requirements.txt
	$(POETRY) export -f requirements.txt --with dev --output requirements-dev.txt

poetry-test:
	$(POETRY) run pytest

poetry-lint:
	$(POETRY) run flake8 src
	$(POETRY) run mypy src

poetry-docs:
	$(POETRY) run sphinx-build -b html docs docs/_build/html

poetry-run:
	@if [ -z "$(CMD)" ]; then \
		echo "Error: CMD is required. Usage: make poetry-run CMD='python src/main.py'"; \
		exit 1; \
	fi
	$(POETRY) run $(CMD)

# Project-specific targets that utilize Poetry
train:
	$(PYTHON) src/main.py

tokenize:
	$(PYTHON) src/main.py --task tokenization

validate:
	$(PYTHON) src/main.py --validate-config

# Release workflow
release: poetry-export build-prod
	@echo "Release build complete - tag with: git tag v$$(poetry version -s)"
	@echo "Then push with: git push origin v$$(poetry version -s)"

# Cleanup commands
clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
	-@rm -rf .pytest_cache .mypy_cache .coverage htmlcov 2> /dev/null
	-@rm -rf docs/_build 2> /dev/null

clean-docker:
	-@docker rmi $(PROJECT_NAME):dev $(PROJECT_NAME):prod 2> /dev/null || true
