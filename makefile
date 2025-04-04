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

.PHONY: help build validate tokenize train dev-shell clean poetry-install poetry-update poetry-shell poetry-export poetry-test poetry-lint poetry-docs poetry-run

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build            - Build Docker image"
	@echo   "container        - Start interactive development shell"
	@echo   "clean            - Clean build artifacts"
	@echo   ""
	@echo   "Poetry commands:"
	@echo   "poetry-install   - Install dependencies using Poetry"
	@echo   "poetry-update    - Update dependencies to latest versions"
	@echo   "poetry-shell     - Activate Poetry virtual environment"
	@echo   "poetry-export    - Export dependencies to requirements.txt"
	@echo   "poetry-test      - Run tests using pytest through Poetry"
	@echo   "poetry-lint      - Run linting through Poetry"


build:
	docker build -t $(PROJECT_NAME) --network=host -f docker/Dockerfile .
	
container:
	$(DOCKER_RUN) --name gplsi_continual_pretraining_framework -it --network=host --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) bash

# Poetry utility commands
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




clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
	-@rm -rf .pytest_cache .mypy_cache .coverage htmlcov 2> /dev/null
	-@rm -rf docs/_build 2> /dev/null
