# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run -v "$(CURDIR)":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only
POETRY = poetry
PYTHON = $(POETRY) run python
TOX = $(POETRY) run tox

# Poetry configuration
POETRY_VENV = .venv
POETRY_INSTALL_OPTS ?= 

# Pip configuration
PIP = pip
PIP_INSTALL_OPTS ?= -e
PIP_VENV = .pip-venv

# Installation method - defaults to poetry, can be overridden with `make INSTALL_METHOD=pip`
INSTALL_METHOD ?= poetry

# Version management
VERSION_BUMP ?= patch  # Options: major, minor, patch
CHANGELOG_FILE = CHANGELOG.md
VERSION = $(shell $(POETRY) version -s)

# CI configuration
TOX_ENV ?= py310  # Default tox environment

.PHONY: help build build-dev build-prod container validate tokenize train clean install install-poetry install-pip poetry-install poetry-update poetry-shell poetry-export poetry-test poetry-lint poetry-docs poetry-run pip-install pip-update pip-venv pip-test pip-run version-bump version-show version-release changelog ci ci-all ci-test ci-lint ci-type ci-coverage

help:
	@echo "Continual Pretraining Framework"
	@echo "Available targets:"
	@echo   "build            - Build Docker image with Poetry (development)"
	@echo   "build-dev        - Build Docker image for development"
	@echo   "build-prod       - Build Docker image for production (smaller size)"
	@echo   "container        - Start interactive development shell"
	@echo   ""
	@echo   "Installation:"
	@echo   "install          - Install project using preferred method (default: poetry)"
	@echo   "                   Override with INSTALL_METHOD=pip"
	@echo   "install-poetry   - Install using Poetry specifically"
	@echo   "install-pip      - Install using pip specifically"
	@echo   ""
	@echo   "Poetry commands:"
	@echo   "poetry-install   - Install dependencies using Poetry"
	@echo   "poetry-update    - Update dependencies to latest versions"
	@echo   "poetry-shell     - Activate Poetry virtual environment"
	@echo   "poetry-export    - Export dependencies to requirements.txt"
	@echo   "poetry-test      - Run tests using pytest through Poetry"
	@echo   "poetry-lint      - Run linting through Poetry"
	@echo   "poetry-docs      - Build documentation using Sphinx"
	@echo   "poetry-run       - Run a command in the Poetry environment (usage: make poetry-run CMD='python src/main.py')"
	@echo   ""
	@echo   "Pip commands:"
	@echo   "pip-venv         - Create virtual environment for pip"
	@echo   "pip-install      - Install using pip in a virtual environment"
	@echo   "pip-test         - Run tests using pytest through pip"
	@echo   "pip-run          - Run a command in the pip environment (usage: make pip-run CMD='python src/main.py')"
	@echo   ""
	@echo   "Version management:"
	@echo   "version-show     - Show current project version"
	@echo   "version-bump     - Bump version (default: patch) - use VERSION_BUMP=major|minor|patch"
	@echo   "version-release  - Prepare a new release (updates changelog, tags, commits)"
	@echo   "changelog        - Open the changelog file for editing"
	@echo   ""
	@echo   "CI commands:"
	@echo   "ci               - Run tests with tox (default: py310)"
	@echo   "                   Override with TOX_ENV=environment"
	@echo   "ci-all           - Run all CI checks (tests, lint, type, coverage)"
	@echo   "ci-test          - Run tests with tox"
	@echo   "ci-lint          - Run linting with tox"
	@echo   "ci-type          - Run type checking with tox"
	@echo   "ci-coverage      - Run test coverage with tox"

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

# Universal installation target that uses either Poetry or pip
install:
	@echo "Installing with $(INSTALL_METHOD)..."
	@$(MAKE) install-$(INSTALL_METHOD)

# Poetry-specific installation
install-poetry: poetry-install

# Pip-specific installation
install-pip: pip-install

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

# Pip commands
pip-venv:
	@echo "Creating pip virtual environment..."
	@if [ ! -d "$(PIP_VENV)" ]; then \
		python3 -m venv $(PIP_VENV); \
	fi

pip-install: pip-venv
	@echo "Installing with pip..."
	@. $(PIP_VENV)/bin/activate && \
	$(PIP) install $(PIP_INSTALL_OPTS) .[dev]

pip-test: pip-venv
	@echo "Running tests with pytest through pip..."
	@. $(PIP_VENV)/bin/activate && \
	pytest

pip-run: pip-venv
	@if [ -z "$(CMD)" ]; then \
		echo "Error: CMD is required. Usage: make pip-run CMD='python src/main.py'"; \
		exit 1; \
	fi
	@echo "Running command with pip virtual environment..."
	@. $(PIP_VENV)/bin/activate && \
	$(CMD)

# Version management commands
version-show:
	@echo "Current version: $(VERSION)"
	@echo "To bump version, use: make version-bump [VERSION_BUMP=major|minor|patch]"

version-bump:
	@echo "Bumping $(VERSION_BUMP) version..."
	@$(POETRY) version $(VERSION_BUMP)
	@NEW_VERSION=$$($(POETRY) version -s); \
	echo "New version: $$NEW_VERSION"
	@echo "Remember to update the CHANGELOG.md file with your changes"
	@echo "Then run 'make version-release' to finalize the release"

changelog:
	@echo "Opening CHANGELOG.md for editing..."
	@$${EDITOR:-nano} $(CHANGELOG_FILE)

version-release:
	@if ! git diff --quiet HEAD -- $(CHANGELOG_FILE); then \
		echo "Changelog has been modified. Proceeding with release."; \
	else \
		echo "ERROR: Please update the CHANGELOG.md file first with your changes using 'make changelog'"; \
		exit 1; \
	fi
	@echo "Preparing release for version $(VERSION)..."
	@TODAY=$$(date +%Y-%m-%d); \
	VERSION_STRING="## [$(VERSION)] - $$TODAY"; \
	sed -i "s/## \[Unreleased\]/## \[Unreleased\]\n\n### Added\n- Future additions will be documented here before release\n\n$$VERSION_STRING/" $(CHANGELOG_FILE)
	@echo "Updating changelog links..."
	@REPO_URL=$$(grep "\[Unreleased\]" $(CHANGELOG_FILE) | sed -E 's/.*https:\/\/github.com\/([^\/]+\/[^\/]+)\/compare.*/https:\/\/github.com\/\1/'); \
	if [ -n "$$REPO_URL" ]; then \
		LAST_VERSION=$$(grep -o "\[[0-9]\+\.[0-9]\+\.[0-9]\+\]" $(CHANGELOG_FILE) | head -1 | tr -d '[]'); \
		sed -i "s|\[Unreleased\]: .*|[Unreleased]: $$REPO_URL/compare/v$(VERSION)...HEAD\n[$(VERSION)]: $$REPO_URL/compare/v$$LAST_VERSION...v$(VERSION)|" $(CHANGELOG_FILE); \
	fi
	@echo "Committing changes..."
	git add $(CHANGELOG_FILE) pyproject.toml
	git commit -m "Release v$(VERSION)"
	git tag -a v$(VERSION) -m "Version $(VERSION)"
	@echo "Release v$(VERSION) prepared!"
	@echo "To push the release, run: git push && git push --tags"
	@echo "To create a GitHub release, visit: $(shell grep -o "https://github.com/[^/]*/[^/]*" $(CHANGELOG_FILE) | head -1)/releases/new?tag=v$(VERSION)"

# CI commands
ci:
	@echo "Running tox with environment: $(TOX_ENV)"
	$(TOX) -e $(TOX_ENV)

ci-all:
	@echo "Running all CI checks with tox"
	$(TOX)

ci-test:
	@echo "Running tests with tox"
	$(TOX) -e py310

ci-lint:
	@echo "Running linting with tox"
	$(TOX) -e lint

ci-type:
	@echo "Running type checking with tox"
	$(TOX) -e type

ci-coverage:
	@echo "Running test coverage with tox"
	$(TOX) -e coverage

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
	-@rm -rf $(PIP_VENV) 2> /dev/null

clean-docker:
	-@docker rmi $(PROJECT_NAME):dev $(PROJECT_NAME):prod 2> /dev/null || true
