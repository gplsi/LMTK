# Cross-platform Makefile for Continual Pretraining Framework
PROJECT_NAME = continual-pretrain
CONFIG_PATH = config
DOCKER_RUN = docker run --rm -v "$(CURDIR)":/workspace
GPU_DEVICES ?= all  # Can specify "0", "0,1" or "none" for CPU-only

.PHONY: help build validate tokenize train dev-shell clean

help:
	@echo Continual Pretraining Framework
	@echo Available targets:
	@echo   build       - Build Docker image
	@echo   validate    - Validate config file (set CONFIG=path.yaml)
	@echo   tokenize    - Run tokenization task
	@echo   train       - Launch distributed training
	@echo   dev-shell   - Start interactive development shell
	@echo   clean       - Clean build artifacts

build:
	docker build -t $(PROJECT_NAME) -f docker/Dockerfile .

validate:
	@$(DOCKER_RUN) $(PROJECT_NAME) \
		python src/main.py --validate --config $(CONFIG_PATH)/$(CONFIG)

tokenize:
	@$(DOCKER_RUN) --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) \
		python src/main.py --task tokenize --config $(CONFIG_PATH)/$(CONFIG)

train:
	@$(DOCKER_RUN) --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) \
		python -m torch.distributed.run \
		--nproc_per_node=$(NUM_GPUS) \
		src/main.py --task train --config $(CONFIG_PATH)/$(CONFIG)

dev-shell:
	$(DOCKER_RUN) -it --gpus '"device=$(GPU_DEVICES)"' $(PROJECT_NAME) bash

clean:
	-@find . -name "*.pyc" -delete 2> /dev/null
	-@rm -rf build dist *.egg-info 2> /dev/null
