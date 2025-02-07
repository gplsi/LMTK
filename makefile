PROJECT_NAME = continual-pretrain
CONFIG_PATH = config

.PHONY: build validate tokenize train

build:
	docker build -t $(PROJECT_NAME) -f docker/dockerfile .

validate:
	@docker run --rm -v $(PWD):/workspace $(PROJECT_NAME) \
		python src/main.py --validate --config $(CONFIG_PATH)/$(CONFIG)

tokenize:
	@docker run --gpus all --rm -v $(PWD):/workspace $(PROJECT_NAME) \
		python src/main.py --task tokenize --config $(CONFIG_PATH)/$(CONFIG)

train:
	@docker run --gpus all --rm -v $(PWD):/workspace $(PROJECT_NAME)
		python -m torch.distributed.run --nproc_per_node=$(NUM_GPUS) \
		src/main.py --task train --config $(CONFIG_PATH)/$(CONFIG)

dev-shell:
	docker run -it --gpus all --rm -v $(PWD):/workspace $(PROJECT_NAME) bash

clean:
	find . -name "*.pyc" -delete
	rm -rf build dist *.egg-info
