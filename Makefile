IMAGE_NAME ?= quay.io/surajd/nanochat
# find the latest commit hash for the nanochat repo from the master branch
NANOCHAT_VERSION ?= $(shell gh api repos/karpathy/nanochat/commits/master --jq '.sha' | cut -c1-7)
IMAGE_TAG ?= $(NANOCHAT_VERSION)-$(shell git rev-parse --short HEAD)-$(shell date +%Y%m%d%H%M%S)
PLATFORM ?= linux/amd64

.PHONY: help build push login

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build image locally (docker buildx --load)
	docker buildx build \
		--platform $(PLATFORM) \
		--build-arg NANOCHAT_VERSION=$(NANOCHAT_VERSION) \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--tag $(IMAGE_NAME):latest \
		--load \
		.

push: ## Build and push image to registry (docker buildx --push)
	docker buildx build \
		--platform $(PLATFORM) \
		--build-arg NANOCHAT_VERSION=$(NANOCHAT_VERSION) \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--tag $(IMAGE_NAME):latest \
		--push \
		.
