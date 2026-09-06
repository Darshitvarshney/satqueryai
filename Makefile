.PHONY: help install install-ml install-geo install-dev dev run test test-api smoke docker

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:      ## Install core backend (API + orchestration)
	pip install -r requirements.txt && pip install --no-deps -e .

install-ml:   ## Install the local Qwen2-VL vision model deps
	pip install -r requirements-ml.txt

install-geo:  ## Install rasterio / pyproj for the geo-spatial specialist
	pip install -r requirements-geo.txt

install-dev:  ## Install core deps + test tooling
	pip install -r requirements-dev.txt && pip install --no-deps -e .

dev:          ## Run the API with autoreload
	uvicorn satquery.main:app --reload --host 0.0.0.0 --port 8000

run:          ## Run the API (production style)
	python -m satquery

test:         ## Run the test suite (VLM + LLM stubbed)
	pytest

test-api:     ## Run the Postman collection against a running server (needs: npm i -g newman)
	newman run docs/postman/SatQuery.postman_collection.json \
	  -e docs/postman/SatQuery.local.postman_environment.json --working-dir .

smoke:        ## Run one analysis end to end with stubs
	python scripts/smoke_test.py

docker:       ## Build the Docker image
	docker build -t satquery-backend:latest .
