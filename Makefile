.PHONY: help install dev test lint run run-docker docker-build docker-up docker-down clean deploy-dev deploy-staging deploy-prod

# ──────────────────────────────────────────────────────────────────────────────
# MLOps Platform — Makefile
# ──────────────────────────────────────────────────────────────────────────────

PYTHON      ?= python
DOCKER      ?= docker
COMPOSE     ?= docker compose
DATABRICKS  ?= databricks
PIP         ?= pip

# Colors
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)MLOps Platform$(RESET) — Available Commands"
	@echo "────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Local Development ────────────────────────────────────────────────────────

install: ## Install dependencies
	$(PIP) install -r requirements.txt

dev: ## Install with dev dependencies
	$(PIP) install -e ".[all]"

test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check src/ tests/ orchestration/
	$(PYTHON) -m ruff format --check src/ tests/ orchestration/

format: ## Auto-format code
	$(PYTHON) -m ruff format src/ tests/ orchestration/
	$(PYTHON) -m ruff check --fix src/ tests/ orchestration/

typecheck: ## Run type checker (mypy)
	$(PYTHON) -m mypy src/ orchestration/

run: ## Run the full pipeline locally
	$(PYTHON) -m orchestration.pipeline_runner

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	$(COMPOSE) build

docker-up: ## Start MLflow UI server (http://localhost:5000)
	$(COMPOSE) up mlflow -d
	@echo ""
	@echo "$(GREEN)✅ MLflow UI running at http://localhost:5000$(RESET)"

docker-run: ## Run full pipeline in Docker
	$(COMPOSE) run --rm pipeline

docker-test: ## Run tests in Docker
	$(COMPOSE) --profile test run --rm tests

docker-shell: ## Open interactive shell in container
	$(COMPOSE) --profile dev run --rm shell

docker-down: ## Stop all containers and remove volumes
	$(COMPOSE) down -v

# ── Databricks Deployment ────────────────────────────────────────────────────

deploy-validate: ## Validate Databricks Asset Bundle
	$(DATABRICKS) bundle validate -t dev

deploy-dev: ## Deploy to Databricks dev workspace
	$(DATABRICKS) bundle deploy -t dev
	@echo "$(GREEN)✅ Deployed to dev$(RESET)"

deploy-staging: ## Deploy to Databricks staging workspace
	$(DATABRICKS) bundle deploy -t staging
	@echo "$(GREEN)✅ Deployed to staging$(RESET)"

deploy-prod: ## Deploy to Databricks production workspace
	$(DATABRICKS) bundle deploy -t prod
	@echo "$(GREEN)✅ Deployed to production$(RESET)"

deploy-run-dev: ## Deploy and run pipeline in dev
	$(DATABRICKS) bundle deploy -t dev
	$(DATABRICKS) bundle run -t dev mlops-platform-daily-pipeline

# ── Terraform ────────────────────────────────────────────────────────────────

tf-init: ## Initialize Terraform
	cd infrastructure/terraform && terraform init

tf-plan: ## Plan Terraform changes
	cd infrastructure/terraform && terraform plan -var-file="prod.tfvars"

tf-apply: ## Apply Terraform changes
	cd infrastructure/terraform && terraform apply -var-file="prod.tfvars" -auto-approve

tf-destroy: ## Destroy Terraform resources
	cd infrastructure/terraform && terraform destroy -var-file="prod.tfvars"

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove generated files
	rm -rf data/ mlruns/ mlartifacts/ .pytest_cache/ htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleaned$(RESET)"
