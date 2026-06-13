.PHONY: install dev lint format type test cov demo clean

install:        ## Install the package (offline-capable core)
	pip install -e .

dev:            ## Install with dev + all providers + server
	pip install -e ".[all,dev]"

lint:           ## Lint with ruff
	ruff check src tests

format:         ## Auto-format with ruff
	ruff check --fix src tests
	ruff format src tests

type:           ## Static type-check
	mypy src/agentic_ai_eval

test:           ## Run the offline, deterministic test suite
	pytest

cov:            ## Tests with coverage
	pytest --cov=agentic_ai_eval --cov-report=term-missing

demo:           ## Run the end-to-end evaluation-engineering demo (offline)
	python examples/research_pipeline_demo.py

check: lint type test  ## Everything CI runs

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -rf runs *.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
