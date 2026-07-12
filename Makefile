.PHONY: install simulate test lint format typecheck check

install:
	pip install -e ".[dev,model,llm,api,dashboard]"

simulate:
	python -m data_sim.simulate

simulate-small:
	python -m data_sim.simulate --n-customers 500 --n-transactions 20000 --output-dir data/simulated_smoke

test:
	pytest -v

lint:
	ruff check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy data_sim features models llm evaluation api

check: lint typecheck test
