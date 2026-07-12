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

# mypy errors on a directory with no .py files - only list packages that exist
# with source in them yet; extend as llm/api/dashboard land.
typecheck:
	mypy data_sim features models evaluation

check: lint typecheck test
