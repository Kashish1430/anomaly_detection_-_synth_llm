.PHONY: install simulate test lint format typecheck check db-up db-down serve load-data dashboard

install:
	pip install -e ".[dev,model,llm,api,dashboard]"

# Postgres (infra/docker-compose.yml) - see api/db.py, api/load_data.py.
db-up:
	docker compose -f infra/docker-compose.yml up -d postgres

db-down:
	docker compose -f infra/docker-compose.yml down

# Loads the tuned model's flagged TEST-split transactions into Postgres -
# run once after db-up and after models/package_artifact.py has produced
# artifacts/model_bundle.joblib.
load-data:
	python -m api.load_data

# Prefer this over `uvicorn api.main:app` directly on Windows - see
# api/serve.py's docstring for why.
serve:
	python -m api.serve

dashboard:
	streamlit run dashboard/app.py

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
	mypy data_sim features models evaluation llm api dashboard

check: lint typecheck test
