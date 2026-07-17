.PHONY: install simulate test lint format typecheck check db-up db-down serve load-data export-scored-sample dashboard

install:
	pip install -e ".[dev,model,llm,api,dashboard]"

# Postgres (infra/docker-compose.yml) - see api/db.py, api/load_data.py.
db-up:
	docker compose -f infra/docker-compose.yml up -d postgres

db-down:
	docker compose -f infra/docker-compose.yml down

# Scores the TEST split with the tuned model and writes a small pre-scored
# sample to data/scored_sample/ - the heavy step (full 1.2M-row dataset,
# model inference), meant to run locally/offline, not on the serving box.
# Run once after models/package_artifact.py has produced artifacts/model_bundle.joblib.
export-scored-sample:
	python -m api.export_scored_sample

# Loads the pre-scored sample (data/scored_sample/, see export-scored-sample)
# into Postgres. Deliberately does no scoring itself - safe to run on a small
# box. Run once after db-up.
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
