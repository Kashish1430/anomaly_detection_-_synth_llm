# LLM-Augmented Transaction Anomaly Detection Engine

[![CI](https://github.com/Kashish1430/anomaly_detection_-_synth_llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Kashish1430/anomaly_detection_-_synth_llm/actions/workflows/ci.yml)

A portfolio project: a synthetic retail-banking transaction dataset, an Isolation Forest / LightGBM anomaly-scoring pipeline, and a Claude reasoning layer that generates investigator-facing explanations for flagged transactions — validated with hypothesis testing and documented to SR 11-7 / PRA SS1/23 model-risk practice.

> Independent portfolio project on synthetic data. Not a real bank engagement or a regulator-reviewed system — see `docs/model_validation_report.md` for the full scope statement.

Full plan and rationale: [`PLAN.md`](PLAN.md). Current status and next steps: [`CLAUDE.md`](CLAUDE.md). What every column/feature means: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Status

Week 2 done (feature pipeline + IsolationForest baseline). Week 3 (LightGBM + threshold tuning) next. No deployed demo yet.

## Quickstart

```bash
python -m venv venv
# Windows
venv\Scripts\pip install -e ".[dev,model,llm,api,dashboard]"
# macOS/Linux
venv/bin/pip install -e ".[dev,model,llm,api,dashboard]"

# generate the synthetic dataset (reproducible, seeded)
python -m data_sim.simulate

# or, on a platform with `make` (CI, macOS/Linux):
make install
make simulate
```

Windows note: this repo's `Makefile` targets are used in CI (which runs on Ubuntu); on Windows, run the underlying `python -m ...` commands directly, as shown above.

## Repository layout

See `PLAN.md` §09 for the full structure and rationale. Top level:

- `data_sim/` — synthetic customer & transaction generator, AML typology injectors
- `features/` — behavioural feature engineering (velocity, peer-deviation, round-amount)
- `models/` — Isolation Forest / LightGBM training, tuning, MLflow tracking
- `llm/` — Claude prompts, structured-output schemas, fact-checking guardrail
- `evaluation/` — hypothesis testing, time-based CV, bias/fairness checks
- `api/` — FastAPI inference & explanation service
- `dashboard/` — Streamlit investigator triage UI
- `docs/` — model validation report, architecture notes, ADRs
- `infra/` — Docker Compose, Dockerfiles, Nginx config
