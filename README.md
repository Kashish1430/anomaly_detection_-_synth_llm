# LLM-Augmented Transaction Anomaly Detection Engine

[![CI](https://github.com/Kashish1430/anomaly_detection_-_synth_llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Kashish1430/anomaly_detection_-_synth_llm/actions/workflows/ci.yml)

A portfolio project: a synthetic retail-banking transaction dataset, an Isolation Forest / LightGBM anomaly-scoring pipeline, and a Claude reasoning layer that generates investigator-facing explanations for flagged transactions — validated with hypothesis testing and documented to SR 11-7 / PRA SS1/23 model-risk practice.

> Independent portfolio project on synthetic data. Not a real bank engagement or a regulator-reviewed system — see `docs/model_validation_report.md` for the full scope statement.

Full plan and rationale: [`PLAN.md`](PLAN.md). Current status and next steps: [`CLAUDE.md`](CLAUDE.md). What every column/feature means: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Status

Weeks 1-7 complete (simulator, features, tuned model, Claude reasoning layer, statistical validation, model-risk report, containerized API + dashboard, CI/CD, live HTTPS deploy) plus a live predict pipeline added after Week 7. Week 8 (polish & buffer) next. Full detail in [`CLAUDE.md`](CLAUDE.md).

**Live demo:** https://18-133-210-144.sslip.io — browse flagged transactions, generate explanations, submit investigator feedback, or score a brand-new transaction end to end (feature engineering -> model -> LLM explanation -> persisted).

## API usage

Interactive docs (Swagger UI, generated from the actual request/response schemas - can't drift out of sync): **https://18-133-210-144.sslip.io/api/docs**

Score a new transaction directly:

```bash
curl -X POST https://18-133-210-144.sslip.io/api/transactions/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST0000000",
    "timestamp": "2026-07-18T14:00:00Z",
    "amount": 15000.0,
    "direction": "debit",
    "channel": "wire",
    "counterparty_id": "CPTY-TEST-001",
    "counterparty_country": "KY",
    "is_cross_border": true
  }'
```

For a `customer_id` that doesn't exist yet, add `new_customer_segment`, `new_customer_home_country`, and `new_customer_declared_risk_rating` to register it on the fly instead of getting a 400.

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

## Scaling beyond a single box

The live demo runs on one EC2 instance (Docker Compose, self-hosted Postgres) by design, not by oversight — see [`docs/architecture.md`](docs/architecture.md#scaling-this-further-not-implemented--out-of-scope) for the Auto Scaling Group / RDS / ALB path this would take under real load, and why it isn't built here (a production-scale architecture costs roughly $60–100+/month just to exist, before serving any traffic — not justified for a portfolio demo).
