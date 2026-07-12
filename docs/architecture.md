# Architecture

Full rationale lives in `PLAN.md` §02. This document tracks the *as-built* architecture as it lands, week by week — `PLAN.md` is the plan, this is the record of what actually exists.

## Data flow

```
data_sim/  (offline, seeded)
  -> customers.parquet, transactions.parquet, manifest.json
     |
features/  (offline)
  -> engineered feature table
     |
models/  (offline)
  -> IsolationForest score -> LightGBM classifier -> trained artifact (joblib)
     |
llm/  (offline batch + capped live path)
  -> Claude explanations + typology classification, cached in Postgres
     |
api/  (FastAPI, serving pre-trained artifacts)
  -> inference + explanation endpoints
     |
dashboard/  (Streamlit)
  -> investigator triage UI
```

## Deployed topology (target: Week 7)

```
Internet -> Nginx (TLS) -> [ / -> Streamlit container ]
                            [ /api -> FastAPI container ]
                                        |
                                   Postgres container (same EC2 host)
```

## Status

- [x] `data_sim/` — customer + transaction simulator with 6 typology injectors (Week 1)
- [ ] `features/` — not started
- [ ] `models/` — not started
- [ ] `llm/` — not started
- [ ] `api/` — not started
- [ ] `dashboard/` — not started
- [ ] `infra/` deployment — not started
