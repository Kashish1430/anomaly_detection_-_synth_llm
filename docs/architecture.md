# Architecture

Full rationale lives in `PLAN.md` §02. This document tracks the *as-built* architecture as it lands, week by week — `PLAN.md` is the plan, this is the record of what actually exists.

## Data flow

Two paths feed the same serving layer: the original offline/batch path (training, and populating the dashboard's initial flagged sample), and the live predict path added after Week 7 (a genuinely new transaction arriving at request time).

```
data_sim/  (offline, seeded)
  -> customers.parquet, transactions.parquet, manifest.json
     |
features/  (offline, full dataset - features/pipeline.py:build_feature_table)
  -> engineered feature table
     |
models/  (offline)
  -> IsolationForest score -> LightGBM classifier -> trained artifact (joblib)
     |
api/export_scored_sample.py + api/load_full_history.py  (offline, one-time)
  -> small pre-scored sample (Postgres `transactions`, is_flagged rows)
  -> full 1.2M-row history + peer_group_stats (Postgres `transactions`/`peer_group_stats`,
     history-only rows - feature-computation context, never individually scored)
     |
llm/  (offline batch + capped live path)
  -> Claude explanations + typology classification, cached in Postgres
```

```
Live predict path (api/main.py: POST /transactions/predict)

  raw transaction (customer_id, amount, channel, ...)
     |
  api/live_features.py
    -> fetches that customer's stored history from Postgres (api/db.py:list_customer_transactions)
    -> runs features/velocity.py, behavioral.py, round_amount.py, contextual.py unchanged,
       on that small per-customer slice
    -> peer_zscore from the precomputed peer_group_stats lookup (not a full peer group scan)
     |
  api/model_bundle.py:score_features  (same function /score uses)
     |
  is_flagged? -> api/explain.py:explain_transaction  (same function /explain uses)
     |
  persisted to Postgres (`transactions`, `explanations`)
     |
  GET /transactions / dashboard  (re-queries Postgres live - no extra step needed)
```

`api/` (FastAPI) serves both paths; `dashboard/` (Streamlit) is a thin client over the API - it never touches Postgres, the model, or the feature pipeline directly.

## Deployed topology

Live at **https://18-133-210-144.sslip.io**.

```
Internet -> Nginx (TLS, host-level - not containerized) -> [ /      -> Streamlit container ]
                                                             [ /api/ -> FastAPI container ]
                                                                         |
                                                                    Postgres container
                                                                    (same EC2 host, all three
                                                                     via Docker Compose)
```

Nginx strips the `/api/` prefix before forwarding, so the `api` container itself doesn't know it's mounted under `/api` - it's told explicitly via `API_ROOT_PATH=/api` (`api/config.py`, only set in the EC2 box's `.env`, empty for local dev) so FastAPI's generated `/docs`/`/openapi.json` references resolve correctly instead of hitting the dashboard's catch-all route.

CI/CD: `.github/workflows/deploy.yml` builds both images on GitHub's runners and pushes to GHCR on every CI-green push to `main`, then SSHes into EC2 for `docker compose pull && up -d` - the box never builds images itself.

## Scaling this further (not implemented — out of scope)

The deployed topology above is a deliberate single-box design (see `PLAN.md` §00/§11), not a limitation we ran out of time to fix. If this needed to handle real production load instead of a portfolio demo, the natural next step replaces the single EC2 box with:

- An Auto Scaling Group + Launch Template instead of one hand-configured instance — new instances install Docker and pull images automatically via EC2 user-data (or a pre-baked golden AMI), not a manual SSH session.
- Model artifacts served from S3 (or baked into the Docker image) instead of a volume-mounted local directory, so any instance can start from nothing.
- Postgres moved off the box to RDS — required once app instances are stateless and disposable and can no longer each hold their own local copy of the data.
- Secrets moved from a box-local `.env` to AWS Secrets Manager / SSM Parameter Store, fetched at boot.
- An Application Load Balancer replacing the single Elastic IP, distributing traffic and fronting the Auto Scaling Group.

**Why it isn't built:** this shifts the project from a near-$0 baseline to a fixed floor of roughly $60–100+/month *just for the architecture to exist* — ALB (~$16–20/mo), RDS (~$15/mo after the free-tier year), and a NAT Gateway if instances sit in private subnets (~$32/mo), all billed hourly regardless of traffic. For a demo with near-zero real load, that cost isn't justified, which is why the self-hosted single-box design was chosen deliberately rather than defaulted into.

## Status

Full week-by-week detail lives in `CLAUDE.md`'s Current Status section - this is a quick top-level summary, not a duplicate.

- [x] `data_sim/` — customer + transaction simulator, 6 typology injectors (Week 1)
- [x] `features/` — 18 leakage-safe engineered features (Week 2)
- [x] `models/` — IsolationForest baseline + tuned LightGBM, Optuna + time-based CV (Weeks 2-3)
- [x] `evaluation/` — hypothesis testing, bias/fairness, sensitivity, PSI monitoring (Weeks 3, 5)
- [x] `llm/` — Claude reasoning layer, provider-swappable, fact-checker, caching (Week 4)
- [x] `docs/model_validation_report.md` — SR 11-7/PRA SS1/23-styled governance report (Week 5)
- [x] `api/` — FastAPI: `/health`, `/score`, `/explain`, `/transactions*`, feedback, and the live
      predict pipeline (`/transactions/predict`) added after Week 7 (Week 6, extended post-Week 7)
- [x] `dashboard/` — Streamlit triage UI, including a "score a new transaction" form (Week 6, extended)
- [x] `infra/` deployment — Docker Compose, GitHub Actions CI/CD, Nginx + Let's Encrypt, live on EC2 (Week 7)
- [ ] Week 8 (polish & buffer) — not yet started; see `CLAUDE.md`'s "Next up"
