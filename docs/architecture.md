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

## Scaling this further (not implemented — out of scope)

The deployed topology above is a deliberate single-box design (see `PLAN.md` §00/§11), not a limitation we ran out of time to fix. If this needed to handle real production load instead of a portfolio demo, the natural next step replaces the single EC2 box with:

- An Auto Scaling Group + Launch Template instead of one hand-configured instance — new instances install Docker and pull images automatically via EC2 user-data (or a pre-baked golden AMI), not a manual SSH session.
- Model artifacts served from S3 (or baked into the Docker image) instead of a volume-mounted local directory, so any instance can start from nothing.
- Postgres moved off the box to RDS — required once app instances are stateless and disposable and can no longer each hold their own local copy of the data.
- Secrets moved from a box-local `.env` to AWS Secrets Manager / SSM Parameter Store, fetched at boot.
- An Application Load Balancer replacing the single Elastic IP, distributing traffic and fronting the Auto Scaling Group.

**Why it isn't built:** this shifts the project from a near-$0 baseline to a fixed floor of roughly $60–100+/month *just for the architecture to exist* — ALB (~$16–20/mo), RDS (~$15/mo after the free-tier year), and a NAT Gateway if instances sit in private subnets (~$32/mo), all billed hourly regardless of traffic. For a demo with near-zero real load, that cost isn't justified, which is why the self-hosted single-box design was chosen deliberately rather than defaulted into.

## Status

- [x] `data_sim/` — customer + transaction simulator with 6 typology injectors (Week 1)
- [ ] `features/` — not started
- [ ] `models/` — not started
- [ ] `llm/` — not started
- [ ] `api/` — not started
- [ ] `dashboard/` — not started
- [ ] `infra/` deployment — not started
