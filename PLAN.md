# LLM-Augmented Transaction Anomaly Detection Engine — Master Plan

A from-scratch, portfolio-grade fraud/AML triage system: synthetic transaction data, an Isolation Forest → LightGBM scoring layer, a Claude reasoning layer for investigator-facing explanations, statistically validated thresholds, model-risk documentation, and a deployed, CV-linkable demo — built and shipped the way a small production team would.

**Horizon:** 6–8 weeks, part-time · **LLM:** Claude API (Haiku + Sonnet) · **Host:** single AWS EC2 + Docker · **Est. cash cost:** $0–20 total

---

## Table of contents

- [00 · Framing & ground rules](#00--framing--ground-rules)
- [01 · What you'll actually learn](#01--what-youll-actually-learn)
- [02 · System architecture](#02--system-architecture)
- [03 · Data strategy](#03--data-strategy--the-simulator)
- [04 · Feature engineering](#04--feature-engineering)
- [05 · Scoring models](#05--scoring-models)
- [06 · Claude reasoning layer](#06--claude-reasoning-layer)
- [07 · Statistical validation](#07--statistical-validation)
- [08 · Model-risk documentation](#08--model-risk-documentation)
- [09 · Repository structure](#09--repository-structure)
- [10 · CI/CD](#10--cicd)
- [11 · Deployment (EC2)](#11--deployment-single-ec2-instance)
- [12 · Risk register](#12--risk-register)
- [13 · Eight-week roadmap](#13--eight-week-roadmap)
- [14 · Cost ledger](#14--cost-ledger)
- [15 · CV bullets, rewritten honestly](#15--cv-bullets-rewritten-honestly)
- [16 · Next step](#16--next-step)

---

## 00 · Framing & ground rules

Three decisions from our scoping conversation shape everything below — stating them up front so the rest of the plan is legible.

| Decision | What we're doing |
|---|---|
| LLM provider | Swapping GPT-4 for **Claude** (Haiku for bulk, Sonnet for hard cases), since you already hold an Anthropic subscription. See the cost note below — this is *not* literally $0. |
| Frontend/hosting | Dropping Next.js/Vercel/Render (you don't know them, and DS interviewers won't ask). Using **Streamlit + FastAPI + Docker on a single EC2 box** — plays to your AWS knowledge, still teaches real deployment skills. |
| Database | Self-hosted Postgres in a Docker container on the same EC2 instance, not RDS — you flagged RDS as costly. RDS stays a documented "upgrade path," not the default. |

> **Read this one.** A Claude.ai **Pro subscription does not include API access.** The API is billed separately through `console.anthropic.com` on pay-as-you-go credits. First action item in Week 1 is opening an API console account and loading a small credit balance — realistically **$5–15 for the entire project** if we use Haiku for the bulk pass and cache aggressively (design in §06). If you want truly $0, the fallback is Llama 3.1 8B via Groq's free API tier — swappable behind the same interface, noted inline where it matters.

> **Integrity rule.** The CV bullets you pasted contain specific numbers — 65% effort reduction, 71%→89% precision, 40% fewer false positives. Treat those as a *target shape*, not a script. We build the measurement pipeline first, run the real experiment, and then write down whatever the pipeline actually reports — including if it's less flattering. A false number on a CV is a bigger risk to your career than a modest true one; an interviewer who asks "walk me through how you got 89%" will find out in ninety seconds if you can't. §15 has the honest rewrite template.

---

## 01 · What you'll actually learn

Mapped so you can tell, week to week, that the time is buying real skill and not just CV copy.

| Domain | Concrete skills exercised |
|---|---|
| Data science core | Synthetic data design with controllable ground truth, behavioural feature engineering, unsupervised + supervised anomaly detection, hyperparameter search, calibration, time-based cross-validation |
| Applied statistics | Two-proportion z-tests for threshold selection, confidence intervals on precision/recall, population stability index for drift |
| LLM engineering | Structured-output prompting, tool-use/JSON schemas, hallucination guardrails, prompt caching, cost governance for a public-facing LLM feature |
| MLOps | Experiment tracking (MLflow), data validation (pandera), model packaging, reproducible environments, containerization |
| Backend / infra | FastAPI service design, Streamlit UI, Docker Compose, Nginx reverse proxy + TLS, EC2 provisioning and hardening |
| Production process | CI/CD with GitHub Actions, branch protection, secrets hygiene, model-risk documentation (SR 11-7 / PRA SS1/23 practice), incident-style risk registers |

---

## 02 · System architecture

Everything heavy (data generation, training, tuning) happens offline/locally. The EC2 box only serves a small, pre-trained artifact — so it can be cheap and small.

```
Simulator → Feature pipeline → IsolationForest + LightGBM → Claude reasoning
(1.2M synthetic   (velocity, peer-      (scikit-learn, LightGBM,   (explanations +
 transactions,     deviation,            Optuna, MLflow)            typology class.,
 injected           round-amount)                                    cached)
 typologies)

Trained artifacts → FastAPI → Streamlit → Nginx + TLS
(joblib model +      (inference +   (investigator    (reverse proxy on
 curated scored       explanation     triage           EC2, single
 sample, shipped      endpoints,      dashboard)        public URL)
 as release asset)    Pydantic
                       schemas)
```

Postgres (Docker, on-box) stores transactions, scores, cached LLM explanations, and investigator feedback — the last of these lets the dashboard demo a feedback loop ("mark false positive"), which is a nice, honest way to show you understand triage workflows without needing a real bank's ticketing system.

---

## 03 · Data strategy — the simulator

No public dataset combines 1.2M labeled retail-banking transactions with AML typology tags. So we *build* the dataset — which conveniently is exactly what your CV bullet already says ("simulated retail-banking dataset"), and it's the only way to get trustworthy ground truth for the precision numbers.

**Simulator design:**

- **Customers (~8–15k):** segment (retail/SME/etc.), typical monthly volume, amount distribution (lognormal), home region, declared risk rating.
- **Transactions (~1.2M):** generated per customer via a Poisson process for frequency and the customer's lognormal amount profile, across channels (branch/online/ATM/wire) and counterparties, over a 12–18 month synthetic window.
- **Injected typologies (~1–3% of transactions), each mapped to a FATF red-flag category:**
  - Structuring/smurfing — clustered transactions just under a reporting threshold
  - Layering/round-tripping — rapid in-and-out movement through intermediate accounts
  - Round-amount anomalies — abnormal frequency of suspiciously round sums
  - Velocity spikes — volume/count far outside the customer's own baseline
  - Peer-group deviation — behaviour inconsistent with the customer's declared segment
  - Geographic risk — sudden activity linked to higher-risk jurisdictions
- **Ground truth** (is_anomalous + typology label) is generated but withheld from the feature set — used only for evaluation and for a stratified "hand-labelled validation set" slice (~5–10k rows), which stands in honestly for a human reviewer and is documented as such.

Data validated on generation with `pandera` schemas (types, ranges, referential integrity between customers/transactions) — this is the "data quality gate," a real production practice, not busywork.

---

## 04 · Feature engineering

| Family | Examples |
|---|---|
| Velocity | Rolling transaction count/sum per customer over 1h / 24h / 7d / 30d windows |
| Peer-group deviation | Robust z-score (median/MAD) of amount vs. the customer's declared segment |
| Round-amount | Flag for round-hundred/round-thousand amounts; frequency of round amounts per customer |
| Behavioural drift | Ratio of current transaction to the customer's own trailing average; new-counterparty flag; channel-switch flag |
| Contextual | Cross-border flag, time-of-day/day-of-week deviation from the customer's own pattern |

---

## 05 · Scoring models

1. **Unsupervised baseline** — `IsolationForest` (scikit-learn) over the engineered features gives an anomaly score with zero labels. A naive fixed-threshold cut on this score becomes your honest *"before"* precision number — whatever it actually measures, that's what replaces the placeholder 71%.
2. **Supervised refinement** — `LightGBM` classifier trained on the hand-labelled slice, using the IsolationForest score plus the full feature set. Tuned with Optuna; every run logged to MLflow so you can show a real experiment history, not just a final number.
3. **Threshold selection** — not eyeballed. See §07.

---

## 06 · Claude reasoning layer

This is the layer most likely to either impress an interviewer or embarrass you, depending on whether it has guardrails. Design it like a feature that will face an auditor, not a chatbot demo.

| Concern | Design choice |
|---|---|
| Cost | **Haiku** for the bulk explanation pass over all flagged transactions (~12–36k of the 1.2M); **Sonnet** only on a stratified ~1–2k sample used to measure typology-classification accuracy against the hand-labelled set. Prompt caching on the fixed system prompt / typology taxonomy. Every response cached in Postgres so re-running an evaluation never re-calls the API. |
| Hallucination | Structured output via tool-use (guaranteed JSON, validated with Pydantic). System prompt explicitly forbids citing any fact not present in the supplied feature dict. A post-hoc **fact-checker** regex-matches every number in the generated explanation against the source data and flags mismatches for human review — a genuinely useful, demonstrable guardrail. |
| Output | Per flagged transaction: natural-language explanation, one label from a fixed FATF-aligned typology taxonomy, a confidence score, and a "likely false positive" flag. |
| Reliability | Async calls with a concurrency cap, exponential backoff on rate limits, all failures fall back to a rule-based templated explanation rather than a broken UI. |
| Public-demo cost control | The deployed dashboard defaults to **precomputed, cached** explanations for a curated set of ~300 interesting flagged cases. A capped "Generate live explanation" button (session-limited) shows the real pipeline working without unbounded API exposure to strangers on the internet. |

---

## 07 · Statistical validation

- **Threshold tuning:** for a grid of candidate cut-points, compute the false-positive rate on held-out folds and run a **two-proportion z-test** (statsmodels `proportions_ztest`, α = 0.05) against the baseline threshold. Pick the threshold that minimizes FP rate subject to a recall floor (e.g. ≥ 80%) — not the one that just looks best.
- **Time-based cross-validation:** expanding-window CV across the synthetic transaction timeline (not random k-fold) — because in real fraud/AML data, patterns drift, and a model that only works on shuffled data is a model that will fail in production. Report precision/recall variance across folds, not just a point estimate.
- **Confidence intervals** on every headline metric that goes on the CV — a number with no interval is not a number an interviewer should trust, and neither should you.

---

## 08 · Model-risk documentation

Produces `docs/model_validation_report.md`, structured like a real SR 11-7 / PRA SS1/23 model validation packet — this is the artifact that makes the CV line about "defensible to a non-technical audit committee" true rather than aspirational.

> **Say this out loud in interviews.** This is an independent portfolio project on synthetic data — it demonstrates fluency with model-risk-management *practice*, not an actual regulatory validation of a production bank system. The report says so on page one. Framing it any other way is the one move that could actually damage your credibility with a technical interviewer who knows the space.

**Report structure:**

- Purpose & scope
- Model description & conceptual soundness
- Data lineage & quality (pandera gate results)
- Development testing (train/test + time-split performance)
- Outcomes analysis / backtesting
- Sensitivity analysis (feature perturbation)
- Benchmarking against the naive rule-based baseline
- Bias & fairness check across simulated customer segments (statistical-parity-style comparison of flagging rates)
- Limitations & assumptions — stated plainly, including "synthetic data" and "labels are simulated, not human-reviewed"
- Ongoing monitoring plan — population stability index thresholds, retraining triggers
- Governance sign-off template (roles named even though you're filling all of them — noted as such)

---

## 09 · Repository structure

```
anomaly-detection-engine/
├── data_sim/            # customer + transaction simulator, typology injectors
├── features/            # feature pipeline (velocity, peer-deviation, round-amount)
├── models/               # training scripts, Optuna tuning, MLflow logging
├── llm/                  # Claude prompts, structured-output schemas, fact-checker, cache
├── evaluation/            # z-tests, time-based CV, bias/fairness checks
├── api/                  # FastAPI app (inference + explanation endpoints)
├── dashboard/             # Streamlit investigator triage UI
├── docs/
│   ├── model_validation_report.md
│   ├── architecture.md
│   └── adr/               # architecture decision records
├── infra/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── nginx/
├── tests/
├── .github/workflows/     # ci.yml, deploy.yml
├── notebooks/             # EDA, exploratory model comparisons
├── pyproject.toml
└── README.md
```

Plus the usual: `.gitignore` (never commit `.env`), `LICENSE` (MIT), a PR template, and branch protection on `main` requiring CI to pass. The full 1.2M-row dataset isn't committed to git — it's regenerated by `make simulate` (seeded, reproducible) and the trained model + a curated scored sample ship as a GitHub Release asset.

---

## 10 · CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | Every PR | ruff lint · black format check · mypy · pytest + coverage · pandera schema smoke test |
| `deploy.yml` | Push to `main` | Build API + dashboard Docker images → push to **GitHub Container Registry** (free, avoids ECR cost) → SSH into EC2 → `docker compose pull && up -d` |

Secrets (EC2 SSH key, Anthropic API key, DB password) live only in GitHub Actions encrypted secrets and the EC2 instance's own `.env` — never in the repo. Add a pre-commit hook running `gitleaks` or similar as a last line of defence.

---

## 11 · Deployment: single EC2 instance

Because training happens offline, the serving box only needs to run inference on pre-scored data — so it can stay small and close to free.

| Component | Choice | Why |
|---|---|---|
| Instance | t3.micro (free-tier eligible) or t3.small (~$15/mo) if micro feels tight | Serving pre-trained artifacts is light; heavy compute never runs here |
| Database | Postgres in Docker, on-box | Avoids RDS cost; you already know RDS as an upgrade path if you ever want it |
| Reverse proxy | Nginx + Let's Encrypt (certbot) | One HTTPS entrypoint routing `/` → Streamlit, `/api` → FastAPI |
| Domain | Optional ~$12/yr (Route 53 or Namecheap) vs. free EC2 public DNS | A real domain reads better on a CV than an `ec2-...amazonaws.com` URL, but it's genuinely optional |

---

## 12 · Risk register

The part you asked for explicitly: what breaks, how likely, how bad, what we do about it now rather than after it happens.

| Risk | Sev | Mitigation |
|---|---|---|
| Synthetic data is unrealistic or leaks the label into a feature | High | Ground-truth typology fields excluded from the feature set by construction; leakage check via a "can a single feature alone predict the label" audit |
| Claude hallucinates a fact in an investigator-facing explanation | High | Structured output + fact-checker cross-referencing every number against source data (§06) |
| Reported metrics are cherry-picked / overfit | High | Time-based CV, confidence intervals, and a fixed held-out set touched only once, at the end |
| CV numbers don't match what the pipeline actually produces | High | Integrity rule in §00 — numbers are written down only after the pipeline runs |
| API key or DB password committed to git | High | `.env` + `.gitignore` + `gitleaks` pre-commit hook + GitHub secret scanning enabled |
| Claude API cost creep from the public demo | Med | Cached-by-default demo, session-capped live-call button (§06) |
| EC2 cost creep after free tier expires | Med | t3.micro sized for light serving load; billing alarm set in AWS Budgets at $10 |
| Class imbalance inflates accuracy-looking metrics | Med | Report precision/recall/PR-AUC, never bare accuracy, on a ~1–3% positive rate |
| Bias/unfair flagging across simulated customer segments | Med | Explicit statistical-parity-style check in the governance report (§08) |
| EC2 box goes down with nobody else on call | Low | Docker Compose `restart: unless-stopped`; a status badge/uptime check (e.g. UptimeRobot free tier) rather than manual polling |
| Scope creep stretches 8 weeks into forever | Med | Phased roadmap (§13) with a hard Definition of Done per week; polish is explicitly Week 8, not interleaved |

---

## 13 · Eight-week roadmap

| Week | Theme | Definition of Done |
|---|---|---|
| 1 | Foundations & simulator | Repo scaffold, pre-commit, Anthropic API console set up. `make simulate` reproducibly generates the 1.2M-row dataset with documented typology injection rates; EDA notebook committed. |
| 2 | Features & baseline | Velocity/peer/round-amount features, IsolationForest baseline. Feature pipeline unit-tested; honest "before" precision number logged to MLflow. |
| 3 | Supervised model & stats | LightGBM + Optuna, two-proportion z-test threshold tuning, time-based CV. Honest "after" precision number with confidence interval. |
| 4 | Claude reasoning layer | Prompting, structured output, fact-checker, caching. End-to-end explanation pipeline on a sample; real API cost logged. |
| 5 | Governance & bias | Model validation report, bias/fairness check, monitoring plan. `docs/model_validation_report.md` complete. |
| 6 | API & dashboard | FastAPI + Streamlit + Docker Compose. Full stack runs locally with one command. |
| 7 | CI/CD & deploy | GitHub Actions, EC2, Nginx + TLS. Live clickable URL, deploy triggered by merge to `main`. |
| 8 | Polish & buffer | README + architecture diagram, model card, finalize CV bullets against real numbers, smoke-test the live demo. A stranger can open the link and understand the project in two minutes. |

---

## 14 · Cost ledger

| Item | Est. | Notes |
|---|---|---|
| GitHub | $0 | Public repo, free Actions minutes, free Container Registry |
| Claude API | $5–15 | Haiku bulk pass + capped Sonnet sample + prompt caching |
| AWS EC2 | $0–15/mo | Free tier if available; ~$15/mo t3.small otherwise. Set a Budgets alarm. |
| Domain (optional) | ~$12/yr | Skippable — EC2 gives a free public DNS name |
| **Total to launch** | **$5–30** | Not $0, but close, and mostly reusable Anthropic credit |

---

## 15 · CV bullets, rewritten honestly

Same structure and ambition as your draft — swapped to Claude, and with the percentages held as variables until the pipeline reports real ones.

```
Reduced simulated manual transaction-review effort by ~[X]% on a self-generated
retail-banking dataset (1.2M synthetic records), by combining an Isolation
Forest / LightGBM anomaly-scoring layer (scikit-learn) with a Claude reasoning
layer that generated natural-language explanations of each flag for
investigator triage.

Improved anomaly precision from [X]% to [Y]% against a hand-labelled
validation set, by engineering behavioural features (velocity, peer-group
deviation, round-amount patterns) and using Claude to classify edge cases
against AML red-flag typologies aligned to the FATF Recommendations and
UK MLR 2017.

Cut false-positive alerts by ~[X]%, by tuning decision thresholds through
hypothesis testing (two-proportion z-tests, α = 0.05) and validating
stability across time-based cross-validation folds.

Documented the full model lifecycle in a validation and bias report
structured to SR 11-7 / PRA SS1/23 model-risk-management practice, making
outputs defensible to a non-technical review audience.
```

Fill the `[X]`/`[Y]` placeholders in Week 8, from the numbers Weeks 2–3 actually produced — and be ready to say "independent project, synthetic data" in the same breath if asked, not as a hedge but as the accurate description of real, demonstrable work.

---

## 16 · Next step

Say the word and we start Week 1: repo scaffold, environment setup, and the transaction/customer simulator. That's the one part everything else depends on, so it's worth getting the typology injection logic right before touching a model.

---

*Master plan — built for a Data Scientist CV project. Numbers throughout are placeholders until the pipeline runs; treat §00 and §15 as binding.*
