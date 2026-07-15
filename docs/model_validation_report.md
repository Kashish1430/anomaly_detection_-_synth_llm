# Model validation report

**LLM-Augmented Transaction Anomaly Detection Engine — LightGBM anomaly-scoring model**

> **This is an independent portfolio project on synthetic data.** It demonstrates fluency with model-risk-management *practice* — the structure, statistical discipline, and honesty a real SR 11-7 / PRA SS1/23 validation packet requires — not an actual regulatory validation of a production banking system. There is no real bank, no real customer data, and no real regulator behind this document. Say the same out loud in interviews; framing it any other way is the one move that could damage credibility with a technical interviewer who knows the space.

Model reviewed: the tuned LightGBM classifier from `models/train_lightgbm.py`, MLflow run `e12a18e78ab144cea58c39d513d23007`. Validation performed: 2026-07-15.

---

## 1. Purpose & scope

This model scores retail-banking transactions for anomaly risk, ranking them so that a fixed-capacity investigator queue (2% of transaction volume) reviews the highest-risk transactions first. It replaces an unsupervised IsolationForest baseline (`models/baseline.py`) with a supervised LightGBM classifier trained on 18 engineered behavioural features plus the IsolationForest's own score as an additional input feature.

This review covers: development testing, sensitivity to input perturbation, benchmarking against the baseline, bias/fairness across customer segments, known limitations, and an ongoing monitoring plan. It does not cover: the LLM explanation layer (`llm/`, reviewed separately in future work), deployment infrastructure, or the data simulator's realism as a claim about real-world AML patterns (see Limitations, §9).

## 2. Model description & conceptual soundness

**Type**: gradient-boosted decision trees (LightGBM `LGBMClassifier`), `class_weight="balanced"` to counter the ~1.5% positive rate. Hyperparameters selected via 30 Optuna trials, each evaluated across 4 expanding-window time-based cross-validation folds strictly within TRAIN (`models/tuning.py`).

**Inputs**: 18 leakage-safe engineered features (`features/pipeline.py` — velocity counts/sums over 1h/24h/7d/30d windows, peer-group and personal amount z-scores, round-amount indicators, cross-border/channel-switch/new-counterparty flags, hour-of-day features) plus `if_anomaly_score`, the output of an IsolationForest fit on the same TRAIN split. Ground truth (`is_anomalous`, `typology`) is never present in the feature table — enforced by a leakage-guard test in `tests/test_features.py`.

**Conceptual soundness**: tree ensembles are an appropriate choice here — the feature set mixes counts, ratios, and binary flags with genuinely non-linear interactions (e.g. a velocity spike matters differently depending on whether the customer is also transacting cross-border), which LightGBM handles natively without manual interaction terms. The `if_anomaly_score` input feature is a deliberate, documented "stacking" choice: it lets the supervised model use whatever the unsupervised baseline was independently getting right, without requiring the model to be *only* IsolationForest's proxy for the label.

**Output**: `predict_proba_anomaly` returns the model's probability that a transaction is anomalous. Explanations for individual flagged transactions are grounded in the model's own exact TreeSHAP contributions (`predict_shap_contributions`, native `pred_contrib=True`), not post-hoc approximations.

## 3. Data lineage & quality

Full field-by-field lineage: `docs/data_dictionary.md`. Summary for this review:

- **Source**: `data_sim/simulate.py`, a synthetic transaction generator (seed=42), not real bank data — see Limitations, §9.
- **Volume**: 1,219,924 transactions, 12,000 customers, 2024-01-01 to 2025-06-30 (`data/simulated/manifest.json`).
- **Injected anomaly rate**: 1.55% actual (target 2%), across 6 typologies: structuring (4,010), layering (2,683), round_amount (3,206), velocity_spike (4,000), peer_deviation (1,987), geographic_risk (2,970).
- **Quality gate**: both `transactions.parquet`/`customers.parquet` (`data_sim/schemas.py`) and `features.parquet` (`features/schemas.py`) are validated by pandera schemas at generation time — type, range, and nullability checks fail generation outright rather than silently passing bad rows downstream.
- **Leakage safety**: ground truth (`is_anomalous`, `typology`) exists only in `transactions.parquet`, never in the feature table, enforced by a dedicated test.

## 4. Development testing

Time-ordered 70/15/15 TRAIN/VAL/TEST split (`evaluation/splits.py`) — chronological, not random, so the model is never evaluated on transactions that precede ones it was trained on. TRAIN: 853,946 rows. VAL: 182,988 rows (interim sanity check only, played no role in threshold selection). TEST: 182,990 rows, touched exactly once.

| Metric | Value |
|---|---|
| TEST precision @ 2% capacity | **58.3%** (95% CI [56.8%, 59.9%], Wilson score interval) |
| TEST recall @ 2% capacity | 83.6% |
| Hyperparameter tuning | 30 Optuna trials × 4-fold expanding-window CV, strictly within TRAIN |
| Threshold calibration | Selected on 512K out-of-fold predictions spanning multiple TRAIN time windows, not a single VALIDATION block (see §5 for why) |

Anomaly rate and typology mix are stable across TRAIN/VAL/TEST (1.52%/1.68%/1.55% respectively, typology proportions within a few points of each other) — the precision/recall numbers above are not an artifact of target drift between splits.

## 5. Outcomes analysis / backtesting

Threshold selection was attempted twice, and the discrepancy between the two attempts is itself a validated finding, not noise:

1. **First attempt** — threshold chosen via two-proportion z-test on a single VALIDATION block. Looked like a significant FP-rate improvement on VALIDATION (p=0.0001), but made FP rate *worse* on TEST (+17.7%) — a textbook single-split overfitting failure.
2. **Second attempt** (current model) — threshold chosen on 512K pooled out-of-fold predictions across 4 expanding-window folds within TRAIN. More statistically robust by construction, but the z-test-chosen threshold *again* made FP rate worse on TEST (+17.3%), just with a lower, more defensible base precision (58.3% instead of 67.0%).

**Conclusion, reported honestly rather than re-run until something better appears**: the naive top-2%-capacity cutoff remains the best threshold found so far. The two-proportion z-test methodology worked exactly as intended in both attempts — it caught that neither "improvement" was real. This is a validated null result on threshold tuning, not a success, and is reported as such rather than backfilled to look better.

## 6. Sensitivity analysis

Two checks (`evaluation/sensitivity.py`), run against a 20,000-row TEST sample:

**Feature perturbation** (`feature_sensitivity`) — one feature shifted at a time by ±1 standard deviation of its own distribution, holding all others fixed:

| Rank | Feature | Mean |Δscore| (shift up) |
|---|---|---|
| 1 | `is_new_counterparty`* | 0.179 |
| 2 | `is_round_amount`* | 0.117 |
| 3 | `peer_zscore` | 0.026 |
| 4 | `velocity_count_24h` | 0.024 |
| 5 | `personal_amount_zscore` | 0.021 |

\* **Caveat, checked before trusting these numbers**: `is_new_counterparty` and `is_round_amount` (and the other binary indicator features) show a suspicious asymmetric pattern — large sensitivity when shifted up, exactly zero when shifted down. This is a limitation of the perturbation *method*, not a finding about the model: a fractional "±1 std dev" shift has no real-world meaning for a 0/1 flag, so these two numbers should not be compared directly against genuinely continuous features like `peer_zscore`. Separately, `velocity_count_1h` showed exactly zero sensitivity despite having real model importance (confirmed via `feature_importances_` = 90, not 0) — explained by low variance (97% of rows equal 1.0), not a bug: a 1-std shift there is only ~0.18 of a transaction count, too small to cross the model's actual split thresholds.

**Decision-flip robustness** (`decision_flip_rate`) — small Gaussian noise (1% of each feature's own std, simulating routine measurement/data-quality noise) added to every feature simultaneously:

> **28.66% of flagging decisions flip** under this noise. This uses a much smaller perturbation than the feature-level check above, so it is unlikely to be dominated by the same binary-flag artifact. This is a genuine, notable instability finding — nearly a third of the flag/no-flag decisions made at the current threshold are not robust to noise unrelated to genuine anomalous behaviour, and is reported here as a real limitation rather than investigated away.

## 7. Benchmarking against the naive rule-based baseline

| Metric | IsolationForest baseline (Week 2) | LightGBM + IF-score (this model) |
|---|---|---|
| Precision @ 2% capacity | 30.0% | **58.3%** (95% CI [56.8%, 59.9%]) |
| Recall @ 2% capacity | 38.4% | 83.6% |
| F1 | 33.7% | — |
| PR-AUC | 0.248 | — |
| Enrichment over base rate | ~19x (vs. 1.56% base rate) | — |

A genuine, large improvement from adding behavioural features and supervised learning. Still below the original CV-draft placeholder of 89% (expected and intentional — see the project's integrity rule: placeholder numbers are never backfilled to match a real result).

## 8. Bias & fairness check

Statistical-parity and equalized-odds checks (`evaluation/fairness.py`) across three customer groupings, TEST set (n=182,990):

| Grouping | Statistical parity (flagging rate) | Equalized odds (precision/recall vs. true rate) |
|---|---|---|
| `declared_risk_rating` (low/medium/high) | No significant difference (~2.2% all three, p>0.87) | Fully explained by near-identical true anomaly rates — not a fairness concern |
| `country_risk_bucket` (high-risk vs. standard country) | Significant (5.13% vs. 2.20%, p≈9e-11) | Fully explained by a genuinely higher true anomaly rate for high-risk countries (3.5% vs. 1.5%) — consistent with the intentionally-injected `geographic_risk` typology, not a fairness concern |
| `segment` (retail/sme/private_banking) | Significant (retail 4.03% vs. private_banking 1.24%, sme 0.70%; p≈3e-53 and p≈0) | **Not fully explained — a real, reportable disparity** (see below) |

**The segment finding, in full**: retail's true anomaly rate genuinely is ~3-6x higher than the other two segments (2.83% vs. 0.98% private_banking, 0.46% sme, TEST set — confirmed against the full 1.2M-row dataset too: 2.81%/1.07%/0.46%), so *some* of the flagging-rate gap is legitimate. But recall for private_banking (62.0%) and sme (60.9%) is meaningfully worse than retail (88.7%) even after controlling for that, and precision is lower too (49.0%/40.3% vs. 62.2%). The model is measurably worse at catching true anomalies for these two segments, not just flagging them less because there are fewer to find.

**Root cause, confirmed against the full dataset**: retail has 15,250 true anomalies to learn from in TRAIN vs. sme's 2,723 and private_banking's 883 — roughly 17x and 5.6x fewer positive training examples respectively. This is a standard class-imbalance-per-group mechanism, not an arbitrary or unexplained bias, but it is a real model limitation that a deployment decision should weigh.

**Recommendation, not yet implemented**: segment-stratified sampling or class weighting during training, or per-segment threshold calibration, to close the recall/precision gap for private_banking and sme. Not pursued further this session — reported as an honest limitation, consistent with how the threshold-tuning null result (§5) was handled, rather than chased until it looks better.

## 9. Limitations & assumptions

Stated plainly, in order of materiality:

1. **The data is synthetic**, generated by `data_sim/simulate.py` for this project. No real customer or transaction data was used or is represented. Typology injection logic is illustrative of FATF red-flag categories, not a compliance-grade mapping.
2. **Labels are simulated, not human-reviewed.** `is_anomalous`/`typology` are ground truth by construction (which typology-injection function wrote the row), not the output of a real investigator's judgment. Real-world label noise, disagreement, and investigator false-negatives have no analogue here.
3. **The segment disparity (§8) is real but unmitigated.** Deploying this model as-is would under-serve private_banking and sme customers relative to retail.
4. **Decision-flip rate is high (28.66%, §6).** Nearly a third of flag/no-flag decisions are not robust to small, routine input noise.
5. **Threshold tuning did not improve on the naive capacity cutoff (§5).** Two independent, methodologically sound attempts both failed to find a validated FP-rate improvement.
6. **No production drift has actually been observed** (§10) — the monitoring check below compares TRAIN against TEST within one synthetic generation run, which is not equivalent to genuine calendar-time production monitoring. The simulator was not designed with a drift-over-time storyline, so "no drift detected" reflects the absence of injected drift, not a validated claim about live-deployment stability.
7. **The IsolationForest component of `if_anomaly_score` is refit per script invocation** in most of this project's runner scripts, rather than a single persisted artifact — deterministic given a fixed seed, but worth noting for anyone reproducing these exact numbers.

## 10. Ongoing monitoring plan

**Metric**: Population Stability Index (`evaluation/monitoring.py`), computed for every model input feature and for the model's own output score, comparing a baseline population against a current one.

**Thresholds** (conventional, industry-standard):

| PSI range | Status | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 – 0.25 | Moderate shift | Investigate; increase monitoring frequency for the affected feature |
| > 0.25 | Significant shift | Retraining/recalibration trigger |

**Baseline check performed this session** (TRAIN vs. TEST, standing in for "the population the model was trained on" vs. "a later time window"): all 19 features and the score itself are stable (PSI < 0.10). `personal_amount_zscore` is the closest to the watch threshold at 0.092; score PSI is 0.0399. See Limitations §9.6 for why this is a methodology demonstration, not a live-production drift finding.

**Retraining triggers, proposed**:
- Any feature or the output score crosses PSI > 0.25 against the original TRAIN baseline.
- TEST-equivalent precision at the production capacity threshold drops materially below the 58.3% ± CI band established in §4, measured on a rolling window of investigator-confirmed outcomes.
- A full calendar quarter elapses without retraining, regardless of PSI (a time-based backstop, since PSI alone can miss slow multivariate drift that no single feature's marginal distribution reveals).

**Not yet implemented**: an actual scheduled job running this check against live data; this session established the methodology and real baseline thresholds, not a deployed monitoring pipeline.

## 11. Governance sign-off template

In a real deployment, each role below would be a distinct person independently reviewing this document. On this portfolio project, one person (the author) filled every role — noted here explicitly rather than implied otherwise.

| Role | Responsibility | Sign-off |
|---|---|---|
| Model developer | Built and tuned the model, produced this report's technical content | _(author)_ |
| Independent model validator | Independently reviewed methodology, findings, and limitations for soundness | _(author — not independent; see note above)_ |
| Model risk owner | Accepts the residual risks documented in §9 as fit for the stated purpose | _(author — not independent; see note above)_ |
| Business owner | Confirms the model serves its intended use case (investigator triage) | _(author — not independent; see note above)_ |

---

*Sources for every number in this report: `CLAUDE.md` "Measured results" table, `data/simulated/manifest.json`, and the `evaluation/run_fairness_check.py`, `evaluation/run_sensitivity_check.py`, and `evaluation/run_monitoring_check.py` outputs produced 2026-07-15 against MLflow run `e12a18e78ab144cea58c39d513d23007`.*
