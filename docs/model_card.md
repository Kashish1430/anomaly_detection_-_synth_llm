# Model card

**LLM-Augmented Transaction Anomaly Detection Engine — LightGBM anomaly-scoring model**

> Independent portfolio project on synthetic data. No real bank, no real customer data, no real regulator. Structured like a real model card because that's the practice being demonstrated, not because this describes a production system. See `docs/model_validation_report.md` for the full SR 11-7 / PRA SS1/23-style validation and every underlying statistical test.

This is the short, at-a-glance companion to the validation report above — written for someone (a recruiter, an interviewer) deciding in two minutes whether the project is real and what it does, not for someone auditing methodology. Every number below is one already measured and logged in `CLAUDE.md`'s "Measured results" table or the validation report; nothing here is invented for readability.

## Model details

- **Type**: LightGBM gradient-boosted classifier (`LGBMClassifier`, `class_weight="balanced"`), stacked on top of an IsolationForest anomaly score fit on the same training split.
- **Version**: MLflow run `e12a18e78ab144cea58c39d513d23007`, tuned via 30 Optuna trials × 4-fold expanding-window time-based cross-validation.
- **Inputs**: 18 leakage-safe engineered behavioural features (velocity, peer-group deviation, round-amount, cross-border/channel-switch/new-counterparty flags) plus the IsolationForest score — `docs/data_dictionary.md` has the field-by-field definitions.
- **Output**: a single probability that a transaction is anomalous, used to rank a fixed-capacity investigator review queue (top 2% of volume).
- **Built by**: one person (the author), for a portfolio project — see the governance section of the validation report for why every sign-off role in a real deployment is the same person here.

## Intended use

- **Primary use case**: rank retail-banking transactions by anomaly risk so a capacity-constrained investigator queue reviews the highest-risk 2% first, with a Claude-generated natural-language explanation attached to each flagged transaction for triage.
- **Primary intended "users"**: this repo's own API/dashboard (`api/`, `dashboard/`) and anyone evaluating the project as a CV work sample.
- **Out of scope**: any real transaction-monitoring decision, any real regulatory submission, any use beyond this project's own synthetic dataset. The model has never seen real transaction data and makes no claim to generalize to it.

## Data

- **Training/eval data**: `data_sim/simulate.py`'s synthetic generator (seed=42) — 1,219,924 transactions, 12,000 customers, 6 injected AML typologies mapped to FATF red-flag categories. Full lineage: `docs/data_dictionary.md`.
- **Split**: time-ordered 70/15/15 TRAIN/VAL/TEST (`evaluation/splits.py`), never randomized, so the model is never evaluated on transactions chronologically before ones it trained on.
- **Ground truth**: known by construction (which typology-injection function wrote the row), not human-labelled — see Limitations below.

## Quantitative performance

| Metric | IsolationForest baseline | This model (LightGBM + IF-score) |
|---|---|---|
| Precision @ 2% capacity | 30.0% | **58.3%** (95% CI [56.8%, 59.9%]) |
| Recall @ 2% capacity | 38.4% | 83.6% |
| Enrichment over base rate | ~19x | — |

- **False-positive alerts at fixed 2% capacity**: **48.5% fewer** false positives than the baseline, at the same fixed 2% review capacity (29.1% baseline precision -> 63.5% tuned precision, both measured on the identical TEST split — `evaluation/run_effort_reduction_check.py`). This is the real baseline-vs-tuned-model jump, distinct from a separate two-proportion z-test attempt to tune the threshold *further* below the naive top-2%-capacity cutoff, which found no statistically validated additional improvement (a reported null result, validation report §5).
- **Manual review effort reduction**: **70.9% less** review volume needed for the same detection outcome — the tuned model matches the baseline's recall (37.5% @ 2% capacity) after reviewing only the top 0.58% of transactions by score, instead of 2% (`evaluation/run_effort_reduction_check.py`).
- **Segment fairness**: retail/sme/private_banking show a real, unmitigated recall/precision gap explained by ~5-17x fewer positive training examples for sme/private_banking, not arbitrary bias — full detail and recommendation in the validation report §8.
- **Robustness**: 28.66% of flag/no-flag decisions flip under 1% feature noise — a real, reported instability, not investigated away this session.

## Ethical considerations

- Built entirely on synthetic, injected-typology data — there is no real customer whose transactions were monitored, scored, or exposed.
- The segment disparity above is a documented, unmitigated limitation, not a resolved fairness guarantee; deploying this as-is would under-serve sme/private_banking customers relative to retail.
- The LLM explanation layer (`llm/`) includes a rule-based fallback and a fact-checker specifically because an ungrounded LLM narrative attached to a real flagging decision would be a real harm in production — not relevant to this synthetic demo's stakes, but designed as if it were.

## Caveats & recommendations

Full list, in order of materiality: `docs/model_validation_report.md` §9. Headlines: labels are simulated rather than human-reviewed; the segment disparity is real and unmitigated; the 28.66% decision-flip rate means a meaningful share of flags aren't robust to routine noise; no live production drift has actually been observed (the PSI check compares TRAIN against TEST within one synthetic run, not real calendar time).

---

*Sources: `CLAUDE.md` "Measured results" table, `docs/model_validation_report.md`, `data/simulated/manifest.json`.*
