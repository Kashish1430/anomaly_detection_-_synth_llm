# Data dictionary

What every field in the simulated dataset and the engineered feature table actually means. Three files, all under `data/simulated/` (regenerated via `python -m data_sim.simulate`, not committed — see `.gitignore`):

- `customers.parquet` — one row per synthetic customer
- `transactions.parquet` — one row per transaction, including ground truth
- `features.parquet` — one row per transaction, engineered features only (no ground truth — see "Leakage safety" below)

Schemas are enforced by pandera at generation time: `data_sim/schemas.py` for the first two, `features/schemas.py` for the third. If this document and the schema ever disagree, the schema is correct — update this file.

---

## `customers.parquet`

Built by `data_sim/customers.py`. Constants referenced below live in `data_sim/config.py`.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | str | `CUST0000000`-style unique ID, assigned in generation order. |
| `segment` | str | `retail` / `sme` / `private_banking`, drawn 80% / 15% / 5% (`SEGMENT_WEIGHTS`). Drives transaction frequency and amount scale. |
| `home_country` | str | ISO-ish country code. 82% `GB` (`DOMESTIC_WEIGHT`), most of the remainder spread across `OTHER_COUNTRIES`, a small 3% slice (`HIGH_RISK_WEIGHT`) from `HIGH_RISK_COUNTRIES`. |
| `declared_risk_rating` | str | `low` / `medium` / `high`, drawn 75% / 20% / 5% (`RISK_RATING_WEIGHTS`). Cosmetic/contextual only right now — not yet consumed by any feature or model. |
| `signup_date` | datetime | Backdated 0–3 years before `start_date`. Not currently used downstream. |
| `relative_monthly_txn_rate` | float | This customer's personal transaction-frequency multiplier: `segment base rate x per-customer lognormal jitter`. Feeds the Poisson draw for how many transactions they get in `data_sim/transactions.py`; not itself a modeling feature. |
| `amount_mu`, `amount_sigma` | float | This customer's personal lognormal amount parameters — segment baseline (`SEGMENT_AMOUNT_PARAMS`) plus a per-customer offset, so two retail customers don't have identical "typical" spend. Not modeling features directly; they're what *generates* the `amount` column, and what `peer_deviation.py` and `behavioral.py` are implicitly trying to (re)discover from the transaction data alone. |
| `peer_group` | str | `{segment}_{home_country}`, e.g. `retail_GB`. The grouping key `peer_deviation.py` computes population statistics over. |

## `transactions.parquet`

Built by `data_sim/transactions.py` (the "normal" rows) plus `data_sim/typologies.py` (the injected anomalous rows), concatenated and time-sorted.

| Column | Type | Meaning |
|---|---|---|
| `transaction_id` | str | `TXN00000000`-style unique ID, assigned after final sort — i.e. it does *not* reveal generation order or customer. |
| `customer_id` | str | FK to `customers.parquet`. |
| `timestamp` | datetime | Transaction time. Day is uniform across the configured date range; hour is `N(13, 4)` clipped to `[0, 24)` — a soft business-hours bias, not a hard cutoff. |
| `amount` | float | Transaction value, always positive (this dataset doesn't net debits/credits into signed amounts — see `direction`). |
| `direction` | str | `debit` (money out) or `credit` (money in). Probability depends on `channel` (`CHANNEL_DEBIT_PROB`) — e.g. `card` is 98% debit, `online` is closer to 55/45. |
| `channel` | str | `online` / `card` / `atm` / `branch` / `wire`. Distribution depends on `segment` (`SEGMENT_CHANNEL_PROBS`) — e.g. `private_banking` skews `wire`, `retail` skews `online`/`card`. |
| `counterparty_id` | str | `CPTY000000`-style ID from a shared pool of 6,000 (`N_COUNTERPARTIES`). 85% of the time it's one of the customer's own "regular" counterparties (~80% same-country, ~20% not); 15% of the time it's a fresh pick from the whole pool — this is what makes `is_new_counterparty` a meaningful feature. |
| `counterparty_country` | str | The counterparty's country, same distribution logic as `customers.home_country`. |
| `is_cross_border` | bool | `counterparty_country != customer's home_country`. Baseline rate is ~10% by construction (see the ADR-adjacent fix in `data_sim/transactions.py` — this was ~31% before regular counterparties were biased toward the customer's own country). |
| `is_anomalous` | bool | **Ground truth.** True only for rows injected by `data_sim/typologies.py`. Stands in for a human investigator's label — see the caveat in `PLAN.md` §03. |
| `typology` | str, nullable | **Ground truth.** One of the six values in the typology reference below, or `null` for normal transactions. |

**Ground truth is not a feature.** `is_anomalous`/`typology` exist only in `transactions.parquet`, never in `features.parquet` — `features/pipeline.py` builds the feature table from `data_sim` output without ever touching these two columns. See the leakage-guard test in `tests/test_features.py`.

### Typology reference

Each row of ground truth maps to one construction rule in `data_sim/typologies.py`, loosely aligned to a FATF red-flag category (illustrative for this project, not a compliance mapping — see `PLAN.md` §00 and §08 for the honesty caveat on that).

| `typology` value | What it represents | How it's built |
|---|---|---|
| `structuring` | Splitting a large sum into pieces just under a reporting threshold to avoid triggering it. | 3–5 wire transactions, 85–99.5% of `reporting_threshold` (£10,000), within a 48h window, to a manufactured one-off counterparty. |
| `layering` | Moving funds in and rapidly back out to obscure the trail. | One large credit, followed within 1–36h by 2–4 debits that together move 75–98% of it back out, to random counterparties. |
| `round_amount` | Suspiciously frequent exact-round-number payments. | 3–5 transactions per event, amounts that are exact multiples of 500/1,000/2,000/5,000. |
| `velocity_spike` | A burst of activity far above the customer's own normal pace. | 8 transactions crammed into a random 72h window, amounts drawn from the customer's *own* normal distribution (deliberately not amount-extreme — see the note below on why this typology is hard to catch). |
| `peer_deviation` | Behaviour that doesn't match the customer's declared segment. | 1–2 transactions sized like a segment two tiers up (e.g. a `retail` customer transacting at `private_banking` scale). |
| `geographic_risk` | Activity linked to a higher-risk jurisdiction. | 1–2 transactions to/from a `HIGH_RISK_COUNTRIES` counterparty — always cross-border by construction. |

---

## `features.parquet`

Built by `features/pipeline.py`, one row per `transaction_id`, columns listed in `FEATURE_COLUMNS`. Every "customer's own baseline" feature is computed from **prior transactions only** (`features/utils.py:expanding_prior_mean_std`) — a customer's first transaction has no history, and gets a neutral fill (documented per-feature below), never a leaked future value.

| Column | Type | Meaning | Leakage safety |
|---|---|---|---|
| `velocity_count_1h` / `_24h` / `_7d` / `_30d` | float | Rolling transaction count for this customer in the trailing window, **including** the current transaction (right-closed window). | Time-based rolling window — safe. Not personalized (see limitation below). |
| `velocity_sum_1h` / `_24h` / `_7d` / `_30d` | float | Same windows, summed `amount` instead of counted. | Same as above. |
| `amount_to_avg_ratio` | float | `amount / (customer's own prior mean amount)`. First transaction → `1.0` (neutral). | Prior-only — safe. |
| `personal_amount_zscore` | float | `(amount - prior mean) / prior std`, std floored at `max(5% of prior mean, 1.0)` to stop a near-zero denominator exploding the z-score (see the bug note in `features/utils.py:safe_zscore` and `features/behavioral.py`). | Prior-only — safe. |
| `is_new_counterparty` | bool | True the first time this exact `(customer_id, counterparty_id)` pair appears in time order. | Prior-only by construction (`duplicated(keep="first")` on a time-sorted frame) — safe. |
| `is_channel_switch` | bool | True if `channel` differs from this customer's immediately preceding transaction. False for a customer's first transaction. | Prior-only — safe. |
| `is_round_amount` | bool | `amount % 100 == 0`. | Deterministic function of the row's own amount — no leakage possible. |
| `round_amount_count_30d` | float | Rolling 30-day count of `is_round_amount` transactions for this customer, current row included. | Time-based rolling window — safe. |
| `peer_zscore` | float | Robust z-score of `amount` against the customer's `peer_group` (`(amount - group median) / (1.4826 x group MAD)`). | **Not point-in-time safe** — see limitation below. |
| `hour_of_day` | int | `timestamp.hour`, 0–23. | Deterministic — safe. |
| `hour_zscore` | float | Same prior-only z-score pattern as `personal_amount_zscore`, applied to `hour_of_day` (std floored at 1.0 hour). | Prior-only — safe. |
| `is_cross_border` | bool | Passed through from `transactions.is_cross_border`. | Deterministic — safe. |

### Known limitations (not bugs — tracked here so they're not rediscovered from scratch)

- **`peer_zscore` uses whole-dataset population statistics**, including transactions that happen after the one being scored. Flagged in `features/peer_deviation.py` at the point it's computed; the fix (per-fold peer stats from the training period only) is scoped into Week 3 alongside time-based CV (`PLAN.md` §07), not fixed yet.
- **Velocity features are not personalized.** `velocity_count_*`/`velocity_sum_*` are raw rolling counts/sums, not compared against the customer's *own* typical velocity the way `personal_amount_zscore` compares amount against the customer's own baseline. Empirically, this is a big part of why the Week 2 IsolationForest baseline catches `structuring` well (91.7% recall — it happens to also look like a personal amount outlier) but `velocity_spike` and `layering` poorly (2.2% and 0.8% recall) — an SME customer's naturally higher transaction count drowns out a retail customer's genuine personal spike in the raw feature. Not fixed yet; noted here as a concrete, evidence-backed candidate for Week 3, not acted on without a separate discussion.
