-- Schema for the anomaly detection engine's on-box Postgres (PLAN.md §02, §11).
-- Applied automatically by the postgres image on first container init - see
-- infra/docker-compose.yml.

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    segment TEXT NOT NULL,
    home_country TEXT NOT NULL,
    declared_risk_rating TEXT NOT NULL,
    peer_group TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    "timestamp" TIMESTAMPTZ NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    direction TEXT NOT NULL,
    channel TEXT NOT NULL,
    counterparty_id TEXT,
    counterparty_country TEXT,
    is_cross_border BOOLEAN NOT NULL,
    -- the 18 features.pipeline.FEATURE_COLUMNS values for this transaction,
    -- stored as a blob rather than one column each so the table doesn't need
    -- a migration every time the feature set changes.
    features JSONB NOT NULL,
    anomaly_probability DOUBLE PRECISION NOT NULL,
    is_flagged BOOLEAN NOT NULL,
    -- ground truth - exists only because this is synthetic data (see
    -- CLAUDE.md's integrity rule). Shown for demo/evaluation context, never
    -- read back as a model input.
    is_anomalous BOOLEAN,
    typology TEXT
);

CREATE INDEX IF NOT EXISTS idx_transactions_flagged ON transactions (is_flagged) WHERE is_flagged;
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions (customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions ("timestamp");

-- Precomputed median/MAD per peer_group (segment x home_country, see
-- data_sim/customers.py) - features/peer_deviation.py needs these to compute
-- peer_zscore, but recomputing them live from a full peer group's raw rows
-- (up to ~7,870 customers' worth of transactions) is exactly the kind of
-- heavy per-request work this box can't afford (see CLAUDE.md's Week 7
-- load_data.py incident). Populated once, offline, by api/load_full_history.py.
CREATE TABLE IF NOT EXISTS peer_group_stats (
    peer_group TEXT PRIMARY KEY,
    peer_median DOUBLE PRECISION NOT NULL,
    peer_mad DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS explanations (
    transaction_id TEXT PRIMARY KEY REFERENCES transactions(transaction_id),
    explanation TEXT NOT NULL,
    typology TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    likely_false_positive BOOLEAN NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('llm', 'fallback')),
    fact_check_passed BOOLEAN,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS investigator_feedback (
    id SERIAL PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    verdict TEXT NOT NULL CHECK (verdict IN ('true_positive', 'false_positive', 'needs_review')),
    note TEXT,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
