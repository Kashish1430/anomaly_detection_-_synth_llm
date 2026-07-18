from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from dashboard.api_client import (
    ApiError,
    explain_transaction,
    get_health,
    get_transaction,
    list_feedback,
    list_transactions,
    predict_transaction,
    submit_feedback,
)
from dashboard.config import DashboardConfig

# Mirrors data_sim/config.py's actual value sets, so the "score a new
# transaction" form only offers inputs the model was actually trained on.
SEGMENTS = ("retail", "sme", "private_banking")
CHANNELS = ("online", "card", "atm", "branch", "wire")
RISK_RATINGS = ("low", "medium", "high")
COUNTRIES = (
    "GB",
    "US",
    "FR",
    "DE",
    "IE",
    "ES",
    "IN",
    "AE",
    "SG",
    "NL",
    "IT",
    "CA",
    "AU",
    "KY",
    "PA",
    "MT",
    "CY",
)

# The raw-transaction context sent alongside features to /explain - mirrors
# llm/generate_explanations.py's TRANSACTION_COLUMNS, since that's what
# llm/prompts.py formats into the LLM prompt.
TRANSACTION_CONTEXT_FIELDS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "direction",
    "channel",
    "counterparty_id",
    "counterparty_country",
    "is_cross_border",
]

st.set_page_config(page_title="Anomaly Detection Triage", layout="wide")
config = DashboardConfig.from_env()

st.title("Transaction Anomaly Triage")
st.caption(
    "Independent portfolio project on **synthetic** retail-banking data - not a real bank "
    "engagement or regulatory validation. See `docs/model_validation_report.md`."
)

with st.sidebar:
    st.subheader("API connection")
    api_base_url = st.text_input("API base URL", value=config.api_base_url)
    try:
        health = get_health(api_base_url)
        st.success(f"Connected - model run `{health['model_run_id'][:12]}...`")
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

st.subheader("Score a new transaction")
st.caption(
    "Simulates a genuinely new transaction arriving: computes features from this "
    "customer's stored history, scores it, generates an explanation if flagged, and "
    "persists it - it'll appear in the flagged list below if it's flagged."
)
with st.form(key="predict-form"):
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        customer_id = st.text_input("Customer ID", help="An existing customer_id, or a new one")
        amount = st.number_input("Amount", min_value=0.01, value=100.0, step=1.0)
        direction = st.selectbox("Direction", ["debit", "credit"])
    with pcol2:
        channel = st.selectbox("Channel", CHANNELS)
        counterparty_id = st.text_input("Counterparty ID (optional)")
        counterparty_country = st.text_input("Counterparty country (optional, e.g. GB)")
    with pcol3:
        is_cross_border = st.checkbox("Cross-border")
        txn_timestamp = st.text_input(
            "Timestamp (UTC, blank = now)", help="ISO format, e.g. 2026-07-18T14:30:00"
        )

    with st.expander(
        "New customer? Provide these (only used if the customer ID doesn't exist yet)"
    ):
        ncol1, ncol2, ncol3 = st.columns(3)
        new_segment = ncol1.selectbox("Segment", SEGMENTS, key="new-segment")
        new_home_country = ncol2.selectbox("Home country", COUNTRIES, key="new-home-country")
        new_risk_rating = ncol3.selectbox(
            "Declared risk rating", RISK_RATINGS, key="new-risk-rating"
        )

    if st.form_submit_button("Score this transaction"):
        if not customer_id:
            st.error("Customer ID is required.")
        else:
            timestamp = txn_timestamp.strip() or datetime.now(UTC).isoformat()
            raw_transaction = {
                "customer_id": customer_id,
                "timestamp": timestamp,
                "amount": amount,
                "direction": direction,
                "channel": channel,
                "counterparty_id": counterparty_id or None,
                "counterparty_country": counterparty_country or None,
                "is_cross_border": is_cross_border,
                "new_customer_segment": new_segment,
                "new_customer_home_country": new_home_country,
                "new_customer_declared_risk_rating": new_risk_rating,
            }
            try:
                with st.spinner("Computing features, scoring, and explaining if flagged..."):
                    result = predict_transaction(api_base_url, raw_transaction)
                st.session_state["predict_result"] = result
            except ApiError as exc:
                st.error(str(exc))

predict_result = st.session_state.get("predict_result")
if predict_result:
    st.metric("Anomaly probability", f"{predict_result['anomaly_probability']:.1%}")
    if predict_result["is_flagged"]:
        st.warning(f"Flagged as `{predict_result.get('typology') or 'unknown'}` - see below.")
        st.write(predict_result["explanation"])
        badge = "🤖 LLM" if predict_result["source"] == "llm" else "📋 Rule-based fallback"
        st.caption(f"Source: {badge}  |  Transaction ID: `{predict_result['transaction_id']}`")
    else:
        st.success(f"Not flagged. Transaction ID: `{predict_result['transaction_id']}`")

st.subheader("Flagged transactions")
limit = st.slider("Rows to show", min_value=10, max_value=200, value=50, step=10)

try:
    transactions = list_transactions(api_base_url, limit=limit)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if not transactions:
    st.info("No flagged transactions loaded yet - run `python -m api.load_data` first.")
    st.stop()

table_df = pd.DataFrame(
    [
        {
            "transaction_id": t["transaction_id"],
            "timestamp": t["timestamp"],
            "amount": t["amount"],
            "channel": t["channel"],
            "counterparty_country": t["counterparty_country"],
            "anomaly_probability": t["anomaly_probability"],
            "typology (ground truth)": t["typology"],
        }
        for t in transactions
    ]
)
st.dataframe(table_df, use_container_width=True, hide_index=True)

selected_id = st.selectbox("Select a transaction to investigate", table_df["transaction_id"])

if selected_id:
    try:
        detail = get_transaction(api_base_url, selected_id)
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Transaction")
        st.json({k: detail[k] for k in TRANSACTION_CONTEXT_FIELDS})
        st.metric("Anomaly probability", f"{detail['anomaly_probability']:.1%}")
        st.metric("Flagged", "Yes" if detail["is_flagged"] else "No")

    with col2:
        st.markdown("### Engineered features")
        st.json(detail["features"])

    st.markdown("### Explanation")
    st.caption(
        "Falls back to a rule-based template if the LLM call fails. A local Ollama backend "
        "can take a while on CPU - that's slow, not broken."
    )
    explanation_key = f"explanation-{selected_id}"
    if st.button("Generate explanation", key=f"explain-btn-{selected_id}"):
        transaction_context = {k: detail[k] for k in TRANSACTION_CONTEXT_FIELDS}
        with st.spinner("Calling the LLM reasoning layer... (can be slow on a local Ollama model)"):
            try:
                st.session_state[explanation_key] = explain_transaction(
                    api_base_url, selected_id, transaction_context, detail["features"]
                )
            except ApiError as exc:
                st.error(str(exc))

    explanation = st.session_state.get(explanation_key)
    if explanation:
        badge = "🤖 LLM" if explanation["source"] == "llm" else "📋 Rule-based fallback"
        st.write(f"**Source:** {badge}")
        st.write(explanation["explanation"])
        st.write(
            f"**Typology:** {explanation['typology']}  |  "
            f"**Confidence:** {explanation['confidence']:.0%}"
        )
        if explanation["likely_false_positive"]:
            st.info("Model's own assessment: likely false positive.")
        if explanation["fact_check_passed"] is False:
            st.warning(
                "Fact-checker flagged a number mismatch in this explanation - verify before "
                "trusting it."
            )

    st.markdown("### Investigator feedback")
    try:
        existing_feedback = list_feedback(api_base_url, selected_id)
    except ApiError as exc:
        st.error(str(exc))
        existing_feedback = []

    if existing_feedback:
        st.write("**Previously recorded:**")
        for entry in existing_feedback:
            line = f"- `{entry['verdict']}` at {entry['submitted_at']}"
            if entry["note"]:
                line += f" — {entry['note']}"
            st.write(line)
    else:
        st.caption("No feedback recorded yet for this transaction.")

    with st.form(key=f"feedback-form-{selected_id}"):
        verdict = st.radio(
            "Verdict", ["true_positive", "false_positive", "needs_review"], horizontal=True
        )
        note = st.text_area("Note (optional)")
        if st.form_submit_button("Submit feedback"):
            try:
                submit_feedback(api_base_url, selected_id, verdict, note or None)
                st.success("Feedback recorded.")
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
