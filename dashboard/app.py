from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import (
    ApiError,
    explain_transaction,
    get_health,
    get_transaction,
    list_feedback,
    list_transactions,
    submit_feedback,
)
from dashboard.config import DashboardConfig

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
