import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Comeback AI", page_icon="⚡", layout="wide")
st.title("⚡ Comeback AI")
st.caption("Your academic comeback, backed by data.")
st.info("Learning/demo system: synthetic training data, never a basis for real student decisions.")

with st.sidebar:
    st.header("System status")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=2)
        health.raise_for_status()
        st.success("API connected")
    except httpx.HTTPError:
        st.error("API unavailable")
    st.markdown("[Interactive API docs](http://localhost:8000/docs)")
    st.caption("Comeback AI runs locally. Groq is optional.")

with st.form("profile"):
    left, right = st.columns(2)
    with left:
        attendance = st.slider("Attendance (%)", 0, 100, 72)
        assignments = st.slider("Assignments completed (%)", 0, 100, 65)
        grade = st.slider("Average grade (%)", 0, 100, 58)
        study = st.slider("Study hours per week", 0, 40, 7)
        failures = st.number_input("Previous failed courses", 0, 20, 1)
    with right:
        commute = st.slider("Commute (minutes)", 0, 180, 45)
        internet = st.checkbox("Reliable internet", True)
        works = st.checkbox("Works part-time")
        stress = st.checkbox("Reports significant stress")
        help_requested = st.checkbox("Has asked for help")
    submitted = st.form_submit_button("Assess support needs", type="primary")

if submitted:
    payload = {
        "attendance_rate": attendance,
        "assignment_completion": assignments,
        "average_grade": grade,
        "study_hours_weekly": study,
        "previous_failures": failures,
        "commute_minutes": commute,
        "has_internet": internet,
        "works_part_time": works,
        "reports_stress": stress,
        "asked_for_help": help_requested,
    }
    try:
        response = httpx.post(f"{API_URL}/v1/risk", json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.session_state["risk"] = result["risk_level"]
        st.metric(
            "Estimated support-risk",
            result["risk_level"].title(),
            f"{result['risk_probability']:.0%}",
        )
        st.progress(result["risk_probability"], text="Estimated risk probability")
        st.subheader("What influenced this signal")
        for factor in result["top_factors"]:
            icon = "↑" if factor["direction"] == "increases risk" else "↓"
            st.write(f"{icon} **{factor['label']}** — {factor['direction']}")
        st.caption(result["note"])
    except httpx.HTTPError as exc:
        st.error(f"The API is unavailable: {exc}")

st.divider()
st.subheader("Ask the support guide")
question = st.text_input(
    "What would help?", placeholder="How can I catch up on missed assignments?"
)
if st.button("Find grounded guidance") and question:
    try:
        response = httpx.post(
            f"{API_URL}/v1/guidance",
            json={"question": question, "risk_level": st.session_state.get("risk")},
            timeout=30,
        )
        response.raise_for_status()
        guidance = response.json()
        st.markdown(guidance["answer"])
        st.caption(f"Answer mode: {guidance['generated_by']}")
        with st.expander("Sources"):
            for source in guidance["sources"]:
                st.write(
                    f"{source['title']} → {source['section']} (relevance {source['score']:.2f})"
                )
    except httpx.HTTPError as exc:
        st.error(f"The API is unavailable: {exc}")
