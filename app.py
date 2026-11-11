# Streamlit entry point for the Hugging Face Space (sdk: streamlit, app_file: app.py)
import json
import uuid

import streamlit as st

from core import db
from core.agent import chat
from core.rag_with_semantic_retriever import retrieve_context
from core.tools import progress_summary

st.set_page_config(page_title="LLM Fitness Assistant", page_icon="🏋️", layout="wide")
st.title("🏋️ LLM Fitness Assistant")
st.caption("Log workouts and meals in plain English. Targets and progress are computed by tested code, not guessed by the model.")

if "default_user" not in st.session_state:
    st.session_state.default_user = f"guest-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("You")
    user_id = st.text_input("Username (no password, so pick something unique)", st.session_state.default_user).strip()
    days = st.slider("Progress window (days)", 1, 30, 7)
    profile = db.get_profile(user_id)
    st.subheader("Profile")
    st.json(profile) if profile else st.write("Not set yet. Tell me your age, sex, height, weight, activity level and goal.")
    s = progress_summary(user_id, days)
    st.subheader("Progress")
    c1, c2 = st.columns(2)
    c1.metric("Sessions", s["workout_sessions"])
    c2.metric("Volume (kg)", s["total_volume_kg"])
    if s["avg_daily_calories"] is not None:
        st.metric("Avg kcal/day", s["avg_daily_calories"], delta=None if not s["targets"] else f"target {s['targets']['calories']}")
    if s["best_estimated_1rm_kg"]:
        st.write("Best est. 1RM (kg)")
        st.bar_chart(s["best_estimated_1rm_kg"])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for g in m.get("guards", []):
            st.caption(f"🛡️ Safety guardrail: {g['rule'].replace('_', ' ')} ({g['action']})")
        for ev in m.get("events", []):
            with st.expander(f"🔧 {ev['tool']}"):
                st.code(json.dumps({"arguments": ev["arguments"], "result": ev["result"]}, indent=2, default=str), language="json")

if prompt := st.chat_input("e.g. 'Bench 80kg 3x8, then a chicken salad around 450 kcal'"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chat(user_id, prompt, history, knowledge=retrieve_context(prompt))
                reply, events, guards = result.reply, result.tool_events, result.guardrails
            except Exception as e:
                reply, events, guards = f"Sorry, something went wrong: `{e}`", [], []
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply, "events": events, "guards": guards})
    st.rerun()  # refresh sidebar metrics with the newly saved data
