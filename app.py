# Streamlit entry point for the Hugging Face Space (sdk: streamlit, app_file: app.py)
import json
import os
import uuid

import streamlit as st

from core import auth, db, usage
from core.agent import chat
from core.foods import ATTRIBUTION
from core.rag import cited_ids
from core.rag_with_semantic_retriever import retrieve_with_sources
from core.ratelimit import RateLimiter
from core.tools import progress_summary

st.set_page_config(page_title="LLM Fitness Assistant", page_icon="🏋️", layout="wide")
st.title("🏋️ LLM Fitness Assistant")
st.caption("Log workouts and meals in plain English. Nutrition comes from USDA data and every number is computed by tested code, not guessed by the model.")

ALLOW_GUEST = os.getenv("ALLOW_GUEST", "1") == "1"


@st.cache_resource
def get_limiter() -> RateLimiter:
    return RateLimiter.from_env()


if "guest_id" not in st.session_state:
    st.session_state.guest_id = f"guest-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- identity: an API key (see `python -m core.auth create`), or an optional shared-budget guest session ----
with st.sidebar:
    st.header("Account")
    key = st.text_input("API key", type="password", help="Create one with: python -m core.auth create <user>")
    if key:
        user_id = auth.verify_key(key.strip())
        if not user_id:
            st.error("Invalid or revoked API key.")
            st.stop()
        guest = False
        st.success(f"Signed in as {user_id}")
    elif ALLOW_GUEST:
        user_id, guest = st.session_state.guest_id, True
        st.info("Guest session: data is temporary and the demo budget is shared with other guests.")
    else:
        st.warning("Enter an API key to continue.")
        st.stop()

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
    ok, used = usage.within_budget(user_id, guest=guest)
    if used["budget_tokens"]:
        st.subheader("Today's usage")
        st.progress(min(1.0, used["used_tokens"] / used["budget_tokens"]),
                    text=f"{used['used_tokens']:,} / {used['budget_tokens']:,} tokens" + (" (shared)" if guest else ""))
    st.caption(f"Nutrition data: {ATTRIBUTION}. Source: U.S. Department of Agriculture, Agricultural Research Service. FoodData Central.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for g in m.get("guards", []):
            st.caption(f"🛡️ Safety guardrail: {g['rule'].replace('_', ' ')} ({g['action']})")
        used_sources = [src for src in m.get("sources", []) if src["n"] in cited_ids(m["content"])]
        if used_sources:
            with st.expander(f"📚 Sources ({len(used_sources)})"):
                for src in used_sources:
                    st.markdown(f"**[{src['n']}] {src['title']}**  \n_{src['source']}_")
        for ev in m.get("events", []):
            with st.expander(f"🔧 {ev['tool']}"):
                st.code(json.dumps({"arguments": ev["arguments"], "result": ev["result"]}, indent=2, default=str), language="json")

if prompt := st.chat_input("e.g. 'Bench 80kg 3x8, then two eggs and a slice of toast'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    reply, events, guards, sources = "", [], [], []
    decision = get_limiter().check(user_id)
    if not decision.allowed:
        reply = f"You're sending messages too quickly. Please wait about {int(decision.retry_after) + 1} seconds."
    elif not ok:
        reply = "The daily token budget has been reached. It resets at 00:00 UTC."
    else:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-11:-1]]
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                knowledge, sources = retrieve_with_sources(prompt)
                result = chat(user_id, prompt, history, knowledge=knowledge,
                              usage_user=usage.GUEST_BUCKET if guest else None)
                reply, events, guards = result.reply, result.tool_events, result.guardrails
            except Exception:
                reply, sources = "Sorry, something went wrong. Please try again.", []
    st.session_state.messages.append({"role": "assistant", "content": reply, "events": events, "guards": guards,
                                      "sources": sources})
    st.rerun()  # refresh the sidebar with newly saved data and usage
