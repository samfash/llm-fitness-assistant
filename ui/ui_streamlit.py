# ui.py

import os

import requests
import streamlit as st

# -------------------------------
# ⚙️ CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="🏋️‍♂️ LLM Fitness Assistant",
    page_icon="💬",
    layout="centered",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
API_KEY = os.getenv("API_KEY", "")  # create with: python -m core.auth create <user>

# -------------------------------
# 🎯 SESSION STATE INITIALIZATION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 🏷️ HEADER
# -------------------------------
st.title("🏋️‍♂️ LLM Fitness Assistant")
st.caption("Your personalized AI coach for workouts, diet, and progress tracking.")

# -------------------------------
# 💬 CHAT INTERFACE
# -------------------------------
# Display conversation history
for chat in st.session_state.messages:
    role = chat["role"]
    message = chat["content"]
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)

# User input box
if prompt := st.chat_input("Type your question or log an activity (e.g. 'I did pushups today')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display immediately
    st.chat_message("user").markdown(prompt)

    # -------------------------------
    # 🔗 SEND REQUEST TO FASTAPI BACKEND
    # -------------------------------
    try:
        payload = {"prompt": prompt, "history": st.session_state.messages[-11:-1]}
        response = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
        if response.status_code == 200:
            reply = response.json().get("assistant_reply", "No response from model.")
        else:
            reply = f"⚠️ Error {response.status_code}: {response.text}"
    except Exception as e:
        reply = f"❌ Could not connect to API. Details: {e}"

    # Display assistant reply
    st.chat_message("assistant").markdown(reply)

    # Save reply to conversation
    st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------------------
# 📊 SIDEBAR: real progress from the API (nothing is faked client-side)
# -------------------------------
with st.sidebar:
    st.header("📈 Progress overview")
    try:
        base = API_URL.rsplit("/", 1)[0]
        p = requests.get(f"{base}/progress", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
        if p.status_code == 200:
            data = p.json()
            st.metric("Workout sessions (7 days)", data["workout_sessions"])
            st.metric("Total volume (kg)", data["total_volume_kg"])
            if data["avg_daily_calories"] is not None:
                st.metric("Avg kcal/day", data["avg_daily_calories"])
        else:
            st.caption(f"Progress unavailable ({p.status_code}). Is API_KEY set?")
    except Exception:
        st.caption("API not reachable.")
    st.markdown("---")
    st.caption("Nutrition data: USDA FoodData Central (public domain, CC0).")
