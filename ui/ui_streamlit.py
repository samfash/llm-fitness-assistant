# ui.py
import streamlit as st
import requests
import json
from datetime import datetime

# -------------------------------
# ⚙️ CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="🏋️‍♂️ LLM Fitness Assistant",
    page_icon="💬",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000/chat"   # ← Replace with your deployed FastAPI endpoint

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
        payload = {"user_input": prompt}
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            reply = response.json().get("response", "No response from model.")
        else:
            reply = f"⚠️ Error {response.status_code}: {response.text}"
    except Exception as e:
        reply = f"❌ Could not connect to API. Details: {e}"

    # Display assistant reply
    st.chat_message("assistant").markdown(reply)

    # Save reply to conversation
    st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------------------
# 📊 OPTIONAL: DISPLAY SUMMARY SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("📈 Progress Overview")
    st.write("Last update:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    st.write("💪 Workouts logged:", len([m for m in st.session_state.messages if 'pushup' in m['content'].lower()]))
    st.write("🍎 Diet entries:", len([m for m in st.session_state.messages if 'calorie' in m['content'].lower()]))
    st.markdown("---")
    st.write("🧠 Powered by LangChain + LoRA + Chroma + FastAPI")

