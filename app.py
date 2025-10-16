# app.py
import streamlit as st
import requests
import uuid

# 🌍 Backend URL (when hosted separately)
# For Spaces, you can call your FastAPI endpoint if hosted elsewhere.
# For now, we’ll directly call your local backend functions if available.

from core.llm_engine import generate_response
from core.rag_with_semantic_retriever import retrieve_context
from core.memory import memory_store, get_user_context

st.set_page_config(page_title="🏋️‍♂️ LLM Fitness Assistant", layout="centered")

# -------------------------------
# 🏋️‍♂️ Header
# -------------------------------
st.title("🏋️‍♂️ LLM Fitness Assistant")
st.write("Your AI-powered personal fitness companion built with LangChain + LoRA + Chroma.")

# -------------------------------
# 🧠 User session handling
# -------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

# -------------------------------
# 💬 Chat Interface
# -------------------------------
st.subheader("💬 Chat with your assistant")
user_input = st.text_area("Ask about your fitness, diet, or workouts:")

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("💭 Thinking..."):
            user_context = get_user_context(user_id)
            retrieved_knowledge = retrieve_context(user_input)

            final_prompt = f"""
            You are a personal fitness assistant.
            Use user’s past logs and external data to respond helpfully.

            Past context: {user_context}
            Retrieved data: {retrieved_knowledge}
            User input: {user_input}
            """

            response = generate_response(final_prompt)
            memory_store(user_id, user_input, response)
            st.success(response)

# -------------------------------
# 🧾 Show Chat Memory
# -------------------------------
if st.checkbox("Show my fitness memory log"):
    st.write(get_user_context(user_id))
