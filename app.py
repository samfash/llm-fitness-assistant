# Streamlit entry point for the Hugging Face Space (sdk: streamlit, app_file: app.py)
import uuid

import streamlit as st

from core.llm_engine import generate_response
from core.memory import get_user_context, memory_store
from core.rag_with_semantic_retriever import retrieve_context

st.set_page_config(page_title="LLM Fitness Assistant", page_icon="🏋️", layout="centered")
st.title("🏋️ LLM Fitness Assistant")
st.caption("Your AI-powered personal fitness companion: RAG + memory + LLM.")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

user_id = st.session_state.user_id

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input("Ask about workouts, diet or log an activity..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                knowledge = retrieve_context(prompt)
                reply = generate_response(prompt, get_user_context(user_id), knowledge)
                memory_store(user_id, prompt, reply)
            except Exception as e:
                reply = f"Sorry, something went wrong: `{e}`"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.header("Memory")
    if st.checkbox("Show my memory log"):
        st.text(get_user_context(user_id) or "(empty)")
