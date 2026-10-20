# Optional standalone REST API:  uvicorn api.api_server:app --reload
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.llm_engine import generate_response
from core.memory import get_user_context, memory_store
from core.rag_with_semantic_retriever import retrieve_context

app = FastAPI(title="LLM Fitness Assistant API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    user_id: str = "default"
    prompt: str


@app.post("/chat")
def chat(query: Query):
    user_input = query.prompt.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty query received.")
    try:
        knowledge = retrieve_context(user_input)
        reply = generate_response(user_input, get_user_context(query.user_id), knowledge)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
    memory_store(query.user_id, user_input, reply)
    return {"assistant_reply": reply}


@app.get("/")
def root():
    return {"message": "LLM Fitness Assistant API is running"}
