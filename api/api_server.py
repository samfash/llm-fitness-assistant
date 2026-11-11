# Optional standalone REST API:  uvicorn api.api_server:app --reload
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import db
from core.agent import chat
from core.rag_with_semantic_retriever import retrieve_context
from core.tools import progress_summary

app = FastAPI(title="LLM Fitness Assistant API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    user_id: str = "default"
    prompt: str


@app.post("/chat")
def chat_endpoint(query: Query):
    text = query.prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty query received.")
    try:
        result = chat(query.user_id, text, knowledge=retrieve_context(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}") from e
    return {"assistant_reply": result.reply, "tool_events": result.tool_events, "guardrails": result.guardrails}


@app.get("/profile/{user_id}")
def profile(user_id: str):
    return db.get_profile(user_id)


@app.get("/progress/{user_id}")
def progress(user_id: str, days: int = 7):
    return progress_summary(user_id, max(1, min(days, 90)))


@app.get("/health")
def health():
    return {"status": "ok"}
