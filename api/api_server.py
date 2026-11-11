# Standalone REST API:  uvicorn api.api_server:app --host 0.0.0.0 --port 8000
#
# Authentication: every endpoint except /health needs  Authorization: Bearer <api key>.
# The user identity comes from the KEY, never from the request body.
# Create keys with:  python -m core.auth create <user_id>
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import auth, db, usage
from core.agent import chat
from core.rag import cited_ids
from core.rag_with_semantic_retriever import retrieve_with_sources
from core.ratelimit import RateLimiter
from core.tools import progress_summary

log = logging.getLogger("api")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class Query(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=10)


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(UTC)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((nxt - now).total_seconds()))


def current_user(authorization: str | None = Header(default=None)) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Missing API key. Send 'Authorization: Bearer <key>'.", headers={"WWW-Authenticate": "Bearer"})
    user = auth.verify_key(token.strip())
    if not user:
        raise HTTPException(401, "Invalid or revoked API key.", headers={"WWW-Authenticate": "Bearer"})
    return user


def limited_user(request: Request, response: Response, user: str = Depends(current_user)) -> str:
    decision = request.app.state.limiter.check(user)
    if not decision.allowed:
        raise HTTPException(429, "Rate limit exceeded. Slow down and retry shortly.",
                            headers={"Retry-After": str(max(1, int(decision.retry_after + 0.999)))})
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    return user


def create_app(limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="LLM Fitness Assistant API", version="3.0")
    app.state.limiter = limiter or RateLimiter.from_env()
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:  # off by default; the Streamlit UI calls this server-side and needs no CORS
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/chat")
    def chat_endpoint(body: Query, user: str = Depends(limited_user)):
        text = body.prompt.strip()
        if not text:
            raise HTTPException(400, "Empty query received.")
        ok, used = usage.within_budget(user)
        if not ok:
            raise HTTPException(429, f"Daily token budget reached ({used['used_tokens']}/{used['budget_tokens']}). Resets at 00:00 UTC.",
                                headers={"Retry-After": str(_seconds_until_utc_midnight())})
        try:
            knowledge, sources = retrieve_with_sources(text)
            result = chat(user, text, history=[m.model_dump() for m in body.history], knowledge=knowledge)
        except Exception:
            log.exception("chat failed")  # details go to the server log, not to the client
            raise HTTPException(500, "Internal error. Please try again.") from None
        cited = cited_ids(result.reply)
        return {"assistant_reply": result.reply, "tool_events": result.tool_events, "guardrails": result.guardrails,
                "sources": [x for x in sources if x["n"] in cited], "usage": result.usage}

    @app.get("/profile")
    def profile(user: str = Depends(limited_user)):
        return db.get_profile(user)

    @app.get("/progress")
    def progress(days: int = 7, user: str = Depends(limited_user)):
        return progress_summary(user, max(1, min(days, 90)))

    @app.get("/usage")
    def my_usage(user: str = Depends(limited_user)):
        _, used = usage.within_budget(user)
        return {"day": usage.today(), **used}

    return app


app = create_app()
