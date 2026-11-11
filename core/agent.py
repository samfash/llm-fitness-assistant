"""Tool-calling agent loop with an LLM abstraction (so it can be unit-tested with a fake model)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from . import usage as usage_store
from .guardrails import check_input, check_output
from .llm_engine import HF_TOKEN, REMOTE_MODEL, generate_response
from .pricing import cost_usd
from .rag.citations import sanitize_citations, valid_ids_from_context
from .tools import execute_tool, tool_specs
from .tracing import content_enabled, hash_id, span
from .usage import estimate_tokens

log = logging.getLogger(__name__)
MAX_STEPS = 5

SYSTEM_PROMPT = """You are a friendly, motivating personal fitness assistant. Today is {today}.

Rules:
- When the user reports a workout, record it with log_workout (one call per exercise).
- NEVER do calorie, macro, BMR/TDEE or 1RM arithmetic yourself. Use calculate_targets / estimate_1rm / get_progress_summary.
- If a tool says profile fields are missing, ask the user for them. Do not guess them.
- If a tool returns an error, fix the arguments or ask the user; never claim something was saved when it was not.
- For meals described in words: call search_food for each food, choose the best match (ask the user if the candidates differ a lot), then call log_foods once with grams or a listed portion. Use log_meal only when the user gives calories/macros themselves or nothing matches (then say the numbers are estimates).
- You are not a doctor. For chest pain, injury, pregnancy, eating-disorder signs or medical conditions, advise seeing a professional instead of giving a plan.
- Background knowledge is numbered [1], [2], ... When you use it, cite it inline like [1]. Only cite numbers you were given. If the knowledge does not cover the question, say you do not have a source for it rather than guessing.
- Be concise.
{knowledge}"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: object  # dict or JSON string, depending on the provider


@dataclass
class AssistantTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None  # {"prompt_tokens", "completion_tokens"} as reported by the provider


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None) -> AssistantTurn: ...


class HFToolClient:
    def __init__(self, model: str = REMOTE_MODEL, token: str | None = HF_TOKEN):
        from huggingface_hub import InferenceClient

        self.model = model
        self.client = InferenceClient(model=model, token=token, timeout=60)

    def chat(self, messages, tools):
        kwargs = {"messages": messages, "max_tokens": 500, "temperature": 0.3}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        out = self.client.chat_completion(**kwargs)
        msg = out.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments) for tc in (msg.tool_calls or [])]
        u, usage = getattr(out, "usage", None), None
        if u is not None and getattr(u, "prompt_tokens", None) is not None:
            usage = {"prompt_tokens": int(u.prompt_tokens), "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0)}
        return AssistantTurn(content=msg.content, tool_calls=calls, usage=usage)


def _empty_usage() -> dict:
    return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None, "estimated": False}


@dataclass
class AgentResult:
    reply: str
    tool_events: list[dict] = field(default_factory=list)
    guardrails: list[dict] = field(default_factory=list)  # {"stage", "rule", "action": block|notice|rewrite}
    usage: dict = field(default_factory=_empty_usage)


class _Meter:
    """Accumulates model usage for one agent run. Cost covers priced models only (None if none were priced)."""

    def __init__(self):
        self.u = _empty_usage()

    def add(self, model: str, prompt: int, completion: int, estimated: bool) -> float | None:
        c = cost_usd(model, prompt, completion)
        self.u["requests"] += 1
        self.u["prompt_tokens"] += prompt
        self.u["completion_tokens"] += completion
        self.u["estimated"] = self.u["estimated"] or estimated
        if c is not None:
            self.u["cost_usd"] = (self.u["cost_usd"] or 0.0) + c
        return c


def _llm_call(llm, messages: list[dict], tools: list[dict] | None, meter: _Meter) -> AssistantTurn:
    model = getattr(llm, "model", type(llm).__name__)
    with span("llm.chat", model=model, offered_tools=len(tools or [])) as sp:
        turn = llm.chat(messages, tools)
        if turn.usage:
            prompt, completion, estimated = turn.usage["prompt_tokens"], turn.usage["completion_tokens"], False
        else:  # provider reported nothing: estimate, and say so
            prompt = estimate_tokens(json.dumps(messages, default=str) + json.dumps(tools or []))
            completion = estimate_tokens((turn.content or "") + json.dumps([c.arguments for c in turn.tool_calls], default=str))
            estimated = True
        cost = meter.add(model, prompt, completion, estimated)
        sp.set(prompt_tokens=prompt, completion_tokens=completion, estimated=estimated, cost_usd=cost,
               tool_calls=len(turn.tool_calls))
        return turn


def _gevent(stage: str, rule: str, action: str) -> dict:
    return {"stage": stage, "rule": rule, "action": action}


def _history_user_texts(history: list[dict] | None) -> list[str]:
    return [m["content"] for m in (history or []) if m.get("role") == "user"]


def _run(user_id: str, user_input: str, llm: LLMClient, history: list[dict] | None, knowledge: str,
         max_steps: int, guardrails: bool, meter: _Meter) -> AgentResult:
    today = date.today().isoformat()
    system_core = SYSTEM_PROMPT.format(today=today, knowledge="")
    gevents: list[dict] = []
    notices: list[str] = []

    if guardrails:
        decision = check_input(user_input, _history_user_texts(history))
        if decision.blocked:  # deterministic short-circuit: the model is never called
            return AgentResult(decision.response, [], [_gevent("input", decision.rule, "block")])
        for rule, text in decision.notices:
            gevents.append(_gevent("input", rule, "notice"))
            notices.append(text)

    system = SYSTEM_PROMPT.format(
        today=today,
        knowledge=f"\nBackground knowledge you may use:\n{knowledge}" if knowledge else "",
    )
    if notices:
        system += "\n\n" + "\n".join(notices)

    def finish(reply: str, events: list[dict]) -> AgentResult:
        if guardrails:
            reply, rule = check_output(reply, system_core)
            if rule:
                gevents.append(_gevent("output", rule, "rewrite"))
            reply, removed = sanitize_citations(reply, valid_ids_from_context(knowledge))
            if removed:  # the model cited a source it was never given
                gevents.append(_gevent("output", "invalid_citation", "sanitize"))
        return AgentResult(reply, events, gevents)

    messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user_input}]
    specs, events = tool_specs(), []

    for step in range(max_steps):
        last = step == max_steps - 1
        turn = _llm_call(llm, messages, None if last else specs, meter)  # last step: force a plain answer
        if not turn.tool_calls:
            return finish((turn.content or "").strip(), events)

        messages.append({
            "role": "assistant", "content": turn.content or "",
            "tool_calls": [{"id": c.id, "type": "function", "function": {
                "name": c.name,
                "arguments": c.arguments if isinstance(c.arguments, str) else json.dumps(c.arguments or {})}}
                for c in turn.tool_calls],
        })
        for c in turn.tool_calls:
            result = execute_tool(user_id, c.name, c.arguments)  # user_id comes from the app, not the model
            events.append({"tool": c.name, "arguments": c.arguments, "result": json.loads(result)})
            messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": result})

    return finish("Sorry, I couldn't finish that request.", events)


def run_agent(user_id: str, user_input: str, llm: LLMClient, history: list[dict] | None = None,
              knowledge: str = "", max_steps: int = MAX_STEPS, guardrails: bool = True) -> AgentResult:
    meter = _Meter()
    with span("agent.run", user=hash_id(user_id), guardrails=guardrails, knowledge_chars=len(knowledge)) as sp:
        result = _run(user_id, user_input, llm, history, knowledge, max_steps, guardrails, meter)
        result.usage = meter.u
        sp.set(tool_calls=len(result.tool_events), steps=meter.u["requests"],
               guardrail_events=[f"{g['stage']}:{g['rule']}:{g['action']}" for g in result.guardrails],
               prompt_tokens=meter.u["prompt_tokens"], completion_tokens=meter.u["completion_tokens"],
               cost_usd=meter.u["cost_usd"], blocked_before_model=meter.u["requests"] == 0, reply_chars=len(result.reply))
        if content_enabled():
            sp.set(input=user_input, output=result.reply)
        return result


def _record_usage(usage_user: str, result: AgentResult) -> None:
    if result.usage["requests"]:
        try:
            usage_store.record(usage_user, result.usage)
        except Exception as e:  # metrics must never break a user request
            log.warning("could not record usage: %s", e)


def chat(user_id: str, user_input: str, history: list[dict] | None = None, knowledge: str = "",
         usage_user: str | None = None) -> AgentResult:
    """App entry point: tool-calling agent via the HF API; plain local generation as a fallback.

    Guardrails are on unless GUARDRAILS=off (used only for A/B evals).
    """
    guard = os.getenv("GUARDRAILS", "on").lower() != "off"
    if HF_TOKEN and not os.getenv("DISABLE_TOOLS"):
        try:
            result = run_agent(user_id, user_input, HFToolClient(), history, knowledge, guardrails=guard)
            _record_usage(usage_user or user_id, result)
            return result
        except Exception as e:
            log.warning("Agent failed (%s); falling back to plain generation", e)

    with span("agent.fallback", user=hash_id(user_id), guardrails=guard) as sp:
        result = _fallback_reply(user_input, history, knowledge, guard)
        sp.set(blocked_before_model=any(g["action"] == "block" for g in result.guardrails),
               guardrail_events=[f"{g['stage']}:{g['rule']}:{g['action']}" for g in result.guardrails])
        return result


def _fallback_reply(user_input: str, history: list[dict] | None, knowledge: str, guard: bool) -> AgentResult:
    """Basic chat without tools (no HF_TOKEN): local model, with the same guardrails."""
    gevents: list[dict] = []
    if guard:
        decision = check_input(user_input, _history_user_texts(history))
        if decision.blocked:
            return AgentResult(decision.response, [], [_gevent("input", decision.rule, "block")])
        gevents = [_gevent("input", rule, "notice") for rule, _ in decision.notices]
    reply = generate_response(user_input, knowledge=knowledge)
    if guard:
        reply, rule = check_output(reply, SYSTEM_PROMPT.format(today=date.today().isoformat(), knowledge=""))
        if rule:
            gevents.append(_gevent("output", rule, "rewrite"))
        reply, removed = sanitize_citations(reply, valid_ids_from_context(knowledge))
        if removed:
            gevents.append(_gevent("output", "invalid_citation", "sanitize"))
    return AgentResult(
        reply + "\n\n_(Logging and tracking need an HF_TOKEN; running in basic chat mode.)_", [], gevents)
