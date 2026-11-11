"""Tool-calling agent loop with an LLM abstraction (so it can be unit-tested with a fake model)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from .guardrails import check_input, check_output
from .llm_engine import HF_TOKEN, REMOTE_MODEL, generate_response
from .tools import execute_tool, tool_specs

log = logging.getLogger(__name__)
MAX_STEPS = 5

SYSTEM_PROMPT = """You are a friendly, motivating personal fitness assistant. Today is {today}.

Rules:
- When the user reports a workout or meal, record it with the matching tool. One tool call per exercise or meal.
- NEVER do calorie, macro, BMR/TDEE or 1RM arithmetic yourself. Use calculate_targets / estimate_1rm / get_progress_summary.
- If a tool says profile fields are missing, ask the user for them. Do not guess them.
- If a tool returns an error, fix the arguments or ask the user; never claim something was saved when it was not.
- Meal macros are your best estimate; say so briefly.
- You are not a doctor. For chest pain, injury, pregnancy, eating-disorder signs or medical conditions, advise seeing a professional instead of giving a plan.
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


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None) -> AssistantTurn: ...


class HFToolClient:
    def __init__(self, model: str = REMOTE_MODEL, token: str | None = HF_TOKEN):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(model=model, token=token, timeout=60)

    def chat(self, messages, tools):
        kwargs = {"messages": messages, "max_tokens": 500, "temperature": 0.3}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        msg = self.client.chat_completion(**kwargs).choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments) for tc in (msg.tool_calls or [])]
        return AssistantTurn(content=msg.content, tool_calls=calls)


@dataclass
class AgentResult:
    reply: str
    tool_events: list[dict] = field(default_factory=list)
    guardrails: list[dict] = field(default_factory=list)  # {"stage", "rule", "action": block|notice|rewrite}


def _gevent(stage: str, rule: str, action: str) -> dict:
    return {"stage": stage, "rule": rule, "action": action}


def _history_user_texts(history: list[dict] | None) -> list[str]:
    return [m["content"] for m in (history or []) if m.get("role") == "user"]


def run_agent(user_id: str, user_input: str, llm: LLMClient, history: list[dict] | None = None,
              knowledge: str = "", max_steps: int = MAX_STEPS, guardrails: bool = True) -> AgentResult:
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
        return AgentResult(reply, events, gevents)

    messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user_input}]
    specs, events = tool_specs(), []

    for step in range(max_steps):
        last = step == max_steps - 1
        turn = llm.chat(messages, None if last else specs)  # last step: force a plain answer
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


def chat(user_id: str, user_input: str, history: list[dict] | None = None, knowledge: str = "") -> AgentResult:
    """App entry point: tool-calling agent via the HF API; plain local generation as a fallback.

    Guardrails are on unless GUARDRAILS=off (used only for A/B evals).
    """
    guard = os.getenv("GUARDRAILS", "on").lower() != "off"
    if HF_TOKEN and not os.getenv("DISABLE_TOOLS"):
        try:
            return run_agent(user_id, user_input, HFToolClient(), history, knowledge, guardrails=guard)
        except Exception as e:
            log.warning("Agent failed (%s); falling back to plain generation", e)

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
    return AgentResult(
        reply + "\n\n_(Logging and tracking need an HF_TOKEN; running in basic chat mode.)_", [], gevents)
