"""Tool registry: schemas shown to the LLM + validated handlers.

Security note: `user_id` is injected by the application, never taken from model output,
so a confused or prompt-injected model cannot read or write another user's data.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

from pydantic import BaseModel, ValidationError

from . import db
from .calculators import compute_targets, epley_1rm
from .foods import ATTRIBUTION, SOURCE_NAMES, Food, Portion, get_food_db
from .schemas import (
    FoodSearchRequest,
    LogFoodsRequest,
    MealEntry,
    NoArgs,
    OneRMRequest,
    Profile,
    ProgressRequest,
    WorkoutEntry,
)
from .tracing import content_enabled, span


def progress_summary(user_id: str, days: int = 7) -> dict:
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    workouts, meals = db.workouts_since(user_id, since), db.meals_since(user_id, since)

    best: dict[str, float] = {}
    for w in workouts:
        if w["weight_kg"] and w["weight_kg"] > 0:
            est = epley_1rm(w["weight_kg"], w["reps"])
            best[w["exercise"]] = max(best.get(w["exercise"], 0), est)

    per_day = defaultdict(lambda: {"calories": 0.0, "protein_g": 0.0})
    for m in meals:
        per_day[m["day"]]["calories"] += m["calories"] or 0
        per_day[m["day"]]["protein_g"] += m["protein_g"] or 0
    n = len(per_day)

    out = {
        "window_days": days,
        "workout_sessions": len({w["day"] for w in workouts}),
        "total_volume_kg": round(sum(w["sets"] * w["reps"] * w["weight_kg"] for w in workouts)),
        "best_estimated_1rm_kg": {k: round(v, 1) for k, v in best.items()},
        "days_with_meals_logged": n,
        "avg_daily_calories": round(sum(d["calories"] for d in per_day.values()) / n) if n else None,
        "avg_daily_protein_g": round(sum(d["protein_g"] for d in per_day.values()) / n) if n else None,
    }
    try:
        out["targets"] = compute_targets(Profile(**db.get_profile(user_id)))
    except (ValueError, ValidationError):
        out["targets"] = None  # profile incomplete
    return out


# ---- handlers: (user_id, validated_model) -> dict ---------------------------------------------
def _update_profile(uid: str, a: Profile) -> dict:
    return {"saved": True, "profile": db.save_profile(uid, a.model_dump(exclude_none=True))}


def _log_workout(uid: str, a: WorkoutEntry) -> dict:
    return {"saved": True, "id": db.add_workout(uid, a.model_dump()), "logged": a.exercise}


def _log_meal(uid: str, a: MealEntry) -> dict:
    return {"saved": True, "id": db.add_meal(uid, a.model_dump()), "note": "macros are the caller's estimates"}


def _targets(uid: str, a: NoArgs) -> dict:
    return compute_targets(Profile(**db.get_profile(uid)))


def _progress(uid: str, a: ProgressRequest) -> dict:
    return progress_summary(uid, a.days)


def _one_rm(uid: str, a: OneRMRequest) -> dict:
    return {"estimated_1rm_kg": round(epley_1rm(a.weight_kg, a.reps), 1), "formula": "Epley"}


def _norm_label(label: str) -> str:
    label = label.lower().strip()
    return label[2:] if label.startswith("1 ") else label  # "1 large" and "large" mean the same


def resolve_portion(food: Food, label: str) -> Portion:
    want = _norm_label(label)
    exact = [p for p in food.portions if _norm_label(p.label) == want]
    if len(exact) == 1:
        return exact[0]
    near = exact or [p for p in food.portions if want in _norm_label(p.label)]
    if len(near) == 1:
        return near[0]
    available = [p.label for p in food.portions][:10]
    raise ValueError(f"Portion '{label}' is {'ambiguous' if near else 'not available'} for '{food.description}'. "
                     f"Use grams or one of: {available}")


def _search_food(uid: str, a: FoodSearchRequest) -> dict:
    fdb = get_food_db()
    if not len(fdb):
        raise ValueError("The food database is not installed on this server; use log_meal with the user's numbers.")
    results = fdb.search(a.query, a.limit)
    if not results:
        return {"results": [], "hint": "No match. Try a simpler or more generic name, or use log_meal if the user gave calories."}
    return {"results": [{"fdc_id": f.fdc_id, "description": f.description, "category": f.category,
                         "source": SOURCE_NAMES.get(f.source, f.source), "per_100g": f.per_100g(),
                         "portions": [{"label": p.label, "grams": p.grams} for p in f.portions[:6]]} for f in results],
            "attribution": ATTRIBUTION}


MAX_MEAL_KCAL = 5000


def _log_foods(uid: str, a: LogFoodsRequest) -> dict:
    fdb = get_food_db()
    lines = []
    for it in a.items:
        food = fdb.get(it.fdc_id)
        if food is None:
            raise ValueError(f"Unknown fdc_id {it.fdc_id}; call search_food first and use an id from its results.")
        grams = it.grams if it.grams is not None else resolve_portion(food, it.portion).grams * it.quantity
        if grams > 3000:
            raise ValueError(f"{grams:g} g of '{food.description}' looks wrong; check the quantity and units.")
        lines.append({"fdc_id": food.fdc_id, "name": food.description, "grams": round(grams, 1),
                      **food.macros_for(grams)})
    totals = {k: round(sum(x[k] for x in lines), 1) for k in ("calories", "protein_g", "carbs_g", "fat_g")}
    if totals["calories"] > MAX_MEAL_KCAL:
        raise ValueError(f"Meal total of {totals['calories']:g} kcal looks too large; check quantities and units.")
    name = a.meal or ", ".join(x["name"].split(",")[0] for x in lines)[:190]
    entry = MealEntry(description=name, date=a.date, **totals)
    meal_id = db.add_meal(uid, {**entry.model_dump(), "items": lines})
    return {"saved": True, "id": meal_id, "items": lines, "totals": totals, "source": ATTRIBUTION}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    model: type[BaseModel]
    handler: Callable[[str, BaseModel], dict]


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("update_profile", "Save or update the user's profile (age, sex, height, weight, activity, goal, equipment, injuries). Only pass fields the user stated.", Profile, _update_profile),
    Tool("log_workout", "Record a completed exercise (one exercise per call).", WorkoutEntry, _log_workout),
    Tool("search_food", "Search the USDA food database by name. Use simple generic names (e.g. 'egg', 'rice cooked', "
         "'bread toasted'). Returns candidates with per-100 g macros and portion sizes. Call once per food. If the "
         "candidates differ a lot in calories, ask the user which one they ate.", FoodSearchRequest, _search_food),
    Tool("log_foods", "Record a meal from USDA foods: pass each food's fdc_id from search_food plus grams, or a portion "
         "label exactly as listed with a quantity (e.g. portion '1 large', quantity 2). Calories and macros are "
         "calculated for you.", LogFoodsRequest, _log_foods),
    Tool("log_meal", "Record a meal using calories/macros the user stated themselves, or when no USDA match exists. "
         "Prefer search_food + log_foods for foods described in words.", MealEntry, _log_meal),
    Tool("calculate_targets", "Compute BMR, TDEE and daily calorie/macro targets from the saved profile.", NoArgs, _targets),
    Tool("get_progress_summary", "Summarise recent workouts and nutrition versus targets.", ProgressRequest, _progress),
    Tool("estimate_1rm", "Estimate one-rep max from a set (weight and reps).", OneRMRequest, _one_rm),
]}


def _simplify(node):
    """Trim pydantic's JSON schema so providers parse it reliably (drop titles, collapse Optional)."""
    if isinstance(node, dict):
        any_of = node.get("anyOf")
        if any_of:
            non_null = [x for x in any_of if x.get("type") != "null"]
            if len(non_null) == 1:
                node = {**{k: v for k, v in node.items() if k != "anyOf"}, **non_null[0]}
        return {k: _simplify(v) for k, v in node.items() if k not in ("title", "default")}
    if isinstance(node, list):
        return [_simplify(x) for x in node]
    return node


def _inline_refs(schema: dict) -> dict:
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def tool_specs() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": t.name, "description": t.description,
            "parameters": _simplify(_inline_refs(t.model.model_json_schema()))}}
        for t in TOOLS.values()
    ]


def execute_tool(user_id: str, name: str, raw_args) -> str:
    """Validate and run a tool; always returns a JSON string (errors are returned, not raised)."""
    with span(f"tool.{name}", tool=name) as sp:
        result = _execute_tool(user_id, name, raw_args)
        sp.set(ok=not result.startswith('{"error"'))
        if content_enabled():
            sp.set(arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args, default=str), result=result[:2000])
        return result


def _execute_tool(user_id: str, name: str, raw_args) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'", "available": list(TOOLS)})
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        validated = tool.model.model_validate(args)
        return json.dumps(tool.handler(user_id, validated))
    except ValidationError as e:
        errs = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
        return json.dumps({"error": "Invalid arguments", "details": errs})
    except (ValueError, json.JSONDecodeError) as e:
        return json.dumps({"error": str(e)})
