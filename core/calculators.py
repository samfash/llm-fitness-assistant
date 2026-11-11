"""Deterministic fitness maths. The LLM calls these through tools; it never does the arithmetic itself."""
from __future__ import annotations

from .schemas import Profile

ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
GOAL_ADJUST = {"lose_fat": 0.80, "maintain": 1.00, "build_muscle": 1.10}
PROTEIN_G_PER_KG = {"lose_fat": 2.0, "maintain": 1.6, "build_muscle": 1.8}
CALORIE_FLOOR = {"female": 1200, "male": 1500}  # never recommend below these


def bmr_mifflin_st_jeor(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + (5 if sex == "male" else -161)


def epley_1rm(weight_kg: float, reps: int) -> float:
    return weight_kg if reps == 1 else weight_kg * (1 + reps / 30)


def compute_targets(p: Profile) -> dict:
    required = ["age", "sex", "height_cm", "weight_kg", "activity_level", "goal"]
    missing = [f for f in required if getattr(p, f) is None]
    if missing:
        raise ValueError(f"Missing profile fields: {', '.join(missing)}")

    bmr = bmr_mifflin_st_jeor(p.sex, p.weight_kg, p.height_cm, p.age)
    tdee = bmr * ACTIVITY[p.activity_level]
    target = tdee * GOAL_ADJUST[p.goal]
    floor = CALORIE_FLOOR[p.sex]
    floor_applied = target < floor
    target = max(target, floor)

    protein = PROTEIN_G_PER_KG[p.goal] * p.weight_kg
    fat = 0.25 * target / 9
    carbs = max(0.0, (target - protein * 4 - fat * 9) / 4)
    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "calories": round(target),
        "protein_g": round(protein),
        "fat_g": round(fat),
        "carbs_g": round(carbs),
        "floor_applied": floor_applied,
        "note": "Estimates only. Not medical advice.",
    }
