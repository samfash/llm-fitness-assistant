"""Validated data models. These double as the JSON schemas the LLM sees as tool parameters."""
from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ActivityLevel = Literal["sedentary", "light", "moderate", "active", "very_active"]
Goal = Literal["lose_fat", "maintain", "build_muscle"]


def _check_date(v: str | None) -> str | None:
    if v is None:
        return v
    _date.fromisoformat(v)  # raises ValueError -> reported back to the model
    return v


class Profile(BaseModel):
    age: int | None = Field(None, ge=18, le=100, description="Age in years (adults only)")
    sex: Literal["male", "female"] | None = Field(None, description="Used for the BMR formula")
    height_cm: float | None = Field(None, ge=100, le=250)
    weight_kg: float | None = Field(None, ge=30, le=300)
    activity_level: ActivityLevel | None = None
    goal: Goal | None = None
    equipment: list[str] | None = Field(None, description="e.g. ['dumbbells', 'pull-up bar']")
    injuries: list[str] | None = Field(None, description="Known injuries or limitations")


class WorkoutEntry(BaseModel):
    exercise: str = Field(..., min_length=2, max_length=80, description="Exercise name, e.g. 'bench press'")
    sets: int = Field(1, ge=1, le=50)
    reps: int = Field(1, ge=1, le=500)
    weight_kg: float = Field(0, ge=0, le=1000, description="0 for bodyweight")
    duration_min: float | None = Field(None, gt=0, le=600, description="For cardio")
    date: str | None = Field(None, description="ISO date YYYY-MM-DD; defaults to today")
    notes: str | None = Field(None, max_length=300)

    _d = field_validator("date")(_check_date)


class MealEntry(BaseModel):
    description: str = Field(..., min_length=2, max_length=200)
    calories: float = Field(..., ge=0, le=5000, description="Estimated kcal")
    protein_g: float = Field(0, ge=0, le=500)
    carbs_g: float = Field(0, ge=0, le=1000)
    fat_g: float = Field(0, ge=0, le=500)
    date: str | None = Field(None, description="ISO date YYYY-MM-DD; defaults to today")

    _d = field_validator("date")(_check_date)


class ProgressRequest(BaseModel):
    days: int = Field(7, ge=1, le=90, description="Look-back window in days")


class OneRMRequest(BaseModel):
    weight_kg: float = Field(..., gt=0, le=1000)
    reps: int = Field(..., ge=1, le=20)


class NoArgs(BaseModel):
    pass
