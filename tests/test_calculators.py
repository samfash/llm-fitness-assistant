import pytest

from core.calculators import bmr_mifflin_st_jeor, compute_targets, epley_1rm
from core.schemas import Profile


def test_bmr_male():
    assert bmr_mifflin_st_jeor("male", 80, 180, 30) == 1780


def test_epley():
    assert epley_1rm(80, 8) == pytest.approx(101.333, rel=1e-3)
    assert epley_1rm(100, 1) == 100


def test_targets_maintenance():
    t = compute_targets(Profile(age=30, sex="male", height_cm=180, weight_kg=80, activity_level="moderate", goal="maintain"))
    assert t["tdee"] == 2759 and t["calories"] == 2759 and not t["floor_applied"]
    assert t["protein_g"] == 128  # 1.6 g/kg


def test_calorie_floor_enforced():
    t = compute_targets(Profile(age=40, sex="female", height_cm=150, weight_kg=40, activity_level="sedentary", goal="lose_fat"))
    assert t["calories"] == 1200 and t["floor_applied"]


def test_missing_fields_reported():
    with pytest.raises(ValueError, match="age"):
        compute_targets(Profile(sex="male"))
