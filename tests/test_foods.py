import json
import re
from pathlib import Path

import pytest

from core import db
from core.food_import import food_row, portion_rows
from core.foods import Food, FoodDB, Portion, get_food_db
from core.schemas import FoodItem
from core.tools import execute_tool, resolve_portion, tool_specs
from evals.food_eval import check_gate, evaluate

ROOT = Path(__file__).resolve().parents[1]
QA = json.loads((ROOT / "evals" / "data" / "food_qa.json").read_text())
EGG_RAW_SR = 171287  # "Egg, whole, raw, fresh" (SR Legacy)


def run(uid, name, args):
    return json.loads(execute_tool(uid, name, args))


@pytest.fixture(scope="module")
def fdb():
    return get_food_db()


# ---- importer: converting USDA JSON records ---------------------------------------------------------
def _fn(nid, amount):
    return {"nutrient": {"id": nid}, "amount": amount}


def test_food_row_energy_falls_back_to_atwater_and_skips_when_missing():
    base = {"fdcId": 1, "description": " Test food ", "foodCategory": {"description": "Cat"}}
    r = food_row({**base, "foodNutrients": [_fn(2047, 250.4), _fn(1003, 10)]}, "sr_legacy")
    assert r["kcal"] == 250.4 and r["protein_g"] == 10 and r["description"] == "Test food" and r["fat_g"] == ""
    r2 = food_row({**base, "foodNutrients": [_fn(1008, 100), _fn(2047, 999)]}, "sr_legacy")
    assert r2["kcal"] == 100  # classic Energy entry wins
    assert food_row({**base, "foodNutrients": [_fn(1003, 10)]}, "sr_legacy") is None


def test_portion_labels_sr_and_fndds():
    sr = {"fdcId": 1, "foodPortions": [
        {"amount": 1.0, "modifier": "large", "gramWeight": 50.0, "sequenceNumber": 2},
        {"amount": 1.0, "modifier": "slice", "gramWeight": 29.0, "sequenceNumber": 1},
        {"amount": 1.0, "modifier": "slice", "gramWeight": 25.0, "sequenceNumber": 3},     # duplicate label: first wins
        {"amount": 3.0, "modifier": "oz", "gramWeight": 85.0, "sequenceNumber": 4},
        {"amount": 1.0, "modifier": "", "gramWeight": 10.0, "sequenceNumber": 5},          # no label: dropped
        {"amount": 1.0, "modifier": "cup", "gramWeight": 0, "sequenceNumber": 6}]}         # zero grams: dropped
    assert [(p["label"], p["grams"]) for p in portion_rows(sr, "sr_legacy")] == [("1 slice", 29.0), ("1 large", 50.0), ("3 oz", 85.0)]
    fn = {"fdcId": 2, "foodPortions": [
        {"portionDescription": "1 cup, cooked", "gramWeight": 158, "sequenceNumber": 1},
        {"portionDescription": "Quantity not specified", "gramWeight": 118, "sequenceNumber": 2},
        {"portionDescription": "Quantity not specified", "gramWeight": 0, "sequenceNumber": 3}]}
    assert [(p["label"], p["grams"]) for p in portion_rows(fn, "fndds")] == [("1 cup, cooked", 158.0), ("typical serving", 118.0)]


def test_tiny_gram_weights_keep_precision_or_are_dropped():
    food = {"fdcId": 3, "foodPortions": [
        {"portionDescription": "1 tablet", "gramWeight": 0.032, "sequenceNumber": 1},
        {"portionDescription": "1 speck", "gramWeight": 0.001, "sequenceNumber": 2}]}
    assert [(p["label"], p["grams"]) for p in portion_rows(food, "fndds")] == [("1 tablet", 0.03)]


# ---- the real dataset --------------------------------------------------------------------------------
def test_real_data_loaded_and_plausible(fdb):
    assert len(fdb) > 13000
    assert {f.source for f in fdb.foods} == {"fndds", "sr_legacy"}
    for f in fdb.foods:
        assert 0 <= f.kcal <= 905, f.description               # nothing exceeds pure fat (~900 kcal/100 g)
        assert f.protein_g + f.fat_g + f.carbs_g <= 106, f.description  # macros cannot exceed the 100 g sample
    assert all(p.grams > 0 for f in fdb.foods for p in f.portions)


def test_known_reference_values(fdb):
    egg = fdb.get(EGG_RAW_SR)
    assert egg.description == "Egg, whole, raw, fresh" and egg.kcal == 143 and egg.protein_g == pytest.approx(12.6, abs=0.1)
    assert {p.label: p.grams for p in egg.portions}["1 large"] == 50


def test_search_basics(fdb):
    assert fdb.search("") == [] and fdb.search("zxqvw nonsense") == [] and fdb.search("the and of") == []
    assert any(f.description.startswith("Egg, whole") for f in fdb.search("egg", 3))
    assert fdb.search("banana", 1)[0].description.lower().startswith("banana")
    assert len(fdb.search("rice", 8)) == 8


def test_every_eval_label_matches_something_real(fdb):
    for item in QA:
        assert any(re.search(p, f.description, re.I) for p in item["accept"] for f in fdb.foods), item["query"]


def test_food_search_quality_gate(fdb):
    res = evaluate(fdb, QA, "all")
    thresholds = json.loads((ROOT / "evals" / "food_thresholds.json").read_text())
    assert check_gate(res, thresholds) == [], res["failures"]


def test_dev_and_test_splits_are_disjoint_and_balanced():
    dev = {q["query"] for q in QA if q["split"] == "dev"}
    test = {q["query"] for q in QA if q["split"] == "test"}
    assert not dev & test and abs(len(dev) - len(test)) <= 1 and len(dev | test) == len(QA)


# ---- small synthetic DB: ranking behaviour ----------------------------------------------------------
def _food(i, desc, kcal=100):
    return Food(i, "fndds", desc, "Cat", kcal, 1, 1, 1)


def test_generic_entry_beats_qualified_variants():
    db_ = FoodDB([_food(1, "Egg, deviled"), _food(2, "Egg, whole, raw"), _food(3, "Egg, Benedict")])
    assert db_.search("egg", 1)[0].fdc_id == 2


def test_exact_head_match_beats_longer_names():
    db_ = FoodDB([_food(1, "Rice pudding"), _food(2, "Rice, white, cooked")])
    assert db_.search("rice", 1)[0].fdc_id == 2


# ---- search_food tool ----------------------------------------------------------------------------------
def test_tool_schemas_stay_flat_with_nested_items():
    blob = json.dumps(tool_specs())
    assert "$ref" not in blob and "$defs" not in blob and "anyOf" not in blob
    spec = next(s for s in tool_specs() if s["function"]["name"] == "log_foods")["function"]["parameters"]
    assert spec["properties"]["items"]["items"]["properties"]["fdc_id"]["type"] == "integer"


def test_search_food_tool_shape():
    out = run("u", "search_food", {"query": "boiled egg", "limit": 3})
    r = out["results"][0]
    assert len(out["results"]) == 3 and {"fdc_id", "description", "per_100g", "portions", "source"} <= set(r)
    assert set(r["per_100g"]) == {"kcal", "protein_g", "carbs_g", "fat_g"} and "USDA" in out["attribution"]
    assert run("u", "search_food", {"query": "zxqvw nonsense"})["results"] == []
    assert "error" in run("u", "search_food", {"query": "e"})


# ---- log_foods: deterministic arithmetic -----------------------------------------------------------
def test_log_foods_by_portion_computes_exact_macros_and_persists():
    out = run("u", "log_foods", {"items": [{"fdc_id": EGG_RAW_SR, "portion": "large", "quantity": 2}], "meal": "breakfast"})
    item = out["items"][0]
    assert out["saved"] and item["grams"] == 100 and item["calories"] == 143.0 and item["protein_g"] == 12.6
    assert out["totals"] == {"calories": 143.0, "protein_g": 12.6, "carbs_g": 0.7, "fat_g": 9.5}
    row = db.meals_since("u", "2000-01-01")[0]
    assert row["description"] == "breakfast" and row["calories"] == 143.0
    assert json.loads(row["items"])[0]["fdc_id"] == EGG_RAW_SR


def test_log_foods_by_grams_and_multiple_items_sum(fdb):
    banana = fdb.search("banana", 1)[0]
    out = run("u", "log_foods", {"items": [{"fdc_id": EGG_RAW_SR, "grams": 50}, {"fdc_id": banana.fdc_id, "grams": 120}]})
    expect_kcal = round(143 * 0.5, 1) + round(banana.kcal * 1.2, 1)
    assert out["totals"]["calories"] == pytest.approx(expect_kcal, abs=0.11)
    assert out["totals"]["protein_g"] == pytest.approx(sum(i["protein_g"] for i in out["items"]), abs=0.11)
    assert "Egg" in db.meals_since("u", "2000-01-01")[0]["description"]


def test_grams_take_precedence_over_portion():
    out = run("u", "log_foods", {"items": [{"fdc_id": EGG_RAW_SR, "grams": 10, "portion": "large", "quantity": 5}]})
    assert out["items"][0]["grams"] == 10


@pytest.mark.parametrize("args,needle", [
    ({"items": [{"fdc_id": 999999999, "grams": 50}]}, "Unknown fdc_id"),
    ({"items": [{"fdc_id": EGG_RAW_SR}]}, "grams"),
    ({"items": [{"fdc_id": EGG_RAW_SR, "grams": 5000}]}, "grams"),
    ({"items": [{"fdc_id": EGG_RAW_SR, "grams": 0}]}, "grams"),
    ({"items": [{"fdc_id": EGG_RAW_SR, "portion": "tablespoon"}]}, "not available"),
    ({"items": []}, "items"),
    ({"items": [{"fdc_id": EGG_RAW_SR, "grams": 50}], "date": "yesterday"}, "date"),
])
def test_log_foods_rejects_bad_input_and_saves_nothing(args, needle):
    out = run("u", "log_foods", args)
    assert "error" in out and needle in json.dumps(out)
    assert db.meals_since("u", "2000-01-01") == []


def test_implausible_meal_total_is_rejected(fdb):
    oil = fdb.search("olive oil", 1)[0]  # ~880 kcal/100 g
    out = run("u", "log_foods", {"items": [{"fdc_id": oil.fdc_id, "grams": 1000}, {"fdc_id": oil.fdc_id, "grams": 1000}]})
    assert "too large" in out["error"] and db.meals_since("u", "2000-01-01") == []


def test_ambiguous_portion_lists_choices(fdb):
    food = Food(1, "sr_legacy", "Test", "C", 100, 1, 1, 1, portions=[Portion("1 cup, sliced", 150), Portion("1 cup, mashed", 225)])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_portion(food, "cup")
    assert resolve_portion(food, "cup, mashed").grams == 225
    assert resolve_portion(food, "1 cup, sliced").grams == 150


def test_fooditem_requires_an_amount():
    with pytest.raises(ValueError):
        FoodItem(fdc_id=1)


def test_meals_are_per_user():
    run("alice", "log_foods", {"items": [{"fdc_id": EGG_RAW_SR, "grams": 100}]})
    assert db.meals_since("bob", "2000-01-01") == []


def test_progress_summary_counts_usda_meals():
    run("u", "log_foods", {"items": [{"fdc_id": EGG_RAW_SR, "grams": 100}]})
    s = run("u", "get_progress_summary", {"days": 7})
    assert s["avg_daily_calories"] == 143 and s["days_with_meals_logged"] == 1


# ---- golden eval cases for food logging work end to end with a correct scripted model ---------------------
def test_food_eval_cases_pass_with_a_correct_model_and_fail_with_a_lazy_one(fdb):
    from core.agent import AssistantTurn, ToolCall
    from evals.runner import load_cases, run_case

    class Scripted:
        def __init__(self, turns):
            self.turns = list(turns)

        def chat(self, messages, tools):
            return self.turns.pop(0)

    cases = {c["id"]: c for c in load_cases(ROOT / "evals" / "cases")}
    rice = fdb.search("rice cooked", 1)[0]
    cup = next((p for p in rice.portions if "cup" in p.label), rice.portions[0])
    good = Scripted([
        AssistantTurn(tool_calls=[ToolCall("1", "search_food", {"query": "rice cooked"})]),
        AssistantTurn(tool_calls=[ToolCall("2", "log_foods", {"items": [{"fdc_id": rice.fdc_id, "portion": cup.label}]})]),
        AssistantTurn(content=f"Logged {rice.description}: about {round(rice.kcal * cup.grams / 100)} kcal.")])
    r = run_case(cases["log_foods_rice"], good)
    assert r.passed, [c for c in r.checks if not c["ok"]]

    lazy = Scripted([AssistantTurn(tool_calls=[ToolCall("1", "log_meal", {"description": "rice", "calories": 200})]),
                     AssistantTurn(content="Logged rice, roughly 205 kcal.")])
    bad = run_case(cases["log_foods_rice"], lazy)
    failed = {c["name"] for c in bad.checks if not c["ok"]}
    assert not bad.passed and {"called:search_food", "called:log_foods", "not_called:log_meal"} <= failed
