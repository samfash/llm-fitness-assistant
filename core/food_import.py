"""One-off importer: USDA FoodData Central JSON downloads -> two compact CSV files.

    pip install ijson
    python -m core.food_import --sr FoodData_Central_sr_legacy_food_json_2018-04.json \\
                               --survey surveyDownload.json --out fooddata

Streams the (large) JSON files, keeps only what the app needs (per-100 g macros + portions), and writes
`foods.csv` and `portions.csv`. The CSVs are small, text, and diff-able, so they can live in git; the app
loads them into memory at start-up.

Source: U.S. Department of Agriculture, Agricultural Research Service. FoodData Central. https://fdc.nal.usda.gov/
(public domain, CC0 1.0).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FOOD_COLUMNS = ["fdc_id", "source", "description", "category", "kcal", "protein_g", "fat_g", "carbs_g",
                "fiber_g", "sugar_g", "sodium_mg"]
PORTION_COLUMNS = ["fdc_id", "label", "grams"]

# nutrient ids; energy falls back to the Atwater entries that Foundation Foods use instead of id 1008
KCAL_IDS = (1008, 2047, 2048)
NUTRIENTS = {"protein_g": (1003,), "fat_g": (1004,), "carbs_g": (1005,), "fiber_g": (1079,),
             "sugar_g": (2000,), "sodium_mg": (1093,)}


def _amounts(food: dict) -> dict[int, float]:
    out = {}
    for fn in food.get("foodNutrients", []):
        n, amt = fn.get("nutrient") or {}, fn.get("amount")
        if n.get("id") is not None and amt is not None:
            out[n["id"]] = float(amt)
    return out


def food_row(food: dict, source: str) -> dict | None:
    a = _amounts(food)
    kcal = next((a[i] for i in KCAL_IDS if i in a), None)
    if kcal is None:
        return None  # no energy value (e.g. FNDDS human milk): useless for logging
    cat = (food.get("foodCategory") or {}).get("description") or \
          (food.get("wweiaFoodCategory") or {}).get("wweiaFoodCategoryDescription") or ""
    row = {"fdc_id": food["fdcId"], "source": source, "description": food["description"].strip(),
           "category": cat, "kcal": round(kcal, 1)}
    for col, ids in NUTRIENTS.items():
        v = next((a[i] for i in ids if i in a), None)
        row[col] = "" if v is None else round(v, 2)
    return row


def portion_rows(food: dict, source: str) -> list[dict]:
    seen, rows = set(), []
    for p in sorted(food.get("foodPortions", []), key=lambda p: p.get("sequenceNumber") or 0):
        gw = p.get("gramWeight") or 0
        if gw <= 0:
            continue
        if source == "sr_legacy":
            amount = p.get("amount") or 1.0
            mod = (p.get("modifier") or "").strip()
            if not mod:
                continue
            label = f"1 {mod}" if amount == 1 else f"{amount:g} {mod}"
        else:  # FNDDS: portionDescription already reads "1 cup, cooked"
            label = (p.get("portionDescription") or "").strip()
            if label.lower() == "quantity not specified":
                label = "typical serving"
        key = label.lower()
        if not label or key in seen:  # SR sometimes lists the same label twice with different weights; keep first
            continue
        grams = round(float(gw), 2 if gw < 1 else 1)  # keep precision for sub-gram portions (e.g. a tablet)
        if grams <= 0:
            continue
        seen.add(key)
        rows.append({"fdc_id": food["fdcId"], "label": label, "grams": grams})
    return rows


def convert(path: str, array_key: str, source: str, foods_w, portions_w) -> tuple[int, int]:
    import ijson  # dev-time dependency only

    n_food = n_skip = 0
    with open(path, "rb") as f:
        for food in ijson.items(f, f"{array_key}.item", use_float=True):
            row = food_row(food, source)
            if row is None:
                n_skip += 1
                continue
            foods_w.writerow(row)
            portions_w.writerows(portion_rows(food, source))
            n_food += 1
    return n_food, n_skip


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sr", help="SR Legacy JSON file")
    p.add_argument("--survey", help="Survey (FNDDS) JSON file")
    p.add_argument("--out", default="fooddata")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "foods.csv", "w", newline="", encoding="utf-8") as ff, \
            open(out / "portions.csv", "w", newline="", encoding="utf-8") as pf:
        fw, pw = csv.DictWriter(ff, FOOD_COLUMNS), csv.DictWriter(pf, PORTION_COLUMNS)
        fw.writeheader()
        pw.writeheader()
        for path, key, source in ((a.survey, "SurveyFoods", "fndds"), (a.sr, "SRLegacyFoods", "sr_legacy")):
            if path:
                n, skipped = convert(path, key, source, fw, pw)
                print(f"{source}: {n} foods written, {skipped} skipped (no energy value)")


if __name__ == "__main__":
    main()
