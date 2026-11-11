"""In-memory food database (USDA FoodData Central: Survey/FNDDS + SR Legacy) with ranked search.

Data: U.S. Department of Agriculture, Agricultural Research Service. FoodData Central. https://fdc.nal.usda.gov/
(public domain, CC0 1.0). Nutrient values are per 100 g.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .rag.bm25 import BM25
from .rag.text import stem, tokenize

log = logging.getLogger(__name__)
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "fooddata"
# neutral "form" words that do not change what the food is; any OTHER description word the query did not
# ask for ("deviled", "Benedict", "creamed") makes an entry less likely to be the generic one wanted
PLAIN_WORDS = frozenset(stem(w) for w in ("whole", "raw", "plain", "fresh", "nfs", "ns", "cooked", "regular"))
ATTRIBUTION = "USDA FoodData Central (public domain, CC0 1.0)"
SOURCE_NAMES = {"fndds": "FNDDS 2021-2023", "sr_legacy": "SR Legacy"}


@dataclass(frozen=True)
class Portion:
    label: str
    grams: float


@dataclass
class Food:
    fdc_id: int
    source: str
    description: str
    category: str
    kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None
    portions: list[Portion] = field(default_factory=list)

    def per_100g(self) -> dict:
        return {"kcal": self.kcal, "protein_g": self.protein_g, "carbs_g": self.carbs_g, "fat_g": self.fat_g}

    def macros_for(self, grams: float) -> dict:
        f = grams / 100.0
        return {"calories": round(self.kcal * f, 1), "protein_g": round(self.protein_g * f, 1),
                "carbs_g": round(self.carbs_g * f, 1), "fat_g": round(self.fat_g * f, 1)}


def _num(v: str) -> float | None:
    return float(v) if v not in ("", None) else None


def load_foods(directory: Path | str = DEFAULT_DIR) -> list[Food]:
    d = Path(directory)
    foods: dict[int, Food] = {}
    with open(d / "foods.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = int(r["fdc_id"])
            foods[fid] = Food(fid, r["source"], r["description"], r["category"], float(r["kcal"]),
                              _num(r["protein_g"]) or 0.0, _num(r["fat_g"]) or 0.0, _num(r["carbs_g"]) or 0.0,
                              _num(r["fiber_g"]), _num(r["sugar_g"]), _num(r["sodium_mg"]))
    with open(d / "portions.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            food = foods.get(int(r["fdc_id"]))
            if food:
                food.portions.append(Portion(r["label"], float(r["grams"])))
    return list(foods.values())


class FoodDB:
    def __init__(self, foods: list[Food]):
        self.foods = foods
        self.by_id = {f.fdc_id: f for f in foods}
        self._tokens = [tokenize(f"{f.description} {f.category}") for f in foods]
        self._desc_tokens = [tokenize(f.description) for f in foods]
        self._head_tokens = [set(tokenize(f.description.split(",")[0])) for f in foods]
        self._bm25 = BM25(self._tokens) if foods else None

    def __len__(self) -> int:
        return len(self.foods)

    def get(self, fdc_id: int) -> Food | None:
        return self.by_id.get(fdc_id)

    def search(self, query: str, limit: int = 5) -> list[Food]:
        q = tokenize(query)
        if not q or not self._bm25:
            return []
        qset = set(q)
        base = self._bm25.scores(q)
        scored = []
        for i, b in enumerate(base):
            if b <= 0:
                continue
            desc = set(self._desc_tokens[i])
            score = b
            if self._head_tokens[i] and self._head_tokens[i] <= qset:
                score += 3.0   # the food's main name is exactly what was asked for ("Egg, ..." for "egg")
            if qset <= desc:
                score += 1.5   # every query word appears in the description
            unexplained = [t for t in self._desc_tokens[i] if t not in qset and t not in PLAIN_WORDS]
            score -= 0.35 * len(unexplained)  # prefer the generic entry over qualified variants
            scored.append((score, i))
        scored.sort(reverse=True)
        return [self.foods[i] for _, i in scored[:limit]]


@lru_cache(maxsize=1)
def get_food_db() -> FoodDB:
    directory = os.getenv("FOODDATA_DIR", str(DEFAULT_DIR))
    try:
        db = FoodDB(load_foods(directory))
        log.info("Loaded %d foods from %s", len(db), directory)
        return db
    except FileNotFoundError:
        log.warning("Food data not found in %s; food tools will report it is unavailable", directory)
        return FoodDB([])
