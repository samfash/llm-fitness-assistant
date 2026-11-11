"""Model price table. Prices are USD per million tokens and come ONLY from configuration:

    MODEL_PRICES='{"org/model": {"input_per_mtok": 0.2, "output_per_mtok": 0.6}}'      (JSON, or a path to a JSON file)

Unknown models have no cost (None), never an invented one: token counts are always tracked, dollars only when
you have told us the price.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _parse(raw: str) -> dict:
    if not raw:
        return {}
    try:
        text = open(raw, encoding="utf-8").read() if os.path.exists(raw) else raw
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        log.warning("MODEL_PRICES is not valid JSON or a readable file (%s); costs will be unknown", e)
        return {}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = _parse(os.getenv("MODEL_PRICES", "")).get(model)
    if not isinstance(price, dict):
        return None
    try:
        return (prompt_tokens * float(price["input_per_mtok"]) + completion_tokens * float(price["output_per_mtok"])) / 1_000_000
    except (KeyError, TypeError, ValueError):
        return None
