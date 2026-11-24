"""Contract validation tests for F1 Pit-Wall WebSocket packets.

Enforces:
1. ``contract/live_packet.json`` is a valid JSON Schema (draft 2020-12).
2. All embedded examples in ``contract/live_packet.json`` pass schema validation.
3. ``drivers[]`` and ``predictions.items[]`` include ``podium_prob_pct`` (0.0..100.0)
   alongside ``win_prob_pct``, sorted descending in the ML leaderboard, and active
   drivers' ``podium_prob_pct`` sums to 300.0% (while ``win_prob_pct`` sums to 100.0%).
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict

import jsonschema
from jsonschema import Draft202012Validator
import pytest

import config


@pytest.fixture(scope="module")
def schema() -> Dict[str, Any]:
    with config.CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def validator(schema: Dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_schema_is_valid_draft_2020_12(schema: Dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert len(schema.get("examples", [])) == 3


def test_all_contract_examples_validate(schema: Dict[str, Any], validator: Draft202012Validator) -> None:
    examples = schema["examples"]
    types_seen = set()
    for example in examples:
        validator.validate(example)
        types_seen.add(example["type"])
    assert types_seen == {"post_quali_prediction", "live_race_update", "replay_frame"}


def test_examples_enforce_top3_podium_300_sum_and_sorting(schema: Dict[str, Any]) -> None:
    for example in schema["examples"]:
        if example["type"] == "post_quali_prediction":
            preds = example["predictions"]
            items = preds["items"]
            podium_vals = [item["podium_prob_pct"] for item in items]
            win_vals = [item["win_prob_pct"] for item in items]
            assert podium_vals == sorted(podium_vals, reverse=True)
            assert all(0.0 <= p <= 100.0 for p in podium_vals)
            assert pytest.approx(sum(podium_vals), abs=0.05) == 300.0
            assert pytest.approx(sum(win_vals), abs=0.05) == 100.0
            assert pytest.approx(preds["sum_podium_pct"], abs=0.05) == 300.0
            assert pytest.approx(preds["sum_win_pct"], abs=0.05) == 100.0
        elif example["type"] == "live_race_update":
            race = example["race"]
            drivers = race["drivers"]
            active_podium = [d["podium_prob_pct"] for d in drivers if not d["retired"]]
            active_win = [d["win_prob_pct"] for d in drivers if not d["retired"]]
            assert all(0.0 <= p <= 100.0 for p in active_podium)
            assert pytest.approx(sum(active_podium), abs=0.05) == 300.0
            assert pytest.approx(sum(active_win), abs=0.05) == 100.0

            preds = race["predictions"]
            assert preds is not None
            p_items = [i["podium_prob_pct"] for i in preds["items"]]
            assert p_items == sorted(p_items, reverse=True)
            assert pytest.approx(sum(p_items), abs=0.05) == 300.0


def test_schema_rejects_podium_prob_above_100(schema: Dict[str, Any], validator: Draft202012Validator) -> None:
    bad = copy.deepcopy(schema["examples"][0])
    bad["predictions"]["items"][0]["podium_prob_pct"] = 105.0
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)


def test_schema_rejects_missing_podium_prob_pct(schema: Dict[str, Any], validator: Draft202012Validator) -> None:
    bad = copy.deepcopy(schema["examples"][1])
    del bad["race"]["drivers"][0]["podium_prob_pct"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)
