"""Tests for the Top-3 (Podium) ML pipeline and RacePredictor.

Verifies:
1. Both ``pre_race_model.joblib`` and ``live_race_model.joblib`` train on
   ``is_podium = (finishing_position <= 3)``.
2. ``normalize_podium_to_percent`` clips individual driver probabilities to
   ``[0.0, 100.0]``, zeroes out ``DNF`` / ``OUT`` drivers, and normalizes active
   drivers to sum to ``300.0%``.
3. ``RacePredictor.predict_post_quali`` and ``RacePredictor.predict_live`` emit
   contract-valid packets sorted descending by ``podium_prob_pct`` with active
   grid summing to ``300.0%`` and ``DNF`` / ``OUT`` drivers at ``0.0%``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator
import numpy as np
import pandas as pd
import pytest

import config
from ml.features import (
    build_in_race_features,
    build_pre_race_features,
    normalize_podium_to_percent,
    normalize_to_percent,
)
from ml.predictor import RaceContext, RacePredictor
from ml.train import save_artifact, train_in_race, train_pre_race


def _synthetic_pre_race_history(seasons: Sequence[int] = (2023, 2024, 2025, 2026)) -> pd.DataFrame:
    drivers = [
        ("max_verstappen", "VER", 1, "Max Verstappen", "red_bull", "Red Bull Racing"),
        ("norris", "NOR", 4, "Lando Norris", "mclaren", "McLaren"),
        ("leclerc", "LEC", 16, "Charles Leclerc", "ferrari", "Ferrari"),
        ("antonelli", "ANT", 12, "Kimi Antonelli", "mercedes", "Mercedes"),
        ("piastri", "PIA", 81, "Oscar Piastri", "mclaren", "McLaren"),
        ("russell", "RUS", 63, "George Russell", "mercedes", "Mercedes"),
        ("hamilton", "HAM", 44, "Lewis Hamilton", "ferrari", "Ferrari"),
        ("sainz", "SAI", 55, "Carlos Sainz", "williams", "Williams"),
        ("alonso", "ALO", 14, "Fernando Alonso", "aston_martin", "Aston Martin"),
        ("gasly", "GAS", 10, "Pierre Gasly", "alpine", "Alpine"),
    ]
    rng = np.random.default_rng(42)
    rows: List[Dict[str, Any]] = []
    for season in seasons:
        for rnd in range(1, 7):
            perm = rng.permutation(len(drivers))
            for grid_idx, d_idx in enumerate(perm, start=1):
                d_id, code, num, name, c_id, c_name = drivers[int(d_idx)]
                # Favour front qualifiers for podium finishes
                noise = int(rng.integers(-1, 3))
                finish = max(1, min(len(drivers), grid_idx + noise))
                rows.append(
                    {
                        "season": season,
                        "round": rnd,
                        "race_name": f"Test GP R{rnd}",
                        "circuit_id": f"circuit_{rnd % 3}",
                        "circuit_name": f"Circuit {rnd % 3}",
                        "race_date": f"{season}-0{rnd}-15",
                        "driver_id": d_id,
                        "driver_code": code,
                        "driver_number": num,
                        "driver_name": name,
                        "constructor_id": c_id,
                        "constructor_name": c_name,
                        "grid": grid_idx,
                        "quali_position": grid_idx,
                        "q1_s": 80.0 + grid_idx * 0.15,
                        "q2_s": 79.5 + grid_idx * 0.15,
                        "q3_s": 79.0 + grid_idx * 0.15,
                        "quali_best_s": 79.0 + grid_idx * 0.15,
                        "finish_position": finish,
                        "position_text": str(finish),
                        "status": "Finished",
                        "points": max(0.0, float(26 - finish * 2)),
                        "laps_completed": 57,
                        "won": 1 if finish == 1 else 0,
                        "podium": 1 if finish <= 3 else 0,
                        "is_podium": 1 if finish <= 3 else 0,
                        "dnf": 0,
                        "completed": True,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_in_race_snapshots(pre_race_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (season, rnd), group in pre_race_df.groupby(["season", "round"]):
        for lap in (5, 15, 30, 45):
            for _, row in group.iterrows():
                pos = int(row["finish_position"])
                gap = float((pos - 1) * 2.2)
                rows.append(
                    {
                        "season": int(season),
                        "round": int(rnd),
                        "session_key": int(season * 100 + rnd),
                        "lap": lap,
                        "total_laps": 57,
                        "driver_number": int(row["driver_number"]),
                        "driver_code": row["driver_code"],
                        "position": pos,
                        "laps_completed": lap,
                        "laps_down": 0,
                        "gap_to_leader_s": gap,
                        "interval_s": 0.0 if pos == 1 else 2.2,
                        "last_lap_s": 90.0 + pos * 0.12,
                        "pace_3_s": 90.0 + pos * 0.12,
                        "ref_lap_s": 90.5,
                        "compound": "MEDIUM" if lap < 25 else "HARD",
                        "tyre_age_laps": lap if lap < 25 else lap - 20,
                        "stint_number": 1 if lap < 25 else 2,
                        "pit_stops_done": 0 if lap < 25 else 1,
                        "track_state": 0,
                        "final_position": pos,
                        "won": int(row["won"]),
                        "podium": int(row["podium"]),
                        "is_podium": int(row["is_podium"]),
                    }
                )
    return pd.DataFrame(rows)


def test_normalize_podium_to_percent_clipping_and_dnf() -> None:
    # Extreme skew where raw top score would exceed 100% without iterative clipping
    scores = [100.0, 50.0, 1.0, 1.0, 1.0, 1.0, 999.0]
    active_mask = [True, True, True, True, True, True, False]  # last driver is DNF/OUT
    result = normalize_podium_to_percent(scores, active_mask=active_mask, decimals=2)

    assert result[-1] == 0.0
    assert all(0.0 <= p <= 100.0 for p in result)
    assert pytest.approx(sum(result), abs=0.01) == 300.0
    assert result[0] == 100.0
    assert result[1] == 100.0


def test_train_and_predict_top3_podium_end_to_end(tmp_path: Path) -> None:
    raw_pre = _synthetic_pre_race_history()
    pre_features = build_pre_race_features(raw_pre)
    assert "is_podium" in pre_features.columns
    assert set(pre_features["is_podium"].dropna().unique()) <= {0.0, 1.0}

    pre_artifact = train_pre_race(pre_features, n_splits=2, min_train_seasons=2)
    assert pre_artifact["target"] == "is_podium"
    pre_model_path = tmp_path / "pre_race_model.joblib"
    save_artifact(pre_artifact, pre_model_path)

    raw_in = _synthetic_in_race_snapshots(raw_pre)
    in_features = build_in_race_features(raw_in, pre_features)
    assert "is_podium" in in_features.columns

    in_artifact = train_in_race(in_features, n_splits=2, min_train_seasons=2)
    assert in_artifact["target"] == "is_podium"
    live_model_path = tmp_path / "live_race_model.joblib"
    save_artifact(in_artifact, live_model_path)
    assert live_model_path.exists()

    history_path = tmp_path / "pre_race_raw.parquet"
    raw_pre.to_parquet(history_path, index=False)

    predictor = RacePredictor(
        pre_race_path=pre_model_path,
        live_race_path=live_model_path,
        history_path=history_path,
    )
    assert predictor.pre_race_available
    assert predictor.in_race_available

    target_rows = pre_features[(pre_features["season"] == 2026) & (pre_features["round"] == 6)].reset_index(drop=True)
    ctx = RaceContext(
        season=2026,
        round_number=6,
        race_name="Las Vegas Grand Prix",
        circuit_id="vegas",
        circuit_name="Las Vegas Strip Circuit",
        race_date="2026-11-21",
        grid_source="jolpica_qualifying",
        features=target_rows,
        meeting_key=1251,
        session_key=9644,
        rotation_deg=90.0,
    )

    schema = json.loads(config.CONTRACT_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    # 1. Post-qualifying prediction packet
    post_quali_packet = predictor.predict_post_quali(context=ctx)
    validator.validate(post_quali_packet)
    pq_items = post_quali_packet["predictions"]["items"]
    pq_podium = [item["podium_prob_pct"] for item in pq_items]
    pq_win = [item["win_prob_pct"] for item in pq_items]
    assert pq_podium == sorted(pq_podium, reverse=True)
    assert all(0.0 <= p <= 100.0 for p in pq_podium)
    assert pytest.approx(sum(pq_podium), abs=0.05) == 300.0
    assert pytest.approx(sum(pq_win), abs=0.05) == 100.0

    # 2. Live in-race prediction with DNF / OUT drivers
    live_drivers: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(target_rows.iterrows(), start=1):
        is_dnf = idx >= 9  # 2 drivers retired / DNF
        live_drivers.append(
            {
                "driver_number": int(row["driver_number"]),
                "code": str(row["driver_code"]),
                "full_name": str(row["driver_name"]),
                "team_name": str(row["constructor_name"]),
                "position": None if is_dnf else idx,
                "gap_to_leader_s": None if is_dnf else float((idx - 1) * 1.8),
                "interval_s": None if is_dnf else (0.0 if idx == 1 else 1.8),
                "laps_down": 0,
                "last_lap_time_s": 91.0 + idx * 0.1,
                "tyre": {"compound": "MEDIUM", "age_laps": 14, "stint_number": 1},
                "pit_stops": [],
                "in_pit": False,
                "retired": is_dnf,
                "status": "DNF" if idx == 9 else ("OUT" if idx == 10 else "RUNNING"),
                "podium_prob_pct": 0.0,
                "win_prob_pct": 0.0,
            }
        )

    live_pred = predictor.predict_live(ctx, live_drivers, lap=18, total_laps=50, track_state="GREEN")
    items = live_pred["items"]
    podium_vals = [i["podium_prob_pct"] for i in items]
    win_vals = [i["win_prob_pct"] for i in items]

    # Sorted descending by podium_prob_pct
    assert podium_vals == sorted(podium_vals, reverse=True)
    assert all(0.0 <= p <= 100.0 for p in podium_vals)
    assert pytest.approx(sum(podium_vals), abs=0.05) == 300.0
    assert pytest.approx(sum(win_vals), abs=0.05) == 100.0

    # Retired / DNF / OUT drivers must have exactly 0.0%
    dnf_numbers = {int(target_rows.iloc[8]["driver_number"]), int(target_rows.iloc[9]["driver_number"])}
    for item in items:
        if item["driver_number"] in dnf_numbers:
            assert item["podium_prob_pct"] == 0.0
            assert item["win_prob_pct"] == 0.0
