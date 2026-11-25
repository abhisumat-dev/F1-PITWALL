"""Comprehensive White-Box Unit & Integration Test Suite for F1 Pit-Wall.

Covers internal branches, mathematical invariants, security boundaries, and
state machines across:
- ML feature engineering (zero future data leakage, pit-lane starts, water-filling clipping)
- Trained production `.joblib` models (`pre_race_model.joblib` & `live_race_model.joblib`)
- 26-circuit 2D geometry catalog & tricky 2026 calendar alias disambiguation
- OpenF1 cancelled-round filtering & 1-to-1 Jolpica round alignment
- Final race classification (`race_results.py`: completed vs upcoming, DNF, lapped, stints)
- SignalR live `session_active` state detection & `F1_LIVE_TOKEN` telemetry gate
- Team Radio Whisper SSRF host allowlist security enforcement
- Sector 1/2/3 PURPLE / GREEN / YELLOW state machine
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from jsonschema import Draft202012Validator
import numpy as np
import pandas as pd
import pytest

import config
from analytics.radio_transcriber import RadioTranscriber
from analytics.sectors_fastest import SectorAndLapTracker
from ml.features import (
    PRE_RACE_FEATURES,
    build_pre_race_features,
    make_pre_race_x,
    normalize_podium_to_percent,
)
from ml.predictor import RaceContext, RacePredictor
from server.circuits import load_circuits_catalog, resolve_circuit_2d
from server.race_results import get_final_race_results
from streams.replay_engine import ReplayEngine
from streams.signalr_live import SignalRLiveProvider


def _validator() -> Draft202012Validator:
    schema = json.loads(config.CONTRACT_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# --------------------------------------------------------------------------- #
# 1. White-Box: Zero Data Leakage & Feature Boundary Tests
# --------------------------------------------------------------------------- #
def test_whitebox_pre_race_features_strict_zero_future_leakage() -> None:
    """Mutating a target race's finish_position/points/won/podium MUST NOT alter its pre-race features."""
    rows: List[Dict[str, Any]] = []
    for rnd in range(1, 7):
        for idx, (did, code, cid) in enumerate(
            [("ver", "VER", "red_bull"), ("nor", "NOR", "mclaren"), ("lec", "LEC", "ferrari"), ("ant", "ANT", "mercedes")],
            start=1,
        ):
            rows.append(
                {
                    "season": 2026,
                    "round": rnd,
                    "race_name": f"GP {rnd}",
                    "circuit_id": "miami",
                    "circuit_name": "Miami",
                    "race_date": f"2026-05-0{rnd}",
                    "driver_id": did,
                    "driver_code": code,
                    "driver_number": idx * 10,
                    "driver_name": code,
                    "constructor_id": cid,
                    "constructor_name": cid.upper(),
                    "grid": 0 if (rnd == 6 and idx == 4) else idx,  # pit-lane start on R6 for driver 4
                    "quali_position": idx,
                    "q1_s": 88.0 + idx * 0.2,
                    "q2_s": 87.5 + idx * 0.2,
                    "q3_s": 87.0 + idx * 0.2,
                    "quali_best_s": 87.0 + idx * 0.2,
                    "finish_position": idx,
                    "position_text": str(idx),
                    "status": "Finished",
                    "points": float(25 - idx * 5),
                    "laps_completed": 57,
                    "won": 1 if idx == 1 else 0,
                    "podium": 1 if idx <= 3 else 0,
                    "is_podium": 1 if idx <= 3 else 0,
                    "dnf": 0,
                    "completed": True,
                }
            )

    df_a = pd.DataFrame(rows)
    df_b = df_a.copy()
    # Flip R6 race results completely (P1 DNF with 0 pts, P4 wins with 25 pts)
    r6_mask = df_b["round"] == 6
    df_b.loc[r6_mask, "finish_position"] = [20, 19, 2, 1]
    df_b.loc[r6_mask, "points"] = [0.0, 0.0, 18.0, 25.0]
    df_b.loc[r6_mask, "won"] = [0, 0, 0, 1]
    df_b.loc[r6_mask, "podium"] = [0, 0, 1, 1]
    df_b.loc[r6_mask, "is_podium"] = [0, 0, 1, 1]
    df_b.loc[r6_mask, "dnf"] = [1, 1, 0, 0]

    feat_a = build_pre_race_features(df_a)
    feat_b = build_pre_race_features(df_b)

    x_a = make_pre_race_x(feat_a[feat_a["round"] == 6])
    x_b = make_pre_race_x(feat_b[feat_b["round"] == 6])

    assert list(x_a.columns) == PRE_RACE_FEATURES
    assert np.allclose(x_a.fillna(-999.0).to_numpy(), x_b.fillna(-999.0).to_numpy(), atol=1e-12)

    # Verify pit-lane start handling (grid == 0 -> pit_lane_start == 1.0, grid_position == field_size + 1)
    pit_row = feat_a[(feat_a["round"] == 6) & (feat_a["driver_code"] == "ANT")].iloc[0]
    assert pit_row["pit_lane_start"] == 1.0
    assert pit_row["grid_position"] == 5.0


def test_whitebox_podium_normalization_edge_cases() -> None:
    # Fewer than 3 active cars -> sums to 100 * n_active
    two_active = normalize_podium_to_percent([10.0, 5.0, 99.0], active_mask=[True, True, False])
    assert two_active == [100.0, 100.0, 0.0]

    # Zero active cars -> all 0.0
    none_active = normalize_podium_to_percent([10.0, 5.0], active_mask=[False, False])
    assert none_active == [0.0, 0.0]

    # All-zero active scores -> uniform distribution summing to 300.0%
    zeros = normalize_podium_to_percent([0.0, 0.0, 0.0, 0.0], active_mask=[True, True, True, True])
    assert pytest.approx(sum(zeros), abs=0.01) == 300.0
    assert all(p == 75.0 for p in zeros)


# --------------------------------------------------------------------------- #
# 2. White-Box: Production Trained Artifacts Verification
# --------------------------------------------------------------------------- #
def test_whitebox_trained_joblib_artifacts_predict_contract_valid() -> None:
    if not config.PRE_RACE_MODEL_PATH.exists() or not config.LIVE_RACE_MODEL_PATH.exists():
        pytest.skip("Production joblib artifacts not present")

    predictor = RacePredictor()
    assert predictor.pre_race_available is True
    assert predictor.in_race_available is True

    # Live inference with mixed active + retired drivers
    prior_rows = pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 21,
                "driver_number": dn,
                "driver_code": code,
                "grid_position": idx,
                "driver_win_rate_10": 0.2 if idx <= 2 else 0.02,
                "driver_podium_rate_10": 0.5 if idx <= 3 else 0.1,
                "driver_avg_finish_5": float(idx + 1),
                "team_points_avg_5": float(30 - idx * 3),
                "team_win_rate_10": 0.25 if idx <= 2 else 0.0,
                "team_points_momentum": 3.5 if idx == 2 else 0.0,
                "team_quali_improvement_slope": 0.2 if idx == 2 else 0.0,
            }
            for idx, (dn, code) in enumerate([(1, "NOR"), (3, "VER"), (63, "RUS"), (16, "LEC"), (44, "HAM")], start=1)
        ]
    )
    ctx = RaceContext(
        season=2026,
        round_number=21,
        race_name="Las Vegas Grand Prix",
        circuit_id="vegas",
        circuit_name="Las Vegas",
        race_date="2026-11-22",
        grid_source="jolpica_qualifying",
        features=prior_rows,
    )
    live_drivers = [
        {"driver_number": 3, "code": "VER", "position": 1, "gap_to_leader_s": 0.0, "interval_s": 0.0, "tyre": {"compound": "HARD", "age_laps": 15}, "retired": False},
        {"driver_number": 1, "code": "NOR", "position": 2, "gap_to_leader_s": 1.2, "interval_s": 1.2, "tyre": {"compound": "HARD", "age_laps": 14}, "retired": False},
        {"driver_number": 63, "code": "RUS", "position": 3, "gap_to_leader_s": 4.5, "interval_s": 3.3, "tyre": {"compound": "MEDIUM", "age_laps": 18}, "retired": False},
        {"driver_number": 16, "code": "LEC", "position": 4, "gap_to_leader_s": 9.1, "interval_s": 4.6, "tyre": {"compound": "HARD", "age_laps": 15}, "retired": False},
        {"driver_number": 44, "code": "HAM", "position": 5, "gap_to_leader_s": None, "interval_s": None, "tyre": {"compound": "SOFT", "age_laps": 5}, "retired": True},
    ]
    pred = predictor.predict_live(ctx, live_drivers, lap=35, total_laps=50, track_state="GREEN")
    assert pytest.approx(pred["sum_podium_pct"], abs=0.05) == 300.0
    assert pytest.approx(pred["sum_win_pct"], abs=0.05) == 100.0
    ham_item = next(i for i in pred["items"] if i["driver_code"] == "HAM")
    assert ham_item["podium_prob_pct"] == 0.0
    assert ham_item["win_prob_pct"] == 0.0


# --------------------------------------------------------------------------- #
# 3. White-Box: 26 Circuits & Alias Disambiguation
# --------------------------------------------------------------------------- #
def test_whitebox_all_26_circuits_geometry_and_2026_aliases() -> None:
    catalog = load_circuits_catalog()
    assert len(catalog) >= 26
    for slug, c in catalog.items():
        pts = c.get("points") or []
        assert len(pts) >= 30, f"{slug} has too few points"
        assert all(math.isfinite(p["x"]) and math.isfinite(p["y"]) for p in pts)
        splits = c.get("sector_splits") or []
        assert len(splits) == 4
        assert splits == sorted(splits) and splits[-1] < len(pts)

    # Tricky 2026 disambiguations
    assert resolve_circuit_2d(circuit_id="miami", grand_prix="Miami Grand Prix")["slug"] == "miami"
    assert resolve_circuit_2d(circuit_id="monaco", grand_prix="Monaco Grand Prix")["slug"] == "monte_carlo"
    assert resolve_circuit_2d(circuit_id="madring", grand_prix="Spanish Grand Prix")["slug"] == "madring"
    assert resolve_circuit_2d(circuit_id="sepang", grand_prix="Bahrain Grand Prix in Malaysia")["slug"] == "sepang"
    assert resolve_circuit_2d(name="Kuala Lumpur", grand_prix="Bahrain Grand Prix")["slug"] == "sepang"
    assert resolve_circuit_2d(circuit_id="losail", grand_prix="Qatar Grand Prix")["slug"] == "lusail"


# --------------------------------------------------------------------------- #
# 4. White-Box: Cancelled-Round Sequential Numbering & Completion Flag
# --------------------------------------------------------------------------- #
def test_whitebox_openf1_cancelled_rounds_do_not_shift_round_numbers() -> None:
    mock_openf1 = MagicMock()
    # Meetings 4 & 5 are cancelled and omitted by list_races(), while get_sessions() still lists them
    mock_openf1.list_races.return_value = [
        {"meeting_key": 101, "meeting_name": "Australian Grand Prix", "circuit_short_name": "Melbourne", "is_cancelled": False},
        {"meeting_key": 102, "meeting_name": "Chinese Grand Prix", "circuit_short_name": "Shanghai", "is_cancelled": False},
        {"meeting_key": 103, "meeting_name": "Japanese Grand Prix", "circuit_short_name": "Suzuka", "is_cancelled": False},
        {"meeting_key": 106, "meeting_name": "Miami Grand Prix", "circuit_short_name": "Miami", "is_cancelled": False},
        {"meeting_key": 122, "meeting_name": "Qatar Grand Prix", "circuit_short_name": "Lusail", "is_cancelled": False},
    ]
    mock_openf1.get_sessions.return_value = [
        {"session_key": 1, "meeting_key": 101, "date_start": "2026-03-08T04:00:00Z", "date_end": "2026-03-08T06:00:00Z"},
        {"session_key": 2, "meeting_key": 102, "date_start": "2026-03-15T07:00:00Z", "date_end": "2026-03-15T09:00:00Z"},
        {"session_key": 3, "meeting_key": 103, "date_start": "2026-03-29T05:00:00Z", "date_end": "2026-03-29T07:00:00Z"},
        {"session_key": 4, "meeting_key": 104, "date_start": "2026-04-12T15:00:00Z", "date_end": "2026-04-12T17:00:00Z"},  # cancelled
        {"session_key": 5, "meeting_key": 105, "date_start": "2026-04-19T17:00:00Z", "date_end": "2026-04-19T19:00:00Z"},  # cancelled
        {"session_key": 6, "meeting_key": 106, "date_start": "2026-05-03T20:00:00Z", "date_end": "2026-05-03T22:00:00Z"},
        {"session_key": 22, "meeting_key": 122, "date_start": "2099-11-29T16:00:00Z", "date_end": "2099-11-29T18:00:00Z"},
    ]

    engine = ReplayEngine(openf1=mock_openf1)
    sessions = engine.list_available_sessions(2026)

    assert [s["round"] for s in sessions] == [1, 2, 3, 4, 5]
    miami = sessions[3]
    assert miami["grand_prix"] == "Miami Grand Prix"
    assert miami["round"] == 4
    assert miami["circuit_slug"] == "miami"
    assert miami["completed"] is True

    qatar = sessions[4]
    assert qatar["completed"] is False


# --------------------------------------------------------------------------- #
# 5. White-Box: Final Race Classification (Completed vs Not Completed)
# --------------------------------------------------------------------------- #
def test_whitebox_final_race_results_completed_and_upcoming() -> None:
    mock_jolpica = MagicMock()
    mock_openf1 = MagicMock()

    # Case A: Uncompleted future race
    mock_jolpica.get_race_results.return_value = []
    mock_jolpica.get_race.return_value = {
        "season": "2026",
        "round": "22",
        "raceName": "Qatar Grand Prix",
        "date": "2026-11-29",
        "Circuit": {"circuitId": "losail", "circuitName": "Lusail International Circuit"},
    }
    upcoming = get_final_race_results(mock_jolpica, mock_openf1, season=2026, round_number=22)
    assert upcoming["completed"] is False
    assert upcoming["drivers"] == []
    assert upcoming["circuit"]["slug"] == "lusail"

    # Case B: Completed race with winner, +gap, lapped car, and DNF
    mock_jolpica.get_race_results.return_value = [
        {
            "season": "2026",
            "round": "21",
            "raceName": "Las Vegas Grand Prix",
            "date": "2026-11-22",
            "Circuit": {"circuitId": "vegas", "circuitName": "Las Vegas Strip Street Circuit"},
            "Results": [
                {
                    "position": "1",
                    "positionText": "1",
                    "number": "3",
                    "grid": "1",
                    "laps": "50",
                    "status": "Finished",
                    "points": "25",
                    "Driver": {"code": "VER", "givenName": "Max", "familyName": "Verstappen"},
                    "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                    "Time": {"time": "1:27:39.624"},
                    "FastestLap": {"rank": "1", "lap": "48", "Time": {"time": "1:35.591"}},
                },
                {
                    "position": "2",
                    "positionText": "R",
                    "number": "16",
                    "grid": "4",
                    "laps": "22",
                    "status": "Collision",
                    "points": "0",
                    "Driver": {"code": "LEC", "givenName": "Charles", "familyName": "Leclerc"},
                    "Constructor": {"constructorId": "ferrari", "name": "Ferrari"},
                },
            ],
        }
    ]
    mock_openf1.get_stints.return_value = [
        {"driver_number": 3, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 20, "tyre_age_at_start": 0},
        {"driver_number": 3, "stint_number": 2, "compound": "HARD", "lap_start": 21, "lap_end": 50, "tyre_age_at_start": 0},
    ]
    mock_openf1.get_laps.return_value = [
        {"driver_number": 3, "lap_number": 48, "lap_duration": 95.591, "duration_sector_1": 27.5, "duration_sector_2": 32.1, "duration_sector_3": 35.991},
    ]
    mock_openf1.get_weather.return_value = [{"air_temperature": 12.0, "track_temperature": 14.5, "humidity": 40, "rainfall": 0}]
    mock_openf1.get_race_control.return_value = [{"date": "2026-11-22T06:00:00Z", "lap_number": 1, "flag": "GREEN", "message": "GREEN LIGHT"}]
    mock_openf1.get.return_value = [{"date": "2026-11-22T07:00:00Z", "driver_number": 3, "recording_url": "https://livetiming.formula1.com/static/clip.mp3"}]

    done = get_final_race_results(mock_jolpica, mock_openf1, season=2026, round_number=21, session_key=11420)
    assert done["completed"] is True
    assert done["lap_total"] == 50
    assert len(done["drivers"]) == 2
    assert done["drivers"][0]["code"] == "VER"
    assert len(done["drivers"][0]["stints"]) == 2
    assert done["drivers"][0]["sectors"]["s1_status"] == "PURPLE"
    assert done["drivers"][1]["retired"] is True
    assert done["drivers"][1]["gap_display"] == "DNF"


# --------------------------------------------------------------------------- #
# 6. White-Box: SignalR Live `session_active` & Security Gate
# --------------------------------------------------------------------------- #
def test_whitebox_signalr_session_active_and_contract_validation(tmp_path: Path) -> None:
    validator = _validator()
    provider = SignalRLiveProvider(state_path=tmp_path / "state.json", live_token="")
    provider.store.apply_update(
        "DriverList",
        {
            "1": {"RacingNumber": "1", "Tla": "NOR", "FullName": "Lando Norris", "TeamName": "McLaren", "TeamColour": "FF8000"},
            "3": {"RacingNumber": "3", "Tla": "VER", "FullName": "Max Verstappen", "TeamName": "Red Bull", "TeamColour": "3671C6"},
            "16": {"RacingNumber": "16", "Tla": "LEC", "FullName": "Charles Leclerc", "TeamName": "Ferrari", "TeamColour": "E80020"},
            "63": {"RacingNumber": "63", "Tla": "RUS", "FullName": "George Russell", "TeamName": "Mercedes", "TeamColour": "27F4D2"},
        },
    )
    provider.store.apply_update(
        "TimingData",
        {
            "Lines": {
                "1": {"Position": "1", "NumberOfLaps": 50},
                "3": {"Position": "2", "NumberOfLaps": 50},
                "16": {"Position": "3", "NumberOfLaps": 50},
                "63": {"Position": "4", "NumberOfLaps": 50},
            }
        },
    )

    # Active live session (`SessionStatus == "Started"`)
    provider.store.apply_update("SessionInfo", {"Key": 9999, "Name": "Race", "SessionStatus": "Started", "Meeting": {"Key": 50, "Name": "Qatar Grand Prix", "Circuit": {"Key": 150, "ShortName": "Lusail"}}})
    pkt_active = provider.build_packet()
    assert pkt_active["session_active"] is True
    assert pkt_active["capabilities"] == {"gps_available": False, "telemetry_available": False}
    validator.validate(pkt_active)

    # Finalised archived session (`SessionStatus == "Finalised"`)
    provider.store.apply_update("SessionInfo", {"SessionStatus": "Finalised", "ArchiveStatus": {"Status": "Complete"}})
    pkt_idle = provider.build_packet()
    assert pkt_idle["session_active"] is False
    validator.validate(pkt_idle)


# --------------------------------------------------------------------------- #
# 7. White-Box: Radio Transcriber SSRF Guard & Sector State Machine
# --------------------------------------------------------------------------- #
def test_whitebox_radio_ssrf_guard_and_sector_colours(tmp_path: Path) -> None:
    transcriber = RadioTranscriber(cache_path=tmp_path / "radio_cache.json")
    for blocked_url in (
        "http://127.0.0.1:8000/api/health",
        "http://169.254.169.254/latest/meta-data/",
        "https://evil.example.com/team_radio.mp3",
        "file:///C:/Windows/win.ini",
    ):
        with pytest.raises(ValueError):
            transcriber.transcribe_url(blocked_url)

    tracker = SectorAndLapTracker()
    tracker.record_lap(1, "NOR", lap_number=1, lap_time_s=92.0, s1_s=28.5, s2_s=31.0, s3_s=32.5)
    assert tracker.driver_sectors_contract(1)["s1_status"] == "PURPLE"

    # Faster driver steals PURPLE; driver 1 setting a slower sector gets YELLOW, personal best gets GREEN
    tracker.record_lap(3, "VER", lap_number=1, lap_time_s=91.0, s1_s=28.0, s2_s=30.8, s3_s=32.2)
    tracker.record_lap(1, "NOR", lap_number=2, lap_time_s=91.8, s1_s=28.3, s2_s=31.4, s3_s=32.1)
    s_nor = tracker.driver_sectors_contract(1)
    assert s_nor["s1_status"] == "GREEN"   # beat own 28.5 -> 28.3, slower than VER's 28.0
    assert s_nor["s2_status"] == "YELLOW"  # slower than own 31.0 -> 31.4
    assert s_nor["s3_status"] == "PURPLE"  # new overall best 32.1 < 32.2
