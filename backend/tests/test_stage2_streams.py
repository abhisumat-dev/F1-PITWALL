"""Unit tests for Stage 2: TyreDegTracker, ReplayEngine, and SignalRLiveProvider."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator
import pytest

import config
from analytics.tyre_deg import TyreDegTracker, classify_tyre_health
from streams.replay_engine import ReplayEngine, drs_to_active, rotate_xy
from streams.signalr_live import (
    RECORD_SEP,
    SignalRLiveProvider,
    SignalRStateStore,
    deep_merge,
    normalise_arrays,
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(config.CONTRACT_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# --------------------------------------------------------------------------- #
# TyreDegTracker tests
# --------------------------------------------------------------------------- #
def test_tyre_deg_excludes_lap1_pit_and_non_green_and_applies_fuel_correction() -> None:
    tracker = TyreDegTracker(min_laps=4, window_laps=8, fuel_correction_s_per_lap=0.035)

    # Lap 1 (must be excluded)
    tracker.record_lap(1, 1, 105.0, stint_number=1)
    # Pit-out lap 2 (must be excluded)
    tracker.record_lap(1, 2, 112.0, stint_number=1, is_pit_out_lap=True)
    # Safety car lap 3 (must be excluded)
    tracker.record_lap(1, 3, 130.0, stint_number=1, track_status="SC")
    # VSC / YELLOW / RED laps (must be excluded)
    tracker.record_lap(1, 4, 118.0, stint_number=1, track_status="VSC")
    tracker.record_lap(1, 5, 102.0, stint_number=1, track_status="YELLOW")
    tracker.record_lap(1, 6, 150.0, stint_number=1, track_status="RED")
    assert tracker.get_driver_deg(1) is None

    # Now feed 5 clean laps where raw lap time stays flat at 90.0s.
    # Because fuel burn gives -0.035s/lap, flat raw times mean true tyre deg is +0.035 s/lap (OPTIMAL < 0.08).
    for lap in range(7, 12):
        tracker.record_lap(1, lap, 90.0, stint_number=1, track_status="GREEN")

    res = tracker.get_driver_deg(1)
    assert res is not None
    assert res.sample_laps == 5
    assert res.fuel_corrected is True
    assert pytest.approx(res.slope_s_per_lap, abs=1e-4) == 0.035
    assert res.tyre_health == "OPTIMAL"

    # Pit-in lap 12 (excluded)
    tracker.record_lap(1, 12, 109.0, stint_number=1, is_pit_in_lap=True)
    assert tracker.get_driver_deg(1).sample_laps == 5  # unchanged

    # New stint 2 with DEGRADING tyres: raw times slow by +0.085 s/lap -> fuel-corrected slope = +0.120 s/lap
    for idx, lap in enumerate(range(14, 19)):
        tracker.record_lap(1, lap, 89.0 + idx * 0.085, stint_number=2, track_status="GREEN")
    res2 = tracker.get_driver_deg(1)
    assert res2 is not None
    assert res2.stint_number == 2
    assert pytest.approx(res2.slope_s_per_lap, abs=1e-4) == 0.120
    assert res2.tyre_health == "DEGRADING"

    # New stint 3 hitting CLIFF (> 0.16 s/lap): raw slows by +0.18 s/lap -> corrected = +0.215 s/lap
    for idx, lap in enumerate(range(25, 30)):
        tracker.record_lap(1, lap, 88.5 + idx * 0.18, stint_number=3, track_status="GREEN")
    res3 = tracker.get_driver_deg(1)
    assert res3 is not None
    assert pytest.approx(res3.slope_s_per_lap, abs=1e-4) == 0.215
    assert res3.tyre_health == "CLIFF"


def test_classify_tyre_health_boundaries() -> None:
    assert classify_tyre_health(0.079) == "OPTIMAL"
    assert classify_tyre_health(0.08) == "DEGRADING"
    assert classify_tyre_health(0.16) == "DEGRADING"
    assert classify_tyre_health(0.161) == "CLIFF"
    assert classify_tyre_health(None) == "UNKNOWN"


# --------------------------------------------------------------------------- #
# ReplayEngine tests
# --------------------------------------------------------------------------- #
def test_rotate_xy_and_nullable_drs() -> None:
    rx, ry = rotate_xy(100.0, 0.0, 90.0)
    assert pytest.approx(rx, abs=0.05) == 0.0
    assert pytest.approx(ry, abs=0.05) == 100.0
    assert drs_to_active(None) is None
    assert drs_to_active(0) is False
    assert drs_to_active(12) is True


def test_replay_engine_4hz_frame_contract_rotation_and_controls(validator: Draft202012Validator) -> None:
    engine = ReplayEngine(tick_interval_s=0.25)
    t0 = 1763780400.0  # 2026-11-22T03:00:00Z

    drivers = [
        {"driver_number": 1, "name_acronym": "VER", "full_name": "Max Verstappen", "team_name": "Red Bull Racing", "team_colour": "3671C6"},
        {"driver_number": 4, "name_acronym": "NOR", "full_name": "Lando Norris", "team_name": "McLaren", "team_colour": "FF8000"},
        {"driver_number": 16, "name_acronym": "LEC", "full_name": "Charles Leclerc", "team_name": "Ferrari", "team_colour": "E80020"},
        {"driver_number": 12, "name_acronym": "ANT", "full_name": "Kimi Antonelli", "team_name": "Mercedes", "team_colour": "27F4D2"},
        {"driver_number": 81, "name_acronym": "PIA", "full_name": "Oscar Piastri", "team_name": "McLaren", "team_colour": "FF8000"},
    ]

    from datetime import datetime, timezone

    def iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    laps: List[Dict[str, Any]] = []
    positions: List[Dict[str, Any]] = []
    intervals: List[Dict[str, Any]] = []
    stints: List[Dict[str, Any]] = []
    locations: List[Dict[str, Any]] = []
    car_data: List[Dict[str, Any]] = []

    for pos_idx, d in enumerate(drivers, start=1):
        dn = d["driver_number"]
        stints.append({"driver_number": dn, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "tyre_age_at_start": 0})
        for lap in range(1, 9):
            ls = t0 + (lap - 1) * 90.0 + pos_idx * 0.5
            dur = 90.0 + pos_idx * 0.15 + lap * 0.02
            laps.append(
                {
                    "driver_number": dn,
                    "lap_number": lap,
                    "date_start": iso(ls),
                    "lap_duration": dur,
                    "duration_sector_1": 26.5 + pos_idx * 0.05,
                    "duration_sector_2": 31.0 + pos_idx * 0.05,
                    "duration_sector_3": dur - (26.5 + pos_idx * 0.05) - (31.0 + pos_idx * 0.05),
                    "is_pit_out_lap": False,
                }
            )
            positions.append({"driver_number": dn, "date": iso(ls), "position": pos_idx})
            intervals.append({"driver_number": dn, "date": iso(ls), "gap_to_leader": (pos_idx - 1) * 1.4, "interval": 0.0 if pos_idx == 1 else 1.4})
        # High-frequency location & car_data (2026 null drs)
        for step in range(80):
            ts = t0 + 180.0 + step * 1.0
            angle = (step / 80.0) * 2.0 * math.pi
            locations.append({"driver_number": dn, "date": iso(ts), "x": 1000.0 * math.cos(angle), "y": 500.0 * math.sin(angle), "z": 10.0})
            car_data.append({"driver_number": dn, "date": iso(ts), "speed": 290.0, "throttle": 100.0, "brake": 0.0, "n_gear": 7, "rpm": 11500, "drs": None})

    engine.load_from_raw(
        session_key=9999,
        meeting_key=1299,
        season=2026,
        round_number=22,
        grand_prix="Las Vegas Grand Prix",
        circuit_id="vegas",
        circuit_name="Las Vegas",
        rotation_deg=90.0,
        start_s=t0,
        end_s=t0 + 750.0,
        drivers=drivers,
        laps=laps,
        positions=positions,
        intervals=intervals,
        stints=stints,
        pits=[],
        team_radio=[
            {
                "driver_number": 1,
                "date": iso(t0 + 200.0),
                "recording_url": "https://livetiming.formula1.com/static/radio_ver_1.mp3",
            }
        ],
        locations=locations,
        car_data=car_data,
        session_result=[
            {"driver_number": 81, "position": 5, "number_of_laps": 3, "dnf": True}
        ],
    )

    # Controls: speed clamping 1x..10x, seek_lap, pause/play
    engine.set_speed(15.0)
    assert engine.status()["speed"] == 10.0
    engine.set_speed(0.1)
    assert engine.status()["speed"] == 1.0
    engine.set_speed(4.0)
    assert engine.status()["speed"] == 4.0

    engine.seek_lap(7)
    frame = engine.build_frame()
    validator.validate(frame)

    assert frame["type"] == "replay_frame"
    assert frame["capabilities"] == {"gps_available": True, "telemetry_available": True}
    assert frame["circuit"]["slug"] == "las_vegas"
    assert frame["circuit"]["track_outline"] is not None
    assert len(frame["circuit"]["track_outline"]) == 260
    assert len(frame["circuit"]["corners"]) == 17
    assert len(frame["circuit"]["sector_splits"]) == 4

    # Session fastest S1, S2, S3, Fastest Lap & Team Radio verified
    sb = frame["race"]["session_best"]
    assert sb["fastest_lap"]["driver_code"] == "VER"
    assert sb["fastest_s1"]["driver_code"] == "VER"
    assert sb["fastest_s2"]["driver_code"] == "VER"
    assert sb["fastest_s3"]["driver_code"] == "VER"
    assert len(frame["race"]["team_radio"]) == 1
    assert frame["race"]["team_radio"][0]["driver_code"] == "VER"

    # 2026 nullable DRS preserved as None
    ver = next(d for d in frame["race"]["drivers"] if d["driver_number"] == 1)
    assert ver["telemetry"] is not None
    assert ver["telemetry"]["drs"] is None
    assert ver["telemetry"]["drs_active"] is None
    assert ver["tyre_deg"] is not None
    assert ver["tyre_health"] in {"OPTIMAL", "DEGRADING", "CLIFF"}

    # Retired driver #81 gets 0.0%, active 4 drivers sum to 300.0%
    pia = next(d for d in frame["race"]["drivers"] if d["driver_number"] == 81)
    assert pia["retired"] is True
    assert pia["podium_prob_pct"] == 0.0
    active_podium_sum = sum(d["podium_prob_pct"] for d in frame["race"]["drivers"] if not d["retired"])
    assert pytest.approx(active_podium_sum, abs=0.05) == 300.0


# --------------------------------------------------------------------------- #
# SignalRLiveProvider tests (differential patch merge + atomic crash recovery)
# --------------------------------------------------------------------------- #
def test_signalr_deep_merge_and_atomic_persistence(tmp_path: Path, validator: Draft202012Validator) -> None:
    # Verify deep_merge and normalise_arrays
    base = {"Lines": {"1": {"Position": "1", "GapToLeader": "", "Sectors": {"0": {"Value": "28.1"}}}}}
    patch = {"Lines": {"1": {"GapToLeader": "+0.000", "Sectors": {"0": {"PersonalFastest": True}}}}}
    merged = deep_merge(base, patch)
    assert merged["Lines"]["1"]["Position"] == "1"
    assert merged["Lines"]["1"]["GapToLeader"] == "+0.000"
    assert merged["Lines"]["1"]["Sectors"]["0"]["Value"] == "28.1"
    assert merged["Lines"]["1"]["Sectors"]["0"]["PersonalFastest"] is True
    assert normalise_arrays({"1": "b", "0": "a"}) == ["a", "b"]

    state_file = tmp_path / "last_state.json"
    provider = SignalRLiveProvider(state_path=state_file, live_token="")

    # Feed SignalR Type 3 initial snapshot + Type 1 differential delta + Type 6 ping
    subscribe_snapshot = {
        "type": 3,
        "invocationId": "0",
        "result": {
            "SessionInfo": {
                "Key": 11250,
                "Name": "Race",
                "StartDate": "2026-11-22T04:00:00",
                "Meeting": {"Key": 1299, "Number": 22, "Name": "Las Vegas Grand Prix", "Circuit": {"Key": 152, "ShortName": "Las Vegas"}},
            },
            "DriverList": {
                "1": {"RacingNumber": "1", "FullName": "Max Verstappen", "Tla": "VER", "TeamName": "Red Bull Racing", "TeamColour": "3671C6"},
                "4": {"RacingNumber": "4", "FullName": "Lando Norris", "Tla": "NOR", "TeamName": "McLaren", "TeamColour": "FF8000"},
                "16": {"RacingNumber": "16", "FullName": "Charles Leclerc", "Tla": "LEC", "TeamName": "Ferrari", "TeamColour": "E80020"},
                "12": {"RacingNumber": "12", "FullName": "Kimi Antonelli", "Tla": "ANT", "TeamName": "Mercedes", "TeamColour": "27F4D2"},
            },
            "TimingData": {
                "Lines": {
                    "1": {
                        "Position": "1",
                        "GapToLeader": "",
                        "NumberOfLaps": 12,
                        "LastLapTime": {"Value": "1:33.100"},
                        "BestLapTime": {"Value": "1:32.008", "Lap": 11},
                        "Sectors": {"0": {"Value": "26.812"}, "1": {"Value": "31.200"}, "2": {"Value": "33.996"}},
                    },
                    "4": {
                        "Position": "2",
                        "GapToLeader": "+1.850",
                        "IntervalToPositionAhead": {"Value": "+1.850"},
                        "NumberOfLaps": 12,
                        "Sectors": {"0": {"Value": "26.950"}, "1": {"Value": "31.044"}, "2": {"Value": "34.110"}},
                    },
                    "16": {
                        "Position": "3",
                        "GapToLeader": "+4.200",
                        "IntervalToPositionAhead": {"Value": "+2.350"},
                        "NumberOfLaps": 12,
                        "Sectors": {"0": {"Value": "27.010"}, "1": {"Value": "31.300"}, "2": {"Value": "33.918"}},
                    },
                    "12": {"Position": "4", "GapToLeader": "+6.100", "IntervalToPositionAhead": {"Value": "+1.900"}, "NumberOfLaps": 12},
                }
            },
            "TimingAppData": {
                "Lines": {
                    "1": {"Stints": {"0": {"Compound": "MEDIUM", "TotalLaps": 12, "New": "true"}}},
                    "4": {"Stints": {"0": {"Compound": "HARD", "TotalLaps": 12, "New": "true"}}},
                    "16": {"Stints": {"0": {"Compound": "MEDIUM", "TotalLaps": 12, "New": "true"}}},
                    "12": {"Stints": {"0": {"Compound": "HARD", "TotalLaps": 12, "New": "true"}}},
                }
            },
            "LapCount": {"CurrentLap": 12, "TotalLaps": 50},
            "WeatherData": {"AirTemp": "17.2", "TrackTemp": "21.5", "Humidity": "38", "Pressure": "944.0", "WindSpeed": "1.2", "WindDirection": "180", "Rainfall": "0"},
            "TrackStatus": {"Status": "1", "Message": "AllClear"},
            "RaceControlMessages": {"Messages": {"0": {"Utc": "2026-11-22T04:01:00Z", "Lap": 1, "Category": "Flag", "Flag": "GREEN", "Message": "GREEN LIGHT"}}},
            "TeamRadio": {"Captures": {"0": {"Utc": "2026-11-22T04:12:30Z", "RacingNumber": "4", "Path": "TeamRadio/LANNOR01.mp3"}}},
        },
    }
    delta_frame = {
        "type": 1,
        "target": "feed",
        "arguments": ["TimingData", {"Lines": {"4": {"GapToLeader": "+1.400"}}}],
    }
    ping_frame = {"type": 6}

    raw_payload = (
        json.dumps(subscribe_snapshot)
        + RECORD_SEP
        + json.dumps(delta_frame)
        + RECORD_SEP
        + json.dumps(ping_frame)
        + RECORD_SEP
    )
    replies = provider.handle_raw_frames(raw_payload)
    assert replies == [{"type": 6}]

    packet = provider.build_packet()
    validator.validate(packet)

    assert packet["type"] == "live_race_update"
    assert packet["mode"] == "free_live"
    assert packet["telemetry_unlocked"] is False
    assert packet["capabilities"] == {"gps_available": False, "telemetry_available": False}
    assert packet["circuit"]["slug"] == "las_vegas"
    assert len(packet["circuit"]["track_outline"]) == 260
    assert packet["race"]["session_best"]["fastest_lap"]["driver_code"] == "VER"
    assert packet["race"]["session_best"]["fastest_s1"]["driver_code"] == "VER"
    assert packet["race"]["session_best"]["fastest_s2"]["driver_code"] == "NOR"
    assert packet["race"]["session_best"]["fastest_s3"]["driver_code"] == "LEC"
    assert len(packet["race"]["team_radio"]) == 1
    assert packet["race"]["team_radio"][0]["driver_code"] == "NOR"

    # Verify locked GPS/telemetry fields are strictly null in free_live mode
    for d in packet["race"]["drivers"]:
        assert d["x"] is None
        assert d["y"] is None
        assert d["telemetry"] is None

    # Verify differential patch applied
    nor = next(d for d in packet["race"]["drivers"] if d["driver_number"] == 4)
    assert nor["gap_to_leader_s"] == 1.4

    # Verify Top-3 podium probabilities sum to 300.0%
    podium_sum = sum(d["podium_prob_pct"] for d in packet["race"]["drivers"])
    assert pytest.approx(podium_sum, abs=0.05) == 300.0

    # Verify atomic crash-recovery persistence & reload
    provider.store.save_atomic()
    assert state_file.exists()
    reloaded = SignalRStateStore(state_path=state_file)
    assert reloaded.lap_count["CurrentLap"] == 12
    assert reloaded.timing_data["Lines"]["4"]["GapToLeader"] == "+1.400"
