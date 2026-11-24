"""Integration tests for Stage 3: FastAPI REST endpoints, CORS, /api/next-race, and WS /ws/live."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import pytest

import config
from clients.jolpica_client import JolpicaClient
from clients.openf1_client import OpenF1Client
from ml.predictor import RacePredictor
from server.main import PitWallRuntime, create_app
from server.schedule import format_multi_timezone, get_next_race_schedule
from streams.replay_engine import ReplayEngine
from streams.signalr_live import SignalRLiveProvider


class _StubJolpica(JolpicaClient):
    """Offline stub for Jolpica schedule & qualifying lookups."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(cache_dir=tmp_path / "jolpica", min_interval_s=0.0)

    def get_schedule(self, season: int) -> List[Dict[str, Any]]:
        if season == 2026:
            return [
                {
                    "season": "2026",
                    "round": "22",
                    "raceName": "Las Vegas Grand Prix",
                    "Circuit": {
                        "circuitId": "vegas",
                        "circuitName": "Las Vegas Strip Street Circuit",
                        "Location": {"locality": "Las Vegas", "country": "USA"},
                    },
                    "date": "2026-11-22",
                    "time": "04:00:00Z",
                },
                {
                    "season": "2026",
                    "round": "23",
                    "raceName": "Qatar Grand Prix",
                    "Circuit": {
                        "circuitId": "losail",
                        "circuitName": "Lusail International Circuit",
                        "Location": {"locality": "Al Daayen", "country": "Qatar"},
                    },
                    "date": "2026-11-29",
                    "time": "16:00:00Z",
                    "Sprint": {"date": "2026-11-28", "time": "14:00:00Z"},
                    "Qualifying": {"date": "2026-11-28", "time": "18:00:00Z"},
                },
            ]
        return []

    def get_qualifying_results(self, season: int, round_number: int | None = None) -> List[Dict[str, Any]]:
        return [
            {
                "season": str(season),
                "round": str(round_number or 23),
                "raceName": "Qatar Grand Prix",
                "date": "2026-11-29",
                "Circuit": {"circuitId": "losail", "circuitName": "Lusail International Circuit"},
                "QualifyingResults": [
                    {"number": "1", "position": "1", "Driver": {"driverId": "ver", "code": "VER", "givenName": "Max", "familyName": "Verstappen"}, "Constructor": {"constructorId": "red_bull", "name": "Red Bull Racing"}, "Q3": "1:20.100"},
                    {"number": "4", "position": "2", "Driver": {"driverId": "nor", "code": "NOR", "givenName": "Lando", "familyName": "Norris"}, "Constructor": {"constructorId": "mclaren", "name": "McLaren"}, "Q3": "1:20.220"},
                    {"number": "16", "position": "3", "Driver": {"driverId": "lec", "code": "LEC", "givenName": "Charles", "familyName": "Leclerc"}, "Constructor": {"constructorId": "ferrari", "name": "Ferrari"}, "Q3": "1:20.310"},
                    {"number": "12", "position": "4", "Driver": {"driverId": "ant", "code": "ANT", "givenName": "Kimi", "familyName": "Antonelli"}, "Constructor": {"constructorId": "mercedes", "name": "Mercedes"}, "Q3": "1:20.490"},
                    {"number": "81", "position": "5", "Driver": {"driverId": "pia", "code": "PIA", "givenName": "Oscar", "familyName": "Piastri"}, "Constructor": {"constructorId": "mclaren", "name": "McLaren"}, "Q3": "1:20.550"},
                ],
            }
        ]

    def get_race_results(self, season: int, round_number: int | None = None) -> List[Dict[str, Any]]:
        return []


class _StubOpenF1(OpenF1Client):
    """Offline stub for OpenF1 meetings/sessions."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(cache_dir=tmp_path / "openf1", min_interval_s=0.0)

    def get(self, endpoint: str, params=None, *, ttl_s=None, use_cache=True):
        if endpoint == "meetings":
            return [
                {"meeting_key": 1298, "meeting_name": "Cancelled Round", "circuit_short_name": "Nowhere", "date_start": "2026-04-01T00:00:00Z", "is_cancelled": True},
                {"meeting_key": 1299, "meeting_name": "Las Vegas Grand Prix", "circuit_short_name": "Las Vegas", "date_start": "2026-11-20T00:00:00Z", "is_cancelled": False},
            ]
        if endpoint == "sessions":
            return [
                {"session_key": 9998, "meeting_key": 1298, "year": 2026, "session_name": "Race", "date_start": "2026-04-03T12:00:00Z", "date_end": "2026-04-03T14:00:00Z"},
                {"session_key": 9999, "meeting_key": 1299, "year": 2026, "session_name": "Race", "circuit_short_name": "Las Vegas", "date_start": "2026-11-22T04:00:00Z", "date_end": "2026-11-22T05:30:00Z"},
            ]
        return []


def _seed_synthetic_replay(engine: ReplayEngine) -> None:
    t0 = 1763784000.0  # 2026-11-22T04:00:00Z

    def iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    drivers = [
        {"driver_number": 1, "name_acronym": "VER", "full_name": "Max Verstappen", "team_name": "Red Bull Racing", "team_colour": "3671C6"},
        {"driver_number": 4, "name_acronym": "NOR", "full_name": "Lando Norris", "team_name": "McLaren", "team_colour": "FF8000"},
        {"driver_number": 16, "name_acronym": "LEC", "full_name": "Charles Leclerc", "team_name": "Ferrari", "team_colour": "E80020"},
        {"driver_number": 12, "name_acronym": "ANT", "full_name": "Kimi Antonelli", "team_name": "Mercedes", "team_colour": "27F4D2"},
    ]
    laps, positions, intervals, stints, locations, car_data = [], [], [], [], [], []
    for idx, d in enumerate(drivers, start=1):
        dn = d["driver_number"]
        stints.append({"driver_number": dn, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "tyre_age_at_start": 0})
        for lap in range(1, 7):
            ls = t0 + (lap - 1) * 90.0 + idx * 0.4
            laps.append({"driver_number": dn, "lap_number": lap, "date_start": iso(ls), "lap_duration": 90.0 + idx * 0.1, "is_pit_out_lap": False})
            positions.append({"driver_number": dn, "date": iso(ls), "position": idx})
            intervals.append({"driver_number": dn, "date": iso(ls), "gap_to_leader": (idx - 1) * 1.5, "interval": 0.0 if idx == 1 else 1.5})
        for step in range(40):
            ts = t0 + 180.0 + step * 1.0
            ang = (step / 40.0) * 2.0 * math.pi
            locations.append({"driver_number": dn, "date": iso(ts), "x": 800.0 * math.cos(ang), "y": 400.0 * math.sin(ang), "z": 5.0})
            car_data.append({"driver_number": dn, "date": iso(ts), "speed": 300.0, "throttle": 100.0, "brake": 0.0, "n_gear": 8, "rpm": 11800, "drs": None})

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
        end_s=t0 + 600.0,
        drivers=drivers,
        laps=laps,
        positions=positions,
        intervals=intervals,
        stints=stints,
        pits=[],
        locations=locations,
        car_data=car_data,
    )


@pytest.fixture
def validator() -> Draft202012Validator:
    schema = json.loads(config.CONTRACT_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


@pytest.fixture
def test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "ALLOW_LIVE_NETWORK", False)
    jolpica = _StubJolpica(tmp_path)
    openf1 = _StubOpenF1(tmp_path)
    predictor = RacePredictor(
        jolpica=jolpica,
        openf1=openf1,
        pre_race_path=tmp_path / "missing_pre.joblib",
        live_race_path=tmp_path / "missing_live.joblib",
        history_path=tmp_path / "missing_hist.parquet",
    )
    replay = ReplayEngine(openf1=openf1, predictor=predictor, tick_interval_s=0.1)
    _seed_synthetic_replay(replay)
    live = SignalRLiveProvider(
        predictor=predictor,
        state_path=tmp_path / "last_state.json",
        live_token="",
        emit_interval_s=0.25,
    )
    runtime = PitWallRuntime(
        jolpica=jolpica,
        openf1=openf1,
        predictor=predictor,
        replay=replay,
        live=live,
        initial_mode="replay",
    )
    app = create_app(runtime, mount_frontend=False)
    with TestClient(app) as client:
        yield client


def test_next_race_timezone_conversions(tmp_path: Path) -> None:
    # 2026-11-29 16:00 UTC -> IST 21:30 (UTC+5:30), UK 16:00 GMT, US Eastern 11:00 EST, US Central 10:00 CST, US Pacific 08:00 PST
    dt_utc = datetime(2026, 11, 29, 16, 0, tzinfo=timezone.utc)
    times = format_multi_timezone(dt_utc)
    assert times["ist"]["time_24h"] == "21:30"
    assert times["uk"]["time_24h"] == "16:00"
    assert times["us_eastern"]["time_24h"] == "11:00"
    assert times["us_central"]["time_24h"] == "10:00"
    assert times["us_pacific"]["time_24h"] == "08:00"

    jolpica = _StubJolpica(tmp_path)
    now = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
    next_race = get_next_race_schedule(jolpica, season=2026, now_utc=now)
    assert next_race["grand_prix"] == "Qatar Grand Prix"
    assert next_race["round"] == 23
    assert next_race["status"] == "upcoming"
    assert next_race["times"]["ist"]["time_24h"] == "21:30"
    assert next_race["times"]["uk"]["time_24h"] == "16:00"
    assert next_race["times"]["us_eastern"]["time_24h"] == "11:00"
    assert next_race["times"]["us_pacific"]["time_24h"] == "08:00"
    session_names = [s["name"] for s in next_race["sessions"]]
    assert "Qualifying" in session_names
    assert "Sprint" in session_names
    assert "Grand Prix" in session_names


def test_api_health_sessions_next_race_and_cors(test_client: TestClient) -> None:
    # Health + CORS header check for chrome-extension:// origin
    res = test_client.get("/api/health", headers={"Origin": "chrome-extension://abcdefghijklmnop"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "chrome-extension://abcdefghijklmnop"
    data = res.json()
    assert data["status"] == "ok"
    assert data["mode"] == "replay"
    assert data["replay"]["loaded"] is True

    # Sessions endpoint filters out cancelled 2026 round
    sess_res = test_client.get("/api/sessions?year=2026")
    assert sess_res.status_code == 200
    sessions = sess_res.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_key"] == 9999
    assert sessions[0]["grand_prix"] == "Las Vegas Grand Prix"

    # Next race endpoint returns IST, UK, and US times
    nr_res = test_client.get("/api/next-race?season=2026")
    assert nr_res.status_code == 200
    nr = nr_res.json()
    assert nr["grand_prix"] == "Qatar Grand Prix"
    assert {"ist", "uk", "us_eastern", "us_central", "us_pacific"} <= set(nr["times"].keys())


def test_post_quali_control_mode_switch_and_websocket(
    test_client: TestClient, validator: Draft202012Validator
) -> None:
    # 1. Replay control actions
    ctrl = test_client.post("/api/replay/control", json={"action": "set_speed", "speed": 5.0})
    assert ctrl.status_code == 200
    assert ctrl.json()["replay"]["speed"] == 5.0

    ctrl_seek = test_client.post("/api/replay/control", json={"action": "seek_lap", "lap": 4})
    assert ctrl_seek.status_code == 200

    ctrl_pause = test_client.post("/api/replay/control", json={"action": "pause"})
    assert ctrl_pause.status_code == 200
    assert ctrl_pause.json()["replay"]["paused"] is True

    # 2. WebSocket receives contract-valid replay_frame
    with test_client.websocket_connect("/ws/live") as ws:
        frame = ws.receive_json()
        validator.validate(frame)
        assert frame["type"] == "replay_frame"
        assert frame["capabilities"] == {"gps_available": True, "telemetry_available": True}
        active_sum = sum(d["podium_prob_pct"] for d in frame["race"]["drivers"] if not d["retired"])
        assert pytest.approx(active_sum, abs=0.05) == 300.0

    # 3. Post-quali Top-3 prediction
    pq_res = test_client.post("/api/predict/post-quali", json={"season": 2026, "round": 23})
    assert pq_res.status_code == 200
    pq_pkt = pq_res.json()
    validator.validate(pq_pkt)
    assert pq_pkt["type"] == "post_quali_prediction"
    items = pq_pkt["predictions"]["items"]
    podium_vals = [i["podium_prob_pct"] for i in items]
    assert podium_vals == sorted(podium_vals, reverse=True)
    assert pytest.approx(sum(podium_vals), abs=0.05) == 300.0
    assert pytest.approx(sum(i["win_prob_pct"] for i in items), abs=0.05) == 100.0

    # 4. Mode switch to free_live -> graceful locked GPS/telemetry
    mode_res = test_client.post("/api/mode", json={"mode": "free_live"})
    assert mode_res.status_code == 200
    assert mode_res.json()["mode"] == "free_live"

    with test_client.websocket_connect("/ws/live") as ws:
        live_pkt = ws.receive_json()
        validator.validate(live_pkt)
        assert live_pkt["type"] == "live_race_update"
        assert live_pkt["mode"] == "free_live"
        assert live_pkt["capabilities"] == {"gps_available": False, "telemetry_available": False}
