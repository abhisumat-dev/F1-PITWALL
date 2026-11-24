"""Unauthenticated F1 SignalR Core live timing provider ($0 Architecture).

Ported selectively from ``harsh779/f1-live-api`` (``client.js``, ``merge.js``,
``state.js``, and ``persistence.js``):

Included:
- Unauthenticated SignalR Core negotiate (``POST /signalrcore/negotiate?negotiateVersion=1``)
  forwarding sticky-session ``AWSALB``/``AWSALBCORS`` cookies on the WebSocket upgrade.
- ``\\x1e``-delimited JSON SignalR Core handshake, Type 1 invocations, Type 3
  initial-snapshot completion frame, and Type 6 keepalive pings.
- Subscribes to free topics: ``TimingData``, ``TimingAppData``, ``LapCount``,
  ``WeatherData``, ``RaceControlMessages``, ``DriverList`` (plus ``TrackStatus``,
  ``SessionInfo``, ``Heartbeat`` for live flag/metadata integrity).
- Optional ``F1_LIVE_TOKEN`` credential hook unlocks ``Position.z`` and ``CarData.z``
  (base64 + raw zlib deflate). Without ``F1_LIVE_TOKEN``, strictly sets
  ``gps_available=False``, ``telemetry_available=False``, ``x=None``, ``y=None``,
  ``telemetry=None`` in ``free_live`` mode.
- Recursive differential JSON patch merger (``deep_merge`` & ``normalise_arrays``).
- Atomic crash-recovery persistence to ``backend/cache/last_state.json`` using
  write-to-temp + ``os.replace``.

Explicitly excluded:
- ``livetiming.formula1.com/static/`` backfill (blocked by HTTP 401 without F1 TV).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx
import websockets

import config
from analytics.tyre_deg import TyreDegTracker
from clients.jolpica_client import parse_lap_time
from ml.features import normalize_podium_to_percent, normalize_to_percent
from ml.predictor import RaceContext, RacePredictor, is_driver_out
from streams.base_provider import BaseStreamProvider

LOGGER = logging.getLogger(__name__)

RECORD_SEP = "\x1e"

BASE_HEADERS = {
    "User-Agent": "BestHTTP",
    "Accept-Encoding": "gzip, identity",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.formula1.com",
}

DEFAULT_LAST_STATE_PATH: Path = config.CACHE_DIR / "last_state.json"

__all__ = [
    "SignalRLiveProvider",
    "SignalRStateStore",
    "deep_merge",
    "normalise_arrays",
    "decompress_topic",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip().lstrip("+")
    if not text or text in {"-", "LAP", "LAPS", "None", "null"}:
        return None
    if "L" in text.upper():
        return None
    try:
        val = float(text)
        return val if math.isfinite(val) else None
    except ValueError:
        return parse_lap_time(text)


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _normalize_compound(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    if text in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}:
        return text
    if text.startswith("S"):
        return "SOFT"
    if text.startswith("M"):
        return "MEDIUM"
    if text.startswith("H"):
        return "HARD"
    if text.startswith("I"):
        return "INTERMEDIATE"
    if text.startswith("W"):
        return "WET"
    return "UNKNOWN"


def _normalize_hex(colour: Any) -> Optional[str]:
    if not colour:
        return None
    cleaned = str(colour).strip().lstrip("#").upper()
    if len(cleaned) == 6 and all(c in "0123456789ABCDEF" for c in cleaned):
        return cleaned
    return None


def _map_track_status(track_status: Mapping[str, Any]) -> str:
    """Map F1 SignalR TrackStatus codes/messages to contract enum."""
    code = str(track_status.get("Status") or "").strip()
    msg = str(track_status.get("Message") or "").strip().upper()
    if code == "1" or "ALLCLEAR" in msg or "CLEAR" in msg:
        return "GREEN"
    if code == "2" or "YELLOW" in msg:
        return "YELLOW"
    if code == "4" or ("SC" in msg and "VSC" not in msg) or "SAFETY CAR" in msg:
        return "SC"
    if code == "5" or "RED" in msg:
        return "RED"
    if code in {"6", "7"} or "VSC" in msg or "VIRTUAL" in msg:
        return "VSC"
    return "GREEN" if not code and not msg else "UNKNOWN"


# --------------------------------------------------------------------------- #
# Recursive differential JSON patch merger (ported from merge.js)
# --------------------------------------------------------------------------- #
def deep_merge(target: Any, patch: Any) -> Any:
    """Recursively merge F1 differential JSON patch into ``target`` without mutating ``target``."""
    if patch is None:
        return target
    if not isinstance(patch, dict):
        return patch
    result: Dict[str, Any] = dict(target) if isinstance(target, dict) else {}
    for key, patch_val in patch.items():
        if key == "_deleted" and isinstance(patch_val, list):
            for del_key in patch_val:
                result.pop(del_key, None)
            continue
        target_val = result.get(key)
        if isinstance(patch_val, dict) and isinstance(target_val, dict):
            result[key] = deep_merge(target_val, patch_val)
        else:
            result[key] = patch_val
    return result


def normalise_arrays(obj: Any) -> Any:
    """Convert F1 numeric-key dicts (``{"0": a, "1": b}``) into ordered lists."""
    if not isinstance(obj, dict):
        return obj
    keys = list(obj.keys())
    if keys and all(re.fullmatch(r"\d+", str(k)) for k in keys):
        ordered = sorted(((int(k), k) for k in keys), key=lambda pair: pair[0])
        return [normalise_arrays(obj[k]) for _, k in ordered]
    return {k: normalise_arrays(v) for k, v in obj.items()}


def decompress_topic(data: Any) -> Any:
    """Decode base64 + raw zlib deflate payload from ``.z`` topics (when token unlocked)."""
    if not isinstance(data, str):
        return data
    try:
        raw = base64.b64decode(data)
        inflated = zlib.decompress(raw, -zlib.MAX_WBITS)
        return json.loads(inflated.decode("utf-8"))
    except Exception:  # noqa: BLE001
        try:
            return json.loads(data)
        except Exception:  # noqa: BLE001
            return data


# --------------------------------------------------------------------------- #
# State store with atomic last_state.json crash-recovery persistence
# --------------------------------------------------------------------------- #
class SignalRStateStore:
    """Maintains merged SignalR topic state and persists atomically to ``last_state.json``."""

    def __init__(self, state_path: Path = DEFAULT_LAST_STATE_PATH, persist_interval_s: float = 10.0) -> None:
        self.state_path = Path(state_path)
        self.persist_interval_s = float(persist_interval_s)
        self._last_persist_at = 0.0

        self.connected: bool = False
        self.last_update: Optional[str] = None
        self.session_info: Dict[str, Any] = {}
        self.driver_list: Dict[str, Any] = {}
        self.timing_data: Dict[str, Any] = {}
        self.timing_app_data: Dict[str, Any] = {}
        self.lap_count: Dict[str, Any] = {}
        self.weather_data: Dict[str, Any] = {}
        self.track_status: Dict[str, Any] = {}
        self.race_control: Dict[str, Any] = {"Messages": []}
        self.car_data: Dict[str, Any] = {}
        self.position: Dict[str, Any] = {}

        self.restore()

    def _reset_session_scoped(self) -> None:
        self.timing_data = {}
        self.timing_app_data = {}
        self.lap_count = {}
        self.race_control = {"Messages": []}
        self.car_data = {}
        self.position = {}

    def apply_update(self, topic: str, data: Any) -> None:
        """Apply a SignalR full snapshot or differential patch for ``topic``."""
        if data is None:
            return
        self.last_update = _now_iso()

        if topic == "SessionInfo":
            incoming_key = data.get("Key") if isinstance(data, dict) else None
            current_key = self.session_info.get("Key")
            if incoming_key is not None and current_key is not None and incoming_key != current_key:
                self._reset_session_scoped()
            self.session_info = deep_merge(self.session_info, data)
        elif topic == "DriverList":
            self.driver_list = deep_merge(self.driver_list, data)
        elif topic == "TimingData":
            self.timing_data = deep_merge(self.timing_data, data)
        elif topic == "TimingAppData":
            self.timing_app_data = deep_merge(self.timing_app_data, data)
        elif topic == "LapCount":
            self.lap_count = deep_merge(self.lap_count, data)
        elif topic == "WeatherData":
            self.weather_data = deep_merge(self.weather_data, data)
        elif topic == "TrackStatus":
            self.track_status = deep_merge(self.track_status, data)
        elif topic in {"CarData", "CarData.z"}:
            self.car_data = deep_merge(self.car_data, data)
        elif topic in {"Position", "Position.z"}:
            self.position = deep_merge(self.position, data)
        elif topic == "RaceControlMessages":
            if isinstance(data, dict) and "Messages" in data:
                existing = list(self.race_control.get("Messages") or [])
                patch = data["Messages"]
                if isinstance(patch, list):
                    existing.extend(patch)
                elif isinstance(patch, dict):
                    for _, msg in sorted(
                        patch.items(),
                        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
                    ):
                        if isinstance(msg, dict):
                            existing.append(msg)
                self.race_control = {"Messages": existing[-100:]}

        now = time.monotonic()
        if now - self._last_persist_at >= self.persist_interval_s:
            self.save_atomic()
            self._last_persist_at = now

    def snapshot(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "last_update": self.last_update,
            "session": self.session_info,
            "drivers": self.driver_list,
            "timing": self.timing_data,
            "timing_app": self.timing_app_data,
            "lap_count": self.lap_count,
            "weather": self.weather_data,
            "track_status": self.track_status,
            "race_control": self.race_control,
        }

    def save_atomic(self) -> None:
        """Atomically persist current state to ``backend/cache/last_state.json``."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(f"{self.state_path.name}.{os.getpid()}.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self.snapshot(), handle, separators=(",", ":"))
            os.replace(tmp, self.state_path)
        except OSError as exc:
            LOGGER.warning("Failed atomic state persistence to %s: %s", self.state_path, exc)

    def restore(self) -> bool:
        """Restore previously persisted crash-recovery state if available."""
        if not self.state_path.exists():
            return False
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            if not isinstance(saved, dict):
                return False
            self.session_info = saved.get("session") or {}
            self.driver_list = saved.get("drivers") or {}
            self.timing_data = saved.get("timing") or {}
            self.timing_app_data = saved.get("timing_app") or {}
            self.lap_count = saved.get("lap_count") or {}
            self.weather_data = saved.get("weather") or {}
            self.track_status = saved.get("track_status") or {}
            self.race_control = saved.get("race_control") or {"Messages": []}
            self.last_update = saved.get("last_update")
            LOGGER.info("Restored SignalR crash-recovery state from %s", self.state_path)
            return True
        except (OSError, ValueError) as exc:
            LOGGER.warning("Could not restore %s: %s", self.state_path, exc)
            return False


# --------------------------------------------------------------------------- #
# SignalR Core Live Stream Provider
# --------------------------------------------------------------------------- #
class SignalRLiveProvider(BaseStreamProvider):
    """Unauthenticated F1 SignalR Core live timing provider emitting ``live_race_update`` packets."""

    FREE_TOPICS: Tuple[str, ...] = (
        "TimingData",
        "TimingAppData",
        "LapCount",
        "WeatherData",
        "RaceControlMessages",
        "DriverList",
        "TrackStatus",
        "SessionInfo",
        "Heartbeat",
    )

    def __init__(
        self,
        *,
        predictor: Optional[RacePredictor] = None,
        state_path: Path = DEFAULT_LAST_STATE_PATH,
        live_token: Optional[str] = None,
        negotiate_url: str = config.SIGNALR_NEGOTIATE_URL,
        ws_base_url: str = config.SIGNALR_URL,
        emit_interval_s: float = 1.0,
    ) -> None:
        super().__init__(mode="free_live")
        self.predictor = predictor
        self.live_token = (live_token if live_token is not None else config.F1_LIVE_TOKEN).strip()
        self.negotiate_url = negotiate_url.rstrip("/")
        self.ws_base_url = ws_base_url.rstrip("/")
        self.emit_interval_s = max(0.25, float(emit_interval_s))

        self.store = SignalRStateStore(state_path=state_path)
        self._tyre_deg = TyreDegTracker()
        self._driver_last_lap_seen: Dict[int, int] = {}
        self._driver_pit_counts: Dict[int, int] = {}
        self._driver_pit_history: Dict[int, List[Dict[str, Any]]] = {}

        self._task: Optional[asyncio.Task[None]] = None
        self._emit_task: Optional[asyncio.Task[None]] = None
        self._last_error: Optional[str] = None
        self._connection_phase: str = "idle"
        self._reconnect_delay_s: float = 2.0

    @property
    def telemetry_unlocked(self) -> bool:
        return bool(self.live_token)

    def subscribed_topics(self) -> List[str]:
        topics = list(self.FREE_TOPICS)
        if self.telemetry_unlocked:
            topics.extend(config.LIVE_TOKEN_TOPICS)
        return topics

    # ------------------------------------------------------------------ #
    # SignalR Core protocol handling
    # ------------------------------------------------------------------ #
    async def _negotiate(self) -> Tuple[str, str]:
        """POST /signalrcore/negotiate?negotiateVersion=1 -> (connection_token, cookie_header)."""
        url = f"{self.negotiate_url}/negotiate?negotiateVersion=1"
        headers = {**BASE_HEADERS, "Content-Type": "application/json"}
        if self.live_token:
            headers["Cookie"] = self.live_token

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json={}, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            cookies = "; ".join(f"{k}={v}" for k, v in resp.cookies.items())
            if self.live_token:
                cookies = f"{cookies}; {self.live_token}" if cookies else self.live_token
            token = payload.get("connectionToken") or payload.get("connectionId")
            if not token:
                raise RuntimeError("SignalR negotiate response missing connectionToken/connectionId")
            return str(token), cookies

    def handle_raw_frames(self, raw_message: str) -> List[Dict[str, Any]]:
        """Process one or more ``\\x1e``-delimited SignalR frames and return replies to send."""
        replies: List[Dict[str, Any]] = []
        frames = [f for f in raw_message.split(RECORD_SEP) if f.strip()]
        for frame in frames:
            try:
                msg = json.loads(frame)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            if msg_type == 1:
                # Server invocation: target == "feed", arguments == [topic, data, ...]
                args = msg.get("arguments")
                if msg.get("target") == "feed" and isinstance(args, list) and len(args) >= 2:
                    self._process_feed_topic(str(args[0]), args[1])
            elif msg_type == 3:
                # Completion of initial Subscribe call containing full snapshot dict
                result = msg.get("result")
                if str(msg.get("invocationId")) == "0" and isinstance(result, dict):
                    for topic, payload in result.items():
                        if payload is not None:
                            self._process_feed_topic(str(topic), payload)
            elif msg_type == 6:
                # Keepalive ping -> pong
                replies.append({"type": 6})
            elif msg_type == 7:
                LOGGER.warning("SignalR server sent close frame: %s", msg.get("error"))
        return replies

    def _process_feed_topic(self, topic: str, raw_data: Any) -> None:
        is_compressed = topic.endswith(".z")
        if is_compressed and not self.telemetry_unlocked:
            # Strict free_live gate: ignore compressed GPS/CarData unless F1_LIVE_TOKEN is set
            return
        data = decompress_topic(raw_data) if is_compressed else raw_data
        if data is None:
            return
        clean_topic = topic[:-2] if is_compressed else topic
        self.store.apply_update(clean_topic, data)

    # ------------------------------------------------------------------ #
    # Contract packet normalization
    # ------------------------------------------------------------------ #
    def build_packet(self) -> Dict[str, Any]:
        """Build a contract-valid ``live_race_update`` packet from the merged SignalR state."""
        session = self.store.session_info or {}
        meeting = session.get("Meeting") or {}
        circuit_obj = meeting.get("Circuit") or {}

        lap_current = _int_or_none(self.store.lap_count.get("CurrentLap")) or 0
        lap_total = _int_or_none(self.store.lap_count.get("TotalLaps"))
        if lap_total is not None:
            self._tyre_deg.set_total_laps(lap_total)
        track_status = _map_track_status(self.store.track_status)

        # Weather
        w = self.store.weather_data or {}
        weather: Optional[Dict[str, Any]] = None
        if w:
            rain_raw = w.get("Rainfall")
            rainfall = (
                str(rain_raw).strip() in {"1", "true", "True"}
                if rain_raw is not None
                else None
            )
            weather = {
                "air_temperature_c": _float_or_none(w.get("AirTemp")),
                "track_temperature_c": _float_or_none(w.get("TrackTemp")),
                "humidity_pct": _float_or_none(w.get("Humidity")),
                "pressure_mbar": _float_or_none(w.get("Pressure")),
                "wind_speed_ms": _float_or_none(w.get("WindSpeed")),
                "wind_direction_deg": _float_or_none(w.get("WindDirection")),
                "rainfall": rainfall,
            }

        # Race control messages
        raw_rc = self.store.race_control.get("Messages") or []
        if isinstance(raw_rc, dict):
            raw_rc = list(normalise_arrays(raw_rc))
        rc_messages: List[Dict[str, Any]] = []
        for msg in raw_rc[-config.RACE_CONTROL_TAIL :]:
            if not isinstance(msg, dict) or not msg.get("Message"):
                continue
            utc_str = str(msg.get("Utc") or _now_iso())
            if not utc_str.endswith("Z") and "+" not in utc_str[10:]:
                utc_str += "Z"
            rc_messages.append(
                {
                    "date": utc_str,
                    "lap_number": _int_or_none(msg.get("Lap")),
                    "category": str(msg["Category"]) if msg.get("Category") is not None else None,
                    "flag": str(msg["Flag"]) if msg.get("Flag") is not None else None,
                    "scope": str(msg["Scope"]) if msg.get("Scope") is not None else None,
                    "driver_number": _int_or_none(msg.get("RacingNumber")),
                    "message": str(msg["Message"]),
                }
            )

        # Unlocked GPS / CarData (strictly null when F1_LIVE_TOKEN is unset)
        gps_available = False
        telemetry_available = False
        latest_pos_cars: Dict[str, Any] = {}
        latest_car_channels: Dict[str, Any] = {}
        if self.telemetry_unlocked:
            pos_list = self.store.position.get("Position")
            if isinstance(pos_list, list) and pos_list:
                latest_pos_cars = pos_list[-1].get("Entries") or {}
                gps_available = bool(latest_pos_cars)
            car_entries = self.store.car_data.get("Entries")
            if isinstance(car_entries, list) and car_entries:
                latest_car_channels = car_entries[-1].get("Cars") or {}
                telemetry_available = bool(latest_car_channels)

        # Per-driver rows
        driver_list = self.store.driver_list or {}
        timing_lines = (self.store.timing_data or {}).get("Lines") or {}
        app_lines = (self.store.timing_app_data or {}).get("Lines") or {}

        all_nums = sorted(
            {str(k) for k in driver_list.keys() if re.fullmatch(r"\d+", str(k))}
            | {str(k) for k in timing_lines.keys() if re.fullmatch(r"\d+", str(k))},
            key=lambda s: int(s),
        )

        drivers_out: List[Dict[str, Any]] = []
        for num_str in all_nums:
            dn = int(num_str)
            d_info = driver_list.get(num_str) or {}
            td = timing_lines.get(num_str) or {}
            app = app_lines.get(num_str) or {}

            position = _int_or_none(td.get("Position"))
            retired = bool(td.get("Retired")) or bool(td.get("KnockedOut"))
            in_pit = bool(td.get("InPit"))
            pit_out = bool(td.get("PitOut"))

            gap_raw = td.get("GapToLeader")
            gap_s = 0.0 if position == 1 else _float_or_none(gap_raw)
            laps_down = 0
            if isinstance(gap_raw, str) and "L" in gap_raw.upper():
                m = re.search(r"(\d+)\s*L", gap_raw.upper())
                if m:
                    laps_down = int(m.group(1))

            int_obj = td.get("IntervalToPositionAhead")
            int_val = int_obj.get("Value") if isinstance(int_obj, dict) else int_obj
            interval_s = 0.0 if position == 1 else _float_or_none(int_val)

            last_lap_obj = td.get("LastLapTime")
            last_lap_s = parse_lap_time(
                last_lap_obj.get("Value") if isinstance(last_lap_obj, dict) else last_lap_obj
            )
            best_lap_obj = td.get("BestLapTime")
            best_lap_s = parse_lap_time(
                best_lap_obj.get("Value") if isinstance(best_lap_obj, dict) else best_lap_obj
            )
            num_laps = _int_or_none(td.get("NumberOfLaps")) or lap_current

            # Track pit stops from NumberOfPitStops counter
            pit_total = _int_or_none(td.get("NumberOfPitStops")) or 0
            prev_pits = self._driver_pit_counts.get(dn, 0)
            if pit_total > prev_pits:
                self._driver_pit_counts[dn] = pit_total
                hist = self._driver_pit_history.setdefault(dn, [])
                hist.append({"lap_number": num_laps, "pit_duration_s": None, "date": _now_iso()})
            pit_stops = list(self._driver_pit_history.get(dn, []))

            # Stints from TimingAppData
            raw_stints = app.get("Stints") or {}
            stint_list = normalise_arrays(raw_stints) if isinstance(raw_stints, dict) else raw_stints
            if not isinstance(stint_list, list):
                stint_list = []

            stints_out: List[Dict[str, Any]] = []
            for idx, st in enumerate(stint_list, start=1):
                if not isinstance(st, dict):
                    continue
                stints_out.append(
                    {
                        "stint_number": idx,
                        "compound": _normalize_compound(st.get("Compound")),
                        "lap_start": _int_or_none(st.get("StartLaps")) or 1,
                        "lap_end": None,
                        "tyre_age_at_start": _int_or_none(st.get("StartLaps")) or 0,
                    }
                )

            latest_stint = stint_list[-1] if stint_list and isinstance(stint_list[-1], dict) else {}
            compound = _normalize_compound(latest_stint.get("Compound"))
            tyre_age = _int_or_none(latest_stint.get("TotalLaps"))
            stint_no = len(stints_out) if stints_out else 1
            is_new_raw = latest_stint.get("New")
            is_new = (
                str(is_new_raw).lower() == "true" if isinstance(is_new_raw, str) else (
                    bool(is_new_raw) if is_new_raw is not None else None
                )
            )

            # Update TyreDegTracker whenever driver finishes a new lap
            if num_laps > self._driver_last_lap_seen.get(dn, 0) and last_lap_s is not None:
                self._driver_last_lap_seen[dn] = num_laps
                self._tyre_deg.record_lap(
                    dn,
                    num_laps,
                    last_lap_s,
                    stint_number=stint_no,
                    is_pit_out_lap=pit_out,
                    is_pit_in_lap=in_pit,
                    track_status=track_status,
                )

            deg_res = self._tyre_deg.get_driver_deg(dn)
            tyre_health = deg_res.tyre_health if deg_res is not None else "UNKNOWN"

            # GPS & telemetry strictly null unless F1_LIVE_TOKEN unlocked
            x_val: Optional[float] = None
            y_val: Optional[float] = None
            telemetry_obj: Optional[Dict[str, Any]] = None
            if self.telemetry_unlocked:
                p_entry = latest_pos_cars.get(num_str) or {}
                if p_entry:
                    x_val = _float_or_none(p_entry.get("X"))
                    y_val = _float_or_none(p_entry.get("Y"))
                c_entry = (latest_car_channels.get(num_str) or {}).get("Channels") or {}
                if c_entry:
                    drs_int = _int_or_none(c_entry.get("45"))
                    telemetry_obj = {
                        "date": _now_iso(),
                        "rpm": _int_or_none(c_entry.get("0")),
                        "speed_kph": _float_or_none(c_entry.get("2")),
                        "gear": _int_or_none(c_entry.get("3")),
                        "throttle_pct": _float_or_none(c_entry.get("4")),
                        "brake_pct": _float_or_none(c_entry.get("5")),
                        "drs": drs_int,
                        "drs_active": (drs_int >= 10) if drs_int is not None else None,
                        "x": x_val,
                        "y": y_val,
                        "z": _float_or_none(p_entry.get("Z")),
                    }

            full_name = str(
                d_info.get("FullName")
                or f"{d_info.get('FirstName', '')} {d_info.get('LastName', '')}".strip()
                or d_info.get("Tla")
                or f"Driver {dn}"
            )
            drivers_out.append(
                {
                    "driver_number": dn,
                    "code": str(d_info.get("Tla") or dn),
                    "full_name": full_name,
                    "team_name": str(d_info.get("TeamName") or "Unknown Team"),
                    "team_colour": _normalize_hex(d_info.get("TeamColour")),
                    "position": None if retired else position,
                    "grid_position": _int_or_none(app.get("GridPos")),
                    "gap_to_leader_s": None if retired else gap_s,
                    "interval_s": None if retired else interval_s,
                    "laps_down": int(laps_down),
                    "last_lap_time_s": last_lap_s,
                    "best_lap_time_s": best_lap_s,
                    "tyre": {
                        "compound": compound,
                        "age_laps": tyre_age,
                        "stint_number": stint_no,
                        "is_new": is_new,
                        "health": tyre_health,
                    },
                    "tyre_health": tyre_health,
                    "stints": stints_out,
                    "pit_stops": pit_stops,
                    "in_pit": in_pit,
                    "retired": retired,
                    "status": "DNF" if retired else ("PIT" if in_pit else "RUNNING"),
                    "x": x_val,
                    "y": y_val,
                    "tyre_deg": deg_res.to_contract() if deg_res is not None else None,
                    "podium_prob_pct": 0.0,
                    "win_prob_pct": 0.0,
                    "win_probability_pct": 0.0,
                    "telemetry": telemetry_obj,
                }
            )

        drivers_out.sort(
            key=lambda d: (
                1 if d["retired"] else 0,
                d["position"] if d["position"] is not None else 999,
                d["driver_number"],
            )
        )

        # Compute Top-3 Podium predictions across active drivers
        predictions: Optional[Dict[str, Any]] = None
        if drivers_out:
            predictions = self._compute_live_predictions(drivers_out, lap_current, lap_total, track_status)
            p_map = {int(i["driver_number"]): i["podium_prob_pct"] for i in predictions["items"]}
            w_map = {int(i["driver_number"]): i["win_prob_pct"] for i in predictions["items"]}
            for d in drivers_out:
                dn = int(d["driver_number"])
                out = is_driver_out(d)
                d["podium_prob_pct"] = 0.0 if out else float(p_map.get(dn, 0.0))
                d["win_prob_pct"] = 0.0 if out else float(w_map.get(dn, 0.0))
                d["win_probability_pct"] = d["win_prob_pct"]

        season_year = None
        start_date = str(session.get("StartDate") or "")
        if len(start_date) >= 4 and start_date[:4].isdigit():
            season_year = int(start_date[:4])

        race_state = {
            "lap_current": lap_current,
            "lap_total": lap_total,
            "track_status": track_status,
            "weather": weather,
            "drivers": drivers_out,
            "race_control": rc_messages,
            "predictions": predictions,
        }

        return {
            "schema_version": config.SCHEMA_VERSION,
            "type": "live_race_update",
            "timestamp": _now_iso(),
            "sequence": self.next_sequence(),
            "source": "signalr_live",
            "mode": "free_live",
            "telemetry_unlocked": self.telemetry_unlocked,
            "capabilities": {
                "gps_available": gps_available,
                "telemetry_available": telemetry_available,
            },
            "meeting_key": _int_or_none(meeting.get("Key")),
            "session_key": _int_or_none(session.get("Key")),
            "season": season_year,
            "round": _int_or_none(meeting.get("Number")),
            "grand_prix": str(meeting.get("Name") or session.get("Name") or "Live F1 Session"),
            "circuit": {
                "circuit_id": str(circuit_obj["Key"]) if circuit_obj.get("Key") is not None else None,
                "name": str(circuit_obj.get("ShortName") or meeting.get("Location") or "Live Circuit"),
                "rotation_deg": None,
                "track_outline": None,
            },
            "race": race_state,
            "tick": race_state,
        }

    def _compute_live_predictions(
        self,
        drivers: List[Dict[str, Any]],
        lap_current: int,
        lap_total: Optional[int],
        track_status: str,
    ) -> Dict[str, Any]:
        active_mask = [not is_driver_out(d) for d in drivers]
        if not any(active_mask):
            return {
                "stage": "in_race",
                "target": "top3_podium",
                "model_version": "heuristic-top3-live-v1",
                "generated_at": _now_iso(),
                "lap": lap_current,
                "items": [
                    {
                        "rank": i,
                        "driver_number": int(d["driver_number"]),
                        "driver_code": str(d["code"]),
                        "podium_prob_pct": 0.0,
                        "win_prob_pct": 0.0,
                        "win_probability_pct": 0.0,
                    }
                    for i, d in enumerate(drivers, start=1)
                ],
                "sum_podium_pct": 300.0,
                "sum_win_pct": 100.0,
                "sum_pct": 300.0,
            }

        # Use trained ML model if loaded
        if self.predictor is not None and self.predictor.in_race_available:
            try:
                import pandas as pd

                rows = [
                    {
                        "season": config.current_season(),
                        "round": 1,
                        "driver_number": int(d["driver_number"]),
                        "driver_code": str(d["code"]),
                        "grid_position": d.get("grid_position") or d.get("position") or 10,
                        "driver_win_rate_10": 0.05,
                        "driver_podium_rate_10": 0.15,
                        "driver_avg_finish_5": float(d.get("position") or 10),
                        "team_points_avg_5": 10.0,
                        "team_win_rate_10": 0.05,
                    }
                    for d in drivers
                ]
                ctx = RaceContext(
                    season=config.current_season(),
                    round_number=1,
                    race_name="Live Session",
                    circuit_id=None,
                    circuit_name=None,
                    race_date=None,
                    grid_source="openf1_starting_grid",
                    features=pd.DataFrame(rows),
                )
                return self.predictor.predict_live(
                    ctx,
                    drivers,
                    lap=max(1, lap_current),
                    total_laps=lap_total or 57,
                    track_state=track_status,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Live SignalR ML fallback: %s", exc)

        total = float(lap_total or 57)
        lap_frac = min(1.0, max(0.05, float(lap_current) / max(1.0, total)))
        sharpness = 0.42 + 0.55 * lap_frac
        raw_podium: List[float] = []
        raw_win: List[float] = []
        for d, active in zip(drivers, active_mask):
            if not active:
                raw_podium.append(0.0)
                raw_win.append(0.0)
                continue
            pos = float(d.get("position") or 20)
            gap = max(0.0, float(d.get("gap_to_leader_s") or 0.0)) + float(d.get("laps_down") or 0) * 90.0
            raw_podium.append(max(1e-6, math.exp(-sharpness * (pos - 1.0)) * math.exp(-gap / 45.0)))
            raw_win.append(max(1e-6, math.exp(-(sharpness * 1.45) * (pos - 1.0)) * math.exp(-gap / 25.0)))

        podium_pcts = normalize_podium_to_percent(raw_podium, active_mask=active_mask, decimals=2, target_sum=300.0)
        active_win = normalize_to_percent([s for s, a in zip(raw_win, active_mask) if a], decimals=2)
        w_it = iter(active_win)
        win_pcts = [next(w_it) if a else 0.0 for a in active_mask]

        items: List[Dict[str, Any]] = []
        for d, p_pct, w_pct in zip(drivers, podium_pcts, win_pcts):
            item: Dict[str, Any] = {
                "driver_number": int(d["driver_number"]),
                "driver_code": str(d["code"]),
                "full_name": str(d["full_name"]),
                "team_name": str(d["team_name"]),
                "podium_prob_pct": float(p_pct),
                "win_prob_pct": float(w_pct),
                "win_probability_pct": float(w_pct),
            }
            if d.get("team_colour"):
                item["team_colour"] = d["team_colour"]
            items.append(item)

        items.sort(key=lambda i: (-i["podium_prob_pct"], -i["win_prob_pct"], i["driver_number"]))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank

        sum_p = round(float(sum(i["podium_prob_pct"] for i in items)), 2)
        sum_w = round(float(sum(i["win_prob_pct"] for i in items)), 2)
        return {
            "stage": "in_race",
            "target": "top3_podium",
            "model_version": "heuristic-top3-live-v1",
            "generated_at": _now_iso(),
            "lap": lap_current,
            "items": items,
            "sum_podium_pct": sum_p,
            "sum_win_pct": sum_w,
            "sum_pct": sum_p,
        }

    # ------------------------------------------------------------------ #
    # Async connection & emitter lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Emit restored state immediately so clients connecting see last_state right away
        await self.publish(self.build_packet())
        self._emit_task = asyncio.create_task(self._emit_loop(), name="f1-signalr-emit")
        if config.ALLOW_LIVE_NETWORK:
            self._task = asyncio.create_task(self._connect_loop(), name="f1-signalr-client")

    async def stop(self) -> None:
        self._running = False
        self.store.connected = False
        self.store.save_atomic()
        for task in (self._task, self._emit_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._emit_task = None

    async def _emit_loop(self) -> None:
        try:
            while self._running:
                await self.publish(self.build_packet())
                await asyncio.sleep(self.emit_interval_s)
        except asyncio.CancelledError:
            raise

    async def _connect_loop(self) -> None:
        while self._running:
            try:
                self._connection_phase = "negotiating"
                token, cookies = await self._negotiate()
                ws_url = f"{self.ws_base_url}?id={quote(token, safe='')}"
                extra_headers = {
                    "User-Agent": "BestHTTP",
                    "Origin": "https://www.formula1.com",
                }
                if cookies:
                    extra_headers["Cookie"] = cookies

                self._connection_phase = "connecting"
                async with websockets.connect(
                    ws_url,
                    additional_headers=extra_headers,
                    ping_interval=None,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    self._reconnect_delay_s = 2.0
                    self._connection_phase = "handshaking"
                    await ws.send(json.dumps({"protocol": "json", "version": 1}) + RECORD_SEP)

                    handshake_raw = await ws.recv()
                    if isinstance(handshake_raw, bytes):
                        handshake_raw = handshake_raw.decode("utf-8", errors="ignore")
                    hs_frames = [f for f in handshake_raw.split(RECORD_SEP) if f.strip()]
                    if hs_frames:
                        hs = json.loads(hs_frames[0])
                        if hs.get("error"):
                            raise RuntimeError(f"SignalR handshake error: {hs['error']}")

                    # Subscribe to topics
                    await ws.send(
                        json.dumps(
                            {
                                "type": 1,
                                "invocationId": "0",
                                "target": "Subscribe",
                                "arguments": [self.subscribed_topics()],
                            }
                        )
                        + RECORD_SEP
                    )
                    self.store.connected = True
                    self._connection_phase = "connected"
                    self._last_error = None

                    if len(hs_frames) > 1:
                        self.handle_raw_frames(RECORD_SEP.join(hs_frames[1:]) + RECORD_SEP)

                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            if not self._running:
                                break
                            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
                            replies = self.handle_raw_frames(text)
                            for reply in replies:
                                await ws.send(json.dumps(reply) + RECORD_SEP)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.store.connected = False
                self._connection_phase = "reconnecting"
                self._last_error = str(exc)
                LOGGER.warning(
                    "SignalR connection error (%s); retrying in %.1fs",
                    exc,
                    self._reconnect_delay_s,
                )
                await asyncio.sleep(self._reconnect_delay_s)
                self._reconnect_delay_s = min(self._reconnect_delay_s * 2.0, 30.0)

    async def _ping_loop(self, ws: Any) -> None:
        try:
            while self._running:
                await asyncio.sleep(15.0)
                await ws.send(json.dumps({"type": 6}) + RECORD_SEP)
        except asyncio.CancelledError:
            raise

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "running": self._running,
            "connected": self.store.connected,
            "phase": self._connection_phase,
            "telemetry_unlocked": self.telemetry_unlocked,
            "gps_available": self.telemetry_unlocked and bool(self.store.position),
            "telemetry_available": self.telemetry_unlocked and bool(self.store.car_data),
            "last_update": self.store.last_update,
            "last_error": self._last_error,
            "topics": self.subscribed_topics(),
        }
