"""OpenF1 2023-2026 Replay Engine (250 ms / 4 Hz resampling, rotation, TyreDeg & Top-3 ML).

Features:
- Loads and caches any completed 2023-2026 OpenF1 race session.
- Filters out cancelled 2026 rounds (``is_cancelled == false`` via ``OpenF1Client``).
- Downloads ``location`` and ``car_data`` in 10-minute chunks (600 s) with
  local parquet/JSON caching.
- Treats ``car_data.drs`` as nullable (``None`` in 2026 sessions) and maps
  valid legacy DRS codes (>= 10) to ``drs_active: bool | None``.
- Rotates ``(x, y)`` circuit coordinates and the derived ``track_outline`` using
  ``meetings[0]["circuit_info_url"]`` rotation angle.
- Resamples the session timeline onto 250 ms (4 Hz) ticks.
- Integrates :class:`analytics.tyre_deg.TyreDegTracker` and Top-3
  :class:`ml.predictor.RacePredictor` (with graceful heuristic fallback if
  joblib models haven't been trained on real data yet, still obeying 300%
  podium sum and 100% win sum).
- Supports ``play``, ``pause``, ``set_speed`` (``1x`` to ``10x``), and ``seek_lap``.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from analytics.radio_transcriber import RadioTranscriber
from analytics.sectors_fastest import SectorAndLapTracker
from analytics.tyre_deg import TyreDegTracker
from clients.openf1_client import OpenF1Client, parse_iso_datetime
from ml.features import (
    TRACK_STATE_NAMES,
    normalize_podium_to_percent,
    normalize_to_percent,
    to_epoch_seconds,
    track_state_at,
    track_state_timeline,
)
from ml.predictor import RaceContext, RacePredictor, is_driver_out
from server.circuits import resolve_circuit_2d
from streams.base_provider import BaseStreamProvider

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ReplayEngine",
    "rotate_xy",
    "drs_to_active",
    "build_track_outline",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _epoch_to_iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(float(epoch_s), tz=timezone.utc).isoformat(timespec="milliseconds")


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
        return None if not np.isfinite(num) else num
    except (TypeError, ValueError):
        return None


def rotate_xy(
    x: Optional[float],
    y: Optional[float],
    rotation_deg: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Rotate 2D track coordinates counter-clockwise by ``rotation_deg`` degrees."""
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return None, None
    if rotation_deg is None or not np.isfinite(rotation_deg) or rotation_deg == 0.0:
        return float(x), float(y)
    rad = math.radians(float(rotation_deg))
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rx = float(x) * cos_a - float(y) * sin_a
    ry = float(x) * sin_a + float(y) * cos_a
    return round(rx, 2), round(ry, 2)


def drs_to_active(drs: Optional[int]) -> Optional[bool]:
    """Convert nullable OpenF1 DRS integer code to boolean (``None`` in 2026)."""
    if drs is None:
        return None
    try:
        code = int(drs)
    except (TypeError, ValueError):
        return None
    # OpenF1 / F1 standard: 10, 12, 14 denote DRS open/active; 0-8 denote closed/eligible
    return code >= 10


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


def build_track_outline(
    locations_by_driver: Mapping[int, Dict[str, np.ndarray]],
    laps_by_driver: Mapping[int, List[Dict[str, Any]]],
    rotation_deg: Optional[float],
    max_points: int = 320,
) -> Optional[List[Dict[str, float]]]:
    """Extract a clean single-lap closed track outline rotated by ``rotation_deg``."""
    best_xs: Optional[np.ndarray] = None
    best_ys: Optional[np.ndarray] = None

    for dn, loc in locations_by_driver.items():
        t_arr = loc.get("t")
        x_arr = loc.get("x")
        y_arr = loc.get("y")
        if t_arr is None or len(t_arr) < 40:
            continue
        # Prefer a clean racing lap (lap 3..10)
        driver_laps = laps_by_driver.get(dn, [])
        chosen_slice: Optional[Tuple[float, float]] = None
        for lap in driver_laps:
            lno = lap.get("lap_number") or 0
            start_s = lap.get("start_s")
            end_s = lap.get("end_s")
            if 3 <= lno <= 12 and start_s and end_s and end_s > start_s and not lap.get("is_pit_out_lap"):
                chosen_slice = (float(start_s), float(end_s))
                break
        if chosen_slice is not None:
            mask = (t_arr >= chosen_slice[0]) & (t_arr <= chosen_slice[1])
            xs, ys = x_arr[mask], y_arr[mask]
            if len(xs) >= 40:
                best_xs, best_ys = xs, ys
                break
        if best_xs is None:
            # Fallback: take first ~120s of movement
            t0 = t_arr[0]
            mask = (t_arr >= t0 + 30.0) & (t_arr <= t0 + 150.0)
            xs, ys = x_arr[mask], y_arr[mask]
            if len(xs) >= 40:
                best_xs, best_ys = xs, ys

    if best_xs is None or best_ys is None or len(best_xs) < 10:
        return None

    # Filter out (0, 0) telemetry glitches
    valid = (np.abs(best_xs) + np.abs(best_ys)) > 1.0
    xs, ys = best_xs[valid], best_ys[valid]
    if len(xs) < 10:
        return None

    step = max(1, len(xs) // max_points)
    points: List[Dict[str, float]] = []
    for x_val, y_val in zip(xs[::step], ys[::step]):
        rx, ry = rotate_xy(float(x_val), float(y_val), rotation_deg)
        if rx is not None and ry is not None:
            points.append({"x": rx, "y": ry})
    return points if len(points) >= 10 else None


@dataclass
class _SessionMeta:
    meeting_key: int
    session_key: int
    season: int
    round_number: Optional[int]
    grand_prix: str
    circuit_id: Optional[str]
    circuit_name: str
    rotation_deg: Optional[float]
    start_s: float
    end_s: float
    total_laps: int


class ReplayEngine(BaseStreamProvider):
    """4 Hz (250 ms) OpenF1 race replay engine with play/pause/speed/seek controls."""

    TICK_INTERVAL_S: float = 0.25  # 250 ms (4 Hz)

    def __init__(
        self,
        *,
        openf1: Optional[OpenF1Client] = None,
        predictor: Optional[RacePredictor] = None,
        transcriber: Optional[RadioTranscriber] = None,
        tick_interval_s: float = 0.25,
    ) -> None:
        super().__init__(mode="replay")
        self.openf1 = openf1 or OpenF1Client()
        self.predictor = predictor
        self.transcriber = transcriber
        self.tick_interval_s = max(0.05, float(tick_interval_s))

        self._meta: Optional[_SessionMeta] = None
        self._track_outline: Optional[List[Dict[str, float]]] = None
        self._circuit_slug: Optional[str] = None
        self._circuit_corners: Optional[List[Dict[str, Any]]] = None
        self._sector_splits: Optional[List[int]] = None
        self._start_finish: Optional[Dict[str, float]] = None
        self._drivers_meta: Dict[int, Dict[str, Any]] = {}
        self._grid_by_driver: Dict[int, int] = {}
        self._laps_by_driver: Dict[int, List[Dict[str, Any]]] = {}
        self._stints_by_driver: Dict[int, List[Dict[str, Any]]] = {}
        self._pits_by_driver: Dict[int, List[Dict[str, Any]]] = {}
        self._pos_by_driver: Dict[int, Dict[str, np.ndarray]] = {}
        self._int_by_driver: Dict[int, Dict[str, np.ndarray]] = {}
        self._loc_by_driver: Dict[int, Dict[str, np.ndarray]] = {}
        self._car_by_driver: Dict[int, Dict[str, np.ndarray]] = {}
        self._weather_times: np.ndarray = np.array([], dtype=float)
        self._weather_rows: List[Dict[str, Any]] = []
        self._rc_times: np.ndarray = np.array([], dtype=float)
        self._rc_rows: List[Dict[str, Any]] = []
        self._radio_times: np.ndarray = np.array([], dtype=float)
        self._radio_rows: List[Dict[str, Any]] = []
        self._track_timeline: List[Tuple[float, int]] = []
        self._lap_start_times: Dict[int, float] = {}
        self._retired_after: Dict[int, float] = {}
        self._race_context: Optional[RaceContext] = None

        self._tyre_deg = TyreDegTracker()
        self._sector_tracker = SectorAndLapTracker()
        self._cached_pred_lap: int = -1
        self._cached_predictions: Optional[Dict[str, Any]] = None

        self._cursor_s: float = 0.0
        self._speed: float = float(config.REPLAY_DEFAULT_SPEED)
        self._paused: bool = True
        self._finished: bool = False
        self._task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Session loading
    # ------------------------------------------------------------------ #
    def list_available_sessions(self, year: int = 2026) -> List[Dict[str, Any]]:
        """List non-cancelled OpenF1 Race sessions for ``year`` (2023-2026).

        Sequential ``round`` numbering is computed *after* skipping cancelled
        rounds so OpenF1 round indices align 1-to-1 with Jolpica's calendar
        (e.g. 2026 Miami = Round 4, Monaco = Round 6, Las Vegas = Round 21).
        """
        races = self.openf1.list_races(year)
        meeting_by_key = {int(m["meeting_key"]): m for m in races if m.get("meeting_key") is not None}
        sessions = self.openf1.get_sessions(year=year, session_name="Race")
        now_iso = _now_iso()
        out: List[Dict[str, Any]] = []
        for session in sessions:
            mk = _int_or_none(session.get("meeting_key"))
            if mk is None or mk not in meeting_by_key:
                continue
            meeting = meeting_by_key[mk]
            round_num = len(out) + 1
            gp_name = str(meeting.get("meeting_name") or session.get("location") or f"Round {round_num}")
            circuit_name = str(session.get("circuit_short_name") or meeting.get("circuit_short_name") or "")
            date_end = session.get("date_end")
            completed = bool(date_end and str(date_end) <= now_iso)
            c2d = resolve_circuit_2d(
                circuit_key=session.get("circuit_key") or meeting.get("circuit_key"),
                name=circuit_name,
                grand_prix=gp_name,
            )
            out.append(
                {
                    "session_key": _int_or_none(session.get("session_key")),
                    "meeting_key": mk,
                    "season": int(year),
                    "round": round_num,
                    "grand_prix": gp_name,
                    "circuit_name": circuit_name,
                    "circuit_slug": c2d.get("slug") if c2d else None,
                    "date_start": session.get("date_start"),
                    "date_end": date_end,
                    "completed": completed,
                    "is_cancelled": bool(meeting.get("is_cancelled", False)),
                }
            )
        return out

    async def load_session(
        self,
        session_key: int,
        *,
        include_telemetry: bool = True,
        telemetry_drivers: Optional[Sequence[int]] = None,
        telemetry_window_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Download (or read from cache) and index an OpenF1 race session for 4 Hz replay."""
        return await asyncio.to_thread(
            self._load_session_sync,
            int(session_key),
            include_telemetry,
            telemetry_drivers,
            telemetry_window_s,
        )

    def _load_session_sync(
        self,
        session_key: int,
        include_telemetry: bool = True,
        telemetry_drivers: Optional[Sequence[int]] = None,
        telemetry_window_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        sessions = self.openf1.get_sessions(session_key=session_key)
        if not sessions:
            raise ValueError(f"OpenF1 session {session_key} not found.")
        sess = sessions[0]
        meeting_key = _int_or_none(sess.get("meeting_key")) or 0
        season = _int_or_none(sess.get("year")) or config.current_season()

        meetings = self.openf1.get_meetings(meeting_key=meeting_key, include_cancelled=False)
        if not meetings:
            # Verify whether the meeting was cancelled in 2026
            all_meetings = self.openf1.get_meetings(meeting_key=meeting_key, include_cancelled=True)
            if all_meetings and bool(all_meetings[0].get("is_cancelled")):
                raise ValueError(f"Meeting {meeting_key} is cancelled and filtered out.")
            meeting = all_meetings[0] if all_meetings else {}
        else:
            meeting = meetings[0]

        rotation_deg = self.openf1.get_circuit_rotation(meeting_key) if meeting_key else None
        grand_prix = str(
            meeting.get("meeting_name")
            or sess.get("meeting_name")
            or sess.get("location")
            or f"Session {session_key}"
        )
        circuit_name = str(
            sess.get("circuit_short_name")
            or meeting.get("circuit_short_name")
            or sess.get("location")
            or "Unknown Circuit"
        )
        circuit_id = str(sess.get("circuit_key")) if sess.get("circuit_key") is not None else None

        # Determine round number among non-cancelled season meetings
        round_number: Optional[int] = None
        try:
            season_races = self.openf1.list_races(season)
            for idx, m in enumerate(season_races, start=1):
                if _int_or_none(m.get("meeting_key")) == meeting_key:
                    round_number = idx
                    break
        except Exception:  # noqa: BLE001
            round_number = None

        drivers_raw = self.openf1.get_drivers(session_key)
        laps_raw = self.openf1.get_laps(session_key)
        pos_raw = self.openf1.get_position(session_key)
        int_raw = self.openf1.get_intervals(session_key)
        stints_raw = self.openf1.get_stints(session_key)
        pits_raw = self.openf1.get_pit(session_key)
        weather_raw = self.openf1.get_weather(session_key)
        rc_raw = self.openf1.get_race_control(session_key)
        try:
            team_radio_raw = self.openf1.get("team_radio", {"session_key": session_key})
        except Exception:  # noqa: BLE001
            team_radio_raw = []
        result_raw = self.openf1.get_session_result(session_key)
        grid_raw = self.openf1.get_starting_grid(session_key)

        date_start = sess.get("date_start")
        date_end = sess.get("date_end")
        start_s = parse_iso_datetime(date_start).timestamp() if date_start else 0.0
        end_s = parse_iso_datetime(date_end).timestamp() if date_end else start_s + 7200.0

        loc_raw: List[Dict[str, Any]] = []
        car_raw: List[Dict[str, Any]] = []
        if include_telemetry and date_start and date_end:
            telem_end_dt = parse_iso_datetime(date_end)
            if telemetry_window_s is not None and telemetry_window_s > 0:
                telem_start_dt = parse_iso_datetime(date_start)
                capped = telem_start_dt.timestamp() + float(telemetry_window_s)
                if capped < telem_end_dt.timestamp():
                    telem_end_dt = datetime.fromtimestamp(capped, tz=timezone.utc)

            # Query location & car_data in 10-minute (600s) chunks across session
            # Using session-wide chunking dramatically reduces API calls (~12 calls per race
            # instead of 20*12=240 calls on the free 30 req/min tier) while respecting the 10-min window rule.
            chunk_s = int(config.OPENF1_TELEMETRY_CHUNK_S)
            cursor = parse_iso_datetime(date_start)
            stop = telem_end_dt
            while cursor < stop:
                win_end = datetime.fromtimestamp(min(cursor.timestamp() + chunk_s, stop.timestamp()), tz=timezone.utc)
                base_params: Dict[str, Any] = {
                    "session_key": session_key,
                    "date>=": cursor.isoformat(),
                    "date<": win_end.isoformat(),
                }
                if telemetry_drivers:
                    for dn in telemetry_drivers:
                        p = {**base_params, "driver_number": int(dn)}
                        loc_raw.extend(self.openf1.get("location", p))
                        car_raw.extend(self.openf1.get("car_data", p))
                else:
                    loc_raw.extend(self.openf1.get("location", base_params))
                    car_raw.extend(self.openf1.get("car_data", base_params))
                cursor = win_end

        return self.load_from_raw(
            session_key=session_key,
            meeting_key=meeting_key,
            season=season,
            round_number=round_number,
            grand_prix=grand_prix,
            circuit_id=circuit_id,
            circuit_name=circuit_name,
            rotation_deg=rotation_deg,
            start_s=start_s,
            end_s=end_s,
            drivers=drivers_raw,
            laps=laps_raw,
            positions=pos_raw,
            intervals=int_raw,
            stints=stints_raw,
            pits=pits_raw,
            weather=weather_raw,
            race_control=rc_raw,
            team_radio=team_radio_raw,
            session_result=result_raw,
            starting_grid=grid_raw,
            locations=loc_raw,
            car_data=car_raw,
        )

    def load_from_raw(
        self,
        *,
        session_key: int,
        meeting_key: int,
        season: int,
        round_number: Optional[int],
        grand_prix: str,
        circuit_id: Optional[str],
        circuit_name: str,
        rotation_deg: Optional[float],
        start_s: float,
        end_s: float,
        drivers: Sequence[Mapping[str, Any]],
        laps: Sequence[Mapping[str, Any]],
        positions: Sequence[Mapping[str, Any]],
        intervals: Sequence[Mapping[str, Any]],
        stints: Sequence[Mapping[str, Any]],
        pits: Sequence[Mapping[str, Any]],
        weather: Sequence[Mapping[str, Any]] = (),
        race_control: Sequence[Mapping[str, Any]] = (),
        team_radio: Sequence[Mapping[str, Any]] = (),
        session_result: Sequence[Mapping[str, Any]] = (),
        starting_grid: Sequence[Mapping[str, Any]] = (),
        locations: Sequence[Mapping[str, Any]] = (),
        car_data: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Index pre-fetched or synthetic OpenF1 session tables into fast numpy lookup arrays."""
        self._drivers_meta = {}
        for row in drivers or []:
            dn = _int_or_none(row.get("driver_number"))
            if dn is None:
                continue
            self._drivers_meta[dn] = {
                "driver_number": dn,
                "code": str(row.get("name_acronym") or row.get("broadcast_name") or dn),
                "full_name": str(row.get("full_name") or row.get("broadcast_name") or f"Driver {dn}"),
                "team_name": str(row.get("team_name") or "Unknown Team"),
                "team_colour": _normalize_hex(row.get("team_colour")),
            }

        self._grid_by_driver = {}
        for row in starting_grid or []:
            dn = _int_or_none(row.get("driver_number"))
            pos = _int_or_none(row.get("position"))
            if dn is not None and pos is not None:
                self._grid_by_driver[dn] = pos

        # Index laps & compute lap_start / lap_end times
        self._laps_by_driver = {}
        lap_rows_tmp: Dict[int, List[Dict[str, Any]]] = {}
        max_lap = 0
        for row in laps or []:
            dn = _int_or_none(row.get("driver_number"))
            lno = _int_or_none(row.get("lap_number"))
            if dn is None or lno is None:
                continue
            max_lap = max(max_lap, lno)
            s_ts = parse_iso_datetime(row["date_start"]).timestamp() if row.get("date_start") else None
            dur = _float_or_none(row.get("lap_duration"))
            lap_rows_tmp.setdefault(dn, []).append(
                {
                    "lap_number": lno,
                    "start_s": s_ts,
                    "lap_duration": dur,
                    "s1_s": _float_or_none(row.get("duration_sector_1")),
                    "s2_s": _float_or_none(row.get("duration_sector_2")),
                    "s3_s": _float_or_none(row.get("duration_sector_3")),
                    "is_pit_out_lap": bool(row.get("is_pit_out_lap")),
                }
            )

        self._lap_start_times = {}
        all_lap_times: List[float] = []
        for dn, items in lap_rows_tmp.items():
            items.sort(key=lambda r: r["lap_number"])
            for i, item in enumerate(items):
                next_s = items[i + 1]["start_s"] if i + 1 < len(items) else None
                if item["start_s"] is not None and item["lap_duration"] is not None:
                    item["end_s"] = item["start_s"] + item["lap_duration"]
                else:
                    item["end_s"] = next_s
                if item["start_s"] is not None:
                    all_lap_times.append(item["start_s"])
                    lno = item["lap_number"]
                    if lno not in self._lap_start_times or item["start_s"] < self._lap_start_times[lno]:
                        self._lap_start_times[lno] = item["start_s"]
                if item["end_s"] is not None:
                    all_lap_times.append(item["end_s"])
            self._laps_by_driver[dn] = items

        # Total laps & retirements from session_result
        self._retired_after = {}
        result_max_laps = 0
        for row in session_result or []:
            dn = _int_or_none(row.get("driver_number"))
            nlaps = _int_or_none(row.get("number_of_laps")) or 0
            result_max_laps = max(result_max_laps, nlaps)
            if dn is not None and (row.get("dnf") or row.get("dns") or row.get("dsq")):
                driver_laps = self._laps_by_driver.get(dn, [])
                # Prefer the lap matching session_result.number_of_laps if recorded
                matching_lap = None
                if nlaps > 0:
                    for item in driver_laps:
                        if item["lap_number"] == nlaps:
                            matching_lap = item
                            break
                chosen_lap = matching_lap or (driver_laps[-1] if driver_laps else None)
                if chosen_lap and chosen_lap.get("end_s") is not None:
                    self._retired_after[dn] = float(chosen_lap["end_s"])
                elif chosen_lap and chosen_lap.get("start_s") is not None:
                    self._retired_after[dn] = float(chosen_lap["start_s"]) + 120.0
                else:
                    self._retired_after[dn] = start_s

        total_laps = max(max_lap, result_max_laps, 1)

        # Refine session time bounds to actual race start if lap 1 timestamp exists
        if 1 in self._lap_start_times and self._lap_start_times[1] > start_s:
            start_s = self._lap_start_times[1]
        if all_lap_times:
            end_s = max(end_s, max(all_lap_times))
        if end_s <= start_s:
            end_s = start_s + 3600.0

        # Stints & pit stops
        self._stints_by_driver = {}
        for row in stints or []:
            dn = _int_or_none(row.get("driver_number"))
            if dn is None:
                continue
            self._stints_by_driver.setdefault(dn, []).append(
                {
                    "stint_number": _int_or_none(row.get("stint_number")) or 1,
                    "compound": _normalize_compound(row.get("compound")),
                    "lap_start": _int_or_none(row.get("lap_start")) or 1,
                    "lap_end": _int_or_none(row.get("lap_end")),
                    "tyre_age_at_start": _int_or_none(row.get("tyre_age_at_start")) or 0,
                }
            )
        for items in self._stints_by_driver.values():
            items.sort(key=lambda s: (s["stint_number"], s["lap_start"]))

        self._pits_by_driver = {}
        for row in pits or []:
            dn = _int_or_none(row.get("driver_number"))
            if dn is None:
                continue
            ts = parse_iso_datetime(row["date"]).timestamp() if row.get("date") else None
            self._pits_by_driver.setdefault(dn, []).append(
                {
                    "lap_number": _int_or_none(row.get("lap_number")),
                    "pit_duration_s": _float_or_none(row.get("pit_duration")),
                    "date": _epoch_to_iso(ts) if ts is not None else None,
                    "_t": ts,
                }
            )
        for items in self._pits_by_driver.values():
            items.sort(key=lambda p: (p["_t"] if p["_t"] is not None else 0.0, p["lap_number"] or 0))

        # Index position stream
        self._pos_by_driver = self._index_timeseries(positions, [("position", float)])
        # Index intervals stream
        self._int_by_driver = self._index_timeseries(
            intervals, [("gap_to_leader", float), ("interval", float)]
        )
        # Index location stream
        self._loc_by_driver = self._index_timeseries(locations, [("x", float), ("y", float), ("z", float)])
        # Index car_data stream (nullable drs)
        self._car_by_driver = self._index_timeseries(
            car_data,
            [
                ("speed", float),
                ("throttle", float),
                ("brake", float),
                ("n_gear", float),
                ("rpm", float),
                ("drs", float),
            ],
        )

        # Weather stream
        w_pairs: List[Tuple[float, Dict[str, Any]]] = []
        for row in weather or []:
            if not row.get("date"):
                continue
            ts = parse_iso_datetime(row["date"]).timestamp()
            w_pairs.append(
                (
                    ts,
                    {
                        "air_temperature_c": _float_or_none(row.get("air_temperature")),
                        "track_temperature_c": _float_or_none(row.get("track_temperature")),
                        "humidity_pct": _float_or_none(row.get("humidity")),
                        "pressure_mbar": _float_or_none(row.get("pressure")),
                        "wind_speed_ms": _float_or_none(row.get("wind_speed")),
                        "wind_direction_deg": _float_or_none(row.get("wind_direction")),
                        "rainfall": bool(row.get("rainfall")) if row.get("rainfall") is not None else None,
                    },
                )
            )
        w_pairs.sort(key=lambda p: p[0])
        self._weather_times = np.asarray([p[0] for p in w_pairs], dtype=float)
        self._weather_rows = [p[1] for p in w_pairs]

        # Race control stream
        rc_pairs: List[Tuple[float, Dict[str, Any]]] = []
        for row in race_control or []:
            if not row.get("date") or not row.get("message"):
                continue
            ts = parse_iso_datetime(row["date"]).timestamp()
            rc_pairs.append(
                (
                    ts,
                    {
                        "date": _epoch_to_iso(ts),
                        "lap_number": _int_or_none(row.get("lap_number")),
                        "category": str(row["category"]) if row.get("category") is not None else None,
                        "flag": str(row["flag"]) if row.get("flag") is not None else None,
                        "scope": str(row["scope"]) if row.get("scope") is not None else None,
                        "driver_number": _int_or_none(row.get("driver_number")),
                        "message": str(row["message"]),
                    },
                )
            )
        rc_pairs.sort(key=lambda p: p[0])
        self._rc_times = np.asarray([p[0] for p in rc_pairs], dtype=float)
        self._rc_rows = [p[1] for p in rc_pairs]
        self._track_timeline = track_state_timeline(self._rc_rows)

        # Team radio stream
        radio_pairs: List[Tuple[float, Dict[str, Any]]] = []
        for row in team_radio or []:
            dn_r = _int_or_none(row.get("driver_number"))
            rec_url = str(row.get("recording_url") or "").strip()
            if dn_r is None or not rec_url or not row.get("date"):
                continue
            ts_r = parse_iso_datetime(row["date"]).timestamp()
            dm_r = self._drivers_meta.get(dn_r) or {}
            radio_pairs.append(
                (
                    ts_r,
                    {
                        "date": _epoch_to_iso(ts_r),
                        "driver_number": dn_r,
                        "driver_code": dm_r.get("code"),
                        "team_name": dm_r.get("team_name"),
                        "team_colour": dm_r.get("team_colour"),
                        "recording_url": rec_url,
                    },
                )
            )
        radio_pairs.sort(key=lambda p: p[0])
        self._radio_times = np.asarray([p[0] for p in radio_pairs], dtype=float)
        self._radio_rows = [p[1] for p in radio_pairs]

        # Resolve official 2D circuit geometry from circuits_2d.json (fallback to GPS outline)
        c2d = resolve_circuit_2d(
            circuit_key=circuit_id,
            circuit_id=circuit_id,
            name=circuit_name,
            grand_prix=grand_prix,
        )
        if c2d and c2d.get("points"):
            self._track_outline = [{"x": float(p["x"]), "y": float(p["y"])} for p in c2d["points"]]
            self._circuit_slug = c2d.get("slug")
            self._circuit_corners = [
                {
                    "number": int(c["number"]),
                    "letter": str(c.get("letter") or "") or None,
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                }
                for c in (c2d.get("corners") or [])
            ]
            self._sector_splits = [int(i) for i in (c2d.get("sector_splits") or [])] or None
            sf = c2d.get("start_finish")
            self._start_finish = (
                {"x": float(sf["x"]), "y": float(sf["y"])}
                if isinstance(sf, dict) and "x" in sf and "y" in sf
                else None
            )
            if rotation_deg is None and c2d.get("rotation_deg") is not None:
                rotation_deg = float(c2d["rotation_deg"])
        else:
            self._track_outline = build_track_outline(self._loc_by_driver, self._laps_by_driver, rotation_deg)
            self._circuit_slug = None
            self._circuit_corners = None
            self._sector_splits = None
            self._start_finish = None

        self._meta = _SessionMeta(
            meeting_key=int(meeting_key),
            session_key=int(session_key),
            season=int(season),
            round_number=round_number,
            grand_prix=grand_prix,
            circuit_id=circuit_id,
            circuit_name=circuit_name,
            rotation_deg=rotation_deg,
            start_s=float(start_s),
            end_s=float(end_s),
            total_laps=int(total_laps),
        )

        # Try building ML RaceContext if predictor has pre-race history
        self._race_context = self._try_build_race_context()

        # Reset playback to start
        self._cursor_s = float(start_s)
        self._finished = False
        self._paused = False
        self._cached_pred_lap = -1
        self._cached_predictions = None
        self._tyre_deg = TyreDegTracker(total_laps=total_laps)
        self._sector_tracker = SectorAndLapTracker()

        return self.status()

    @staticmethod
    def _index_timeseries(
        rows: Sequence[Mapping[str, Any]],
        fields: Sequence[Tuple[str, Any]],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        if not rows:
            return {}
        df = pd.DataFrame(list(rows))
        if df.empty or "driver_number" not in df.columns or "date" not in df.columns:
            return {}
        df = df.dropna(subset=["driver_number", "date"]).copy()
        if df.empty:
            return {}
        df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
        df = df.dropna(subset=["driver_number"])
        df["driver_number"] = df["driver_number"].astype(int)
        df["_t"] = to_epoch_seconds(df["date"])
        df = df.dropna(subset=["_t"]).sort_values(["driver_number", "_t"], kind="mergesort")

        indexed: Dict[int, Dict[str, np.ndarray]] = {}
        for dn, grp in df.groupby("driver_number", sort=False):
            entry: Dict[str, np.ndarray] = {"t": grp["_t"].to_numpy(dtype=float)}
            for col, _ in fields:
                if col in grp.columns:
                    entry[col] = pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=float)
                else:
                    entry[col] = np.full(len(grp), np.nan, dtype=float)
            indexed[int(dn)] = entry
        return indexed

    def _try_build_race_context(self) -> Optional[RaceContext]:
        if self._meta is None:
            return None
        if self.predictor is not None and self._meta.round_number is not None:
            try:
                return self.predictor.build_race_context(
                    self._meta.season, self._meta.round_number, use_openf1=False
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Using lightweight replay RaceContext: %s", exc)
        # Lightweight context so ML inference can still run without Jolpica round match
        rows = []
        for idx, (dn, dm) in enumerate(sorted(self._drivers_meta.items()), start=1):
            grid_pos = self._grid_by_driver.get(dn, idx)
            rows.append(
                {
                    "season": self._meta.season,
                    "round": self._meta.round_number or 1,
                    "driver_number": dn,
                    "driver_code": dm["code"],
                    "driver_name": dm["full_name"],
                    "constructor_name": dm["team_name"],
                    "grid_position": grid_pos,
                    "driver_win_rate_10": 0.05,
                    "driver_podium_rate_10": 0.15,
                    "driver_avg_finish_5": float(grid_pos),
                    "team_points_avg_5": max(1.0, 22.0 - grid_pos),
                    "team_win_rate_10": 0.05,
                }
            )
        return RaceContext(
            season=self._meta.season,
            round_number=self._meta.round_number or 1,
            race_name=self._meta.grand_prix,
            circuit_id=self._meta.circuit_id,
            circuit_name=self._meta.circuit_name,
            race_date=None,
            grid_source="openf1_starting_grid",
            features=pd.DataFrame(rows),
            meeting_key=self._meta.meeting_key,
            session_key=self._meta.session_key,
            rotation_deg=self._meta.rotation_deg,
        )

    # ------------------------------------------------------------------ #
    # Playback controls
    # ------------------------------------------------------------------ #
    def play(self) -> Dict[str, Any]:
        self._paused = False
        if self._finished and self._meta is not None:
            self._cursor_s = self._meta.start_s
            self._finished = False
            self._tyre_deg.reset()
            self._sector_tracker.reset()
            self._cached_pred_lap = -1
        return self.status()

    def pause(self) -> Dict[str, Any]:
        self._paused = True
        return self.status()

    def set_speed(self, speed: float) -> Dict[str, Any]:
        clamped = max(1.0, min(10.0, float(speed)))
        self._speed = clamped
        return self.status()

    def seek_lap(self, lap_number: int) -> Dict[str, Any]:
        if self._meta is None:
            raise ValueError("No replay session loaded.")
        target_lap = max(1, min(self._meta.total_laps, int(lap_number)))
        if target_lap in self._lap_start_times:
            self._cursor_s = self._lap_start_times[target_lap]
        else:
            frac = (target_lap - 1) / max(1, self._meta.total_laps)
            self._cursor_s = self._meta.start_s + frac * (self._meta.end_s - self._meta.start_s)
        self._finished = False
        self._tyre_deg.reset()
        self._sector_tracker.reset()
        self._cached_pred_lap = -1
        self._cached_predictions = None
        return self.status()

    # ------------------------------------------------------------------ #
    # Frame construction (250 ms / 4 Hz)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lookup_interp(series: Optional[Dict[str, np.ndarray]], t_s: float, col: str) -> Optional[float]:
        if not series:
            return None
        t_arr = series["t"]
        if t_arr.size == 0:
            return None
        vals = series[col]
        idx = int(np.searchsorted(t_arr, t_s, side="right")) - 1
        if idx < 0:
            val = vals[0]
            return None if not np.isfinite(val) else float(val)
        if idx + 1 < t_arr.size:
            t0, t1 = float(t_arr[idx]), float(t_arr[idx + 1])
            v0, v1 = float(vals[idx]), float(vals[idx + 1])
            if np.isfinite(v0) and np.isfinite(v1) and 0.0 < (t1 - t0) <= 2.5:
                alpha = (t_s - t0) / (t1 - t0)
                return float(v0 + alpha * (v1 - v0))
        val = vals[idx]
        return None if not np.isfinite(val) else float(val)

    @staticmethod
    def _lookup_prev(series: Optional[Dict[str, np.ndarray]], t_s: float, col: str) -> Optional[float]:
        if not series:
            return None
        t_arr = series["t"]
        if t_arr.size == 0:
            return None
        idx = int(np.searchsorted(t_arr, t_s, side="right")) - 1
        if idx < 0:
            return None
        val = series[col][idx]
        return None if not np.isfinite(val) else float(val)

    def _compute_predictions(
        self,
        drivers: List[Dict[str, Any]],
        lap_current: int,
        track_status: str,
    ) -> Dict[str, Any]:
        """Compute Top-3 Podium predictions (300% sum) & Win predictions (100% sum)."""
        if (
            self.predictor is not None
            and self.predictor.in_race_available
            and self._race_context is not None
            and self._meta is not None
        ):
            try:
                pace_map = {
                    int(d["driver_number"]): p
                    for d in drivers
                    if (p := self._tyre_deg.get_clean_pace_3(int(d["driver_number"]))) is not None
                }
                return self.predictor.predict_live(
                    self._race_context,
                    drivers,
                    lap=lap_current,
                    total_laps=self._meta.total_laps,
                    track_state=track_status,
                    pace_by_driver=pace_map,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Live ML predictor failed on lap %s; falling back to heuristic: %s", lap_current, exc)

        # Deterministic physics/position-gap fallback when models aren't trained on real data yet
        active_mask = [not is_driver_out(d) for d in drivers]
        lap_frac = min(1.0, max(0.05, lap_current / max(1, (self._meta.total_laps if self._meta else 57))))
        sharpness = 0.42 + 0.55 * lap_frac

        raw_podium_scores: List[float] = []
        raw_win_scores: List[float] = []
        for d, active in zip(drivers, active_mask):
            if not active:
                raw_podium_scores.append(0.0)
                raw_win_scores.append(0.0)
                continue
            pos = float(d.get("position") or 20)
            gap = float(d.get("gap_to_leader_s") or 0.0)
            laps_down = float(d.get("laps_down") or 0)
            eff_gap = max(0.0, gap) + laps_down * 90.0
            # Higher score for top positions & tight gaps
            p_score = math.exp(-sharpness * (pos - 1.0)) * math.exp(-eff_gap / (35.0 + 45.0 * (1.0 - lap_frac)))
            w_score = math.exp(-(sharpness * 1.45) * (pos - 1.0)) * math.exp(-eff_gap / (20.0 + 30.0 * (1.0 - lap_frac)))
            raw_podium_scores.append(max(1e-6, p_score))
            raw_win_scores.append(max(1e-6, w_score))

        podium_pcts = normalize_podium_to_percent(
            raw_podium_scores,
            active_mask=active_mask,
            decimals=config.PROBABILITY_DECIMALS,
            target_sum=300.0,
            max_per_driver=100.0,
        )
        active_win_raw = [s for s, a in zip(raw_win_scores, active_mask) if a]
        active_win_pcts = normalize_to_percent(active_win_raw, decimals=config.PROBABILITY_DECIMALS)
        win_iter = iter(active_win_pcts)
        win_pcts = [next(win_iter) if a else 0.0 for a in active_mask]

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

        sum_podium = round(float(sum(i["podium_prob_pct"] for i in items)), config.PROBABILITY_DECIMALS)
        sum_win = round(float(sum(i["win_prob_pct"] for i in items)), config.PROBABILITY_DECIMALS)
        return {
            "stage": "in_race",
            "target": "top3_podium",
            "model_version": "heuristic-top3-fallback-v1",
            "generated_at": _now_iso(),
            "lap": int(lap_current),
            "items": items,
            "sum_podium_pct": sum_podium,
            "sum_win_pct": sum_win,
            "sum_pct": sum_podium,
        }

    def build_frame(self, t_s: Optional[float] = None) -> Dict[str, Any]:
        """Build one contract-valid ``replay_frame`` packet at timestamp ``t_s``."""
        if self._meta is None:
            raise ValueError("No replay session loaded.")
        cur_s = float(self._cursor_s if t_s is None else t_s)

        state_code = track_state_at(self._track_timeline, cur_s)
        track_status = TRACK_STATE_NAMES.get(state_code, "GREEN")
        if cur_s >= self._meta.end_s - 0.5:
            track_status = "CHEQUERED"

        # Determine weather at cur_s
        weather: Optional[Dict[str, Any]] = None
        if self._weather_times.size > 0:
            w_idx = int(np.searchsorted(self._weather_times, cur_s, side="right")) - 1
            if w_idx >= 0:
                weather = self._weather_rows[w_idx]

        # Recent race control messages up to cur_s
        rc_tail: List[Dict[str, Any]] = []
        if self._rc_times.size > 0:
            rc_end = int(np.searchsorted(self._rc_times, cur_s, side="right"))
            start_idx = max(0, rc_end - config.RACE_CONTROL_TAIL)
            rc_tail = self._rc_rows[start_idx:rc_end]

        # Recent team radio clips up to cur_s (enriched with cached faster-whisper transcripts)
        radio_tail: List[Dict[str, Any]] = []
        if self._radio_times.size > 0:
            r_end = int(np.searchsorted(self._radio_times, cur_s, side="right"))
            for clip in self._radio_rows[max(0, r_end - 25) : r_end]:
                row_copy = dict(clip)
                if self.transcriber is not None:
                    row_copy["transcript"] = self.transcriber.get_cached(clip["recording_url"])
                radio_tail.append(row_copy)

        gps_available = bool(self._loc_by_driver)
        telemetry_available = bool(self._car_by_driver)

        # Build per-driver states
        leader_lap = 1
        driver_completed_laps: Dict[int, int] = {}
        for dn, laps in self._laps_by_driver.items():
            comp = 0
            for lap in laps:
                end_s = lap.get("end_s")
                start_s = lap.get("start_s")
                if end_s is not None and end_s <= cur_s:
                    comp = max(comp, int(lap["lap_number"]))
                elif start_s is not None and start_s <= cur_s:
                    leader_lap = max(leader_lap, int(lap["lap_number"]))
            driver_completed_laps[dn] = comp
            leader_lap = max(leader_lap, comp)
        leader_lap = min(self._meta.total_laps, max(1, leader_lap))

        drivers_list: List[Dict[str, Any]] = []
        for dn, meta in self._drivers_meta.items():
            pos_val = self._lookup_prev(self._pos_by_driver.get(dn), cur_s, "position")
            position = int(round(pos_val)) if pos_val is not None and pos_val >= 1 else None
            gap = self._lookup_prev(self._int_by_driver.get(dn), cur_s, "gap_to_leader")
            interval = self._lookup_prev(self._int_by_driver.get(dn), cur_s, "interval")
            if position == 1:
                gap = 0.0
                interval = 0.0

            retired = dn in self._retired_after and cur_s >= self._retired_after[dn]
            completed = driver_completed_laps.get(dn, 0)
            laps_down = max(0, leader_lap - completed - 1) if not retired else 0

            # Pit stops up to cur_s
            pits_so_far: List[Dict[str, Any]] = []
            pit_in_laps: set[int] = set()
            in_pit = False
            for p in self._pits_by_driver.get(dn, []):
                pt = p.get("_t")
                plap = p.get("lap_number")
                if (pt is not None and pt <= cur_s) or (pt is None and plap is not None and plap <= completed):
                    pits_so_far.append(
                        {
                            "lap_number": plap,
                            "pit_duration_s": p.get("pit_duration_s"),
                            "date": p.get("date"),
                        }
                    )
                    if plap is not None:
                        pit_in_laps.add(int(plap))
                    if pt is not None:
                        dur = float(p.get("pit_duration_s") or 22.0)
                        if pt <= cur_s <= pt + dur:
                            in_pit = True

            # Stints up to current lap
            stints_all = self._stints_by_driver.get(dn, [])
            active_stint: Optional[Dict[str, Any]] = None
            stints_visible: List[Dict[str, Any]] = []
            current_lap_for_stint = max(1, completed)
            for st in stints_all:
                if st["lap_start"] <= current_lap_for_stint:
                    active_stint = st
                    stints_visible.append(dict(st))

            compound = active_stint["compound"] if active_stint else "UNKNOWN"
            stint_number = active_stint["stint_number"] if active_stint else 1
            tyre_age = (
                int(active_stint["tyre_age_at_start"] + max(0, current_lap_for_stint - active_stint["lap_start"]))
                if active_stint
                else None
            )
            is_new = (active_stint["tyre_age_at_start"] == 0) if active_stint else None

            # Feed completed laps into TyreDegTracker
            last_lap_time: Optional[float] = None
            best_lap_time: Optional[float] = None
            for lap in self._laps_by_driver.get(dn, []):
                end_s = lap.get("end_s")
                if end_s is None or end_s > cur_s:
                    continue
                lno = int(lap["lap_number"])
                dur = lap.get("lap_duration")
                if dur is not None and dur > 0:
                    last_lap_time = float(dur)
                    if best_lap_time is None or dur < best_lap_time:
                        best_lap_time = float(dur)
                    lap_stint = 1
                    for st in stints_all:
                        if st["lap_start"] <= lno:
                            lap_stint = int(st["stint_number"])
                    lap_flag = TRACK_STATE_NAMES.get(track_state_at(self._track_timeline, float(end_s)), "GREEN")
                    self._tyre_deg.record_lap(
                        dn,
                        lno,
                        float(dur),
                        stint_number=lap_stint,
                        is_pit_out_lap=bool(lap.get("is_pit_out_lap")),
                        is_pit_in_lap=(lno in pit_in_laps),
                        track_status=lap_flag,
                    )
                self._sector_tracker.record_lap(
                    dn,
                    meta["code"],
                    lap_number=lno,
                    lap_time_s=float(dur) if dur is not None and dur > 0 else None,
                    s1_s=lap.get("s1_s"),
                    s2_s=lap.get("s2_s"),
                    s3_s=lap.get("s3_s"),
                    team_name=meta["team_name"],
                    team_colour=meta["team_colour"],
                )

            deg_res = self._tyre_deg.get_driver_deg(dn)
            tyre_health = deg_res.tyre_health if deg_res is not None else "UNKNOWN"
            tyre_deg_dict = deg_res.to_contract() if deg_res is not None else None

            # Location (rotated x, y) & Car Telemetry (nullable drs)
            rx: Optional[float] = None
            ry: Optional[float] = None
            rz: Optional[float] = None
            if gps_available and dn in self._loc_by_driver:
                raw_x = self._lookup_interp(self._loc_by_driver.get(dn), cur_s, "x")
                raw_y = self._lookup_interp(self._loc_by_driver.get(dn), cur_s, "y")
                rz = self._lookup_interp(self._loc_by_driver.get(dn), cur_s, "z")
                rx, ry = rotate_xy(raw_x, raw_y, self._meta.rotation_deg)

            telemetry: Optional[Dict[str, Any]] = None
            if telemetry_available and dn in self._car_by_driver:
                c_ser = self._car_by_driver[dn]
                spd = self._lookup_interp(c_ser, cur_s, "speed")
                thr = self._lookup_interp(c_ser, cur_s, "throttle")
                brk = self._lookup_interp(c_ser, cur_s, "brake")
                gear = self._lookup_prev(c_ser, cur_s, "n_gear")
                rpm = self._lookup_interp(c_ser, cur_s, "rpm")
                drs_raw = self._lookup_prev(c_ser, cur_s, "drs")
                drs_int = int(round(drs_raw)) if drs_raw is not None else None
                telemetry = {
                    "date": _epoch_to_iso(cur_s),
                    "speed_kph": round(spd, 1) if spd is not None else None,
                    "throttle_pct": round(thr, 1) if thr is not None else None,
                    "brake_pct": round(brk, 1) if brk is not None else None,
                    "gear": int(round(gear)) if gear is not None else None,
                    "rpm": int(round(rpm)) if rpm is not None else None,
                    "drs": drs_int,
                    "drs_active": drs_to_active(drs_int),
                    "x": rx,
                    "y": ry,
                    "z": round(rz, 2) if rz is not None else None,
                }
            elif rx is not None and ry is not None:
                telemetry = {
                    "date": _epoch_to_iso(cur_s),
                    "speed_kph": None,
                    "throttle_pct": None,
                    "brake_pct": None,
                    "gear": None,
                    "rpm": None,
                    "drs": None,
                    "drs_active": None,
                    "x": rx,
                    "y": ry,
                    "z": round(rz, 2) if rz is not None else None,
                }

            drivers_list.append(
                {
                    "driver_number": dn,
                    "code": meta["code"],
                    "full_name": meta["full_name"],
                    "team_name": meta["team_name"],
                    "team_colour": meta["team_colour"],
                    "position": None if retired else position,
                    "grid_position": self._grid_by_driver.get(dn),
                    "gap_to_leader_s": None if retired else (round(gap, 3) if gap is not None else None),
                    "interval_s": None if retired else (round(interval, 3) if interval is not None else None),
                    "laps_down": int(laps_down),
                    "last_lap_time_s": round(last_lap_time, 3) if last_lap_time is not None else None,
                    "best_lap_time_s": round(best_lap_time, 3) if best_lap_time is not None else None,
                    "tyre": {
                        "compound": compound,
                        "age_laps": tyre_age,
                        "stint_number": stint_number,
                        "is_new": is_new,
                        "health": tyre_health,
                    },
                    "tyre_health": tyre_health,
                    "stints": stints_visible,
                    "pit_stops": pits_so_far,
                    "in_pit": bool(in_pit),
                    "retired": bool(retired),
                    "status": "DNF" if retired else ("PIT" if in_pit else "RUNNING"),
                    "x": rx,
                    "y": ry,
                    "tyre_deg": tyre_deg_dict,
                    "podium_prob_pct": 0.0,
                    "win_prob_pct": 0.0,
                    "win_probability_pct": 0.0,
                    "telemetry": telemetry,
                }
            )

        # Attach per-driver sector states after all drivers' laps up to cur_s have been recorded
        for d in drivers_list:
            d["sectors"] = self._sector_tracker.driver_sectors_contract(int(d["driver_number"]))

        # Sort timing order by running position (retired at bottom)
        drivers_list.sort(
            key=lambda d: (
                1 if d["retired"] else 0,
                d["position"] if d["position"] is not None else 999,
                d["driver_number"],
            )
        )

        # Update Top-3 ML predictions on lap change (or first frame)
        if drivers_list and (self._cached_predictions is None or self._cached_pred_lap != leader_lap):
            self._cached_predictions = self._compute_predictions(drivers_list, leader_lap, track_status)
            self._cached_pred_lap = leader_lap

        if self._cached_predictions is not None:
            p_map = {int(i["driver_number"]): i["podium_prob_pct"] for i in self._cached_predictions["items"]}
            w_map = {int(i["driver_number"]): i["win_prob_pct"] for i in self._cached_predictions["items"]}
            for d in drivers_list:
                dn = int(d["driver_number"])
                out = is_driver_out(d)
                d["podium_prob_pct"] = 0.0 if out else float(p_map.get(dn, 0.0))
                d["win_prob_pct"] = 0.0 if out else float(w_map.get(dn, 0.0))
                d["win_probability_pct"] = d["win_prob_pct"]

        elapsed_s = max(0.0, cur_s - self._meta.start_s)
        duration_s = max(0.0, self._meta.end_s - self._meta.start_s)

        race_state = {
            "lap_current": int(leader_lap),
            "lap_total": int(self._meta.total_laps),
            "track_status": track_status,
            "weather": weather,
            "drivers": drivers_list,
            "race_control": rc_tail,
            "team_radio": radio_tail,
            "session_best": self._sector_tracker.session_best_contract(),
            "predictions": self._cached_predictions,
        }

        return {
            "schema_version": config.SCHEMA_VERSION,
            "type": "replay_frame",
            "timestamp": _now_iso(),
            "sequence": self.next_sequence(),
            "source": "openf1_replay",
            "mode": "replay",
            "capabilities": {
                "gps_available": gps_available,
                "telemetry_available": telemetry_available,
            },
            "meeting_key": self._meta.meeting_key,
            "session_key": self._meta.session_key,
            "season": self._meta.season,
            "round": self._meta.round_number,
            "grand_prix": self._meta.grand_prix,
            "circuit": {
                "circuit_id": self._meta.circuit_id,
                "slug": self._circuit_slug,
                "name": self._meta.circuit_name,
                "rotation_deg": self._meta.rotation_deg,
                "track_outline": self._track_outline,
                "sector_splits": self._sector_splits,
                "start_finish": self._start_finish,
                "corners": self._circuit_corners,
            },
            "replay": {
                "session_start": _epoch_to_iso(self._meta.start_s),
                "session_time": _epoch_to_iso(cur_s),
                "elapsed_s": round(elapsed_s, 3),
                "duration_s": round(duration_s, 3),
                "speed": float(self._speed),
                "paused": bool(self._paused),
                "finished": bool(self._finished),
            },
            "race": race_state,
            "tick": race_state,
        }

    # ------------------------------------------------------------------ #
    # Async loop lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="f1-replay-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        try:
            while self._running:
                if self._meta is not None:
                    frame = self.build_frame(self._cursor_s)
                    await self.publish(frame)
                    if not self._paused and not self._finished:
                        self._cursor_s += self.tick_interval_s * self._speed
                        if self._cursor_s >= self._meta.end_s:
                            self._cursor_s = self._meta.end_s
                            self._finished = True
                            self._paused = True
                await asyncio.sleep(self.tick_interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Replay loop error: %s", exc)
            self._running = False

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "running": self._running,
            "loaded": self._meta is not None,
            "session_key": self._meta.session_key if self._meta else None,
            "grand_prix": self._meta.grand_prix if self._meta else None,
            "season": self._meta.season if self._meta else None,
            "speed": self._speed,
            "paused": self._paused,
            "finished": self._finished,
            "elapsed_s": round(max(0.0, self._cursor_s - self._meta.start_s), 2) if self._meta else 0.0,
            "duration_s": round(max(0.0, self._meta.end_s - self._meta.start_s), 2) if self._meta else 0.0,
            "total_laps": self._meta.total_laps if self._meta else None,
            "gps_available": bool(self._loc_by_driver),
            "telemetry_available": bool(self._car_by_driver),
        }
