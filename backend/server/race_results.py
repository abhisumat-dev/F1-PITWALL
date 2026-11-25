"""Final Race Classification & Post-Race Summary Loader (No Replay Needed).

Combines official Jolpica-F1 final race classification (exact final grid order
after the final lap, gaps/statuses, points, fastest lap) with lightweight
OpenF1 post-race metadata (complete tyre stint timelines across all laps,
S1/S2/S3 fastest sector records, weather, FIA Race Control, and Team Radio)
without downloading heavy 4 Hz GPS/car telemetry streams.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from analytics.sectors_fastest import SectorAndLapTracker
from clients.jolpica_client import JolpicaClient, JolpicaError, parse_lap_time
from clients.openf1_client import OpenF1Client, OpenF1Error
from server.circuits import resolve_circuit_2d
from server.standings import TEAM_COLOURS_FALLBACK

LOGGER = logging.getLogger(__name__)

__all__ = ["get_final_race_results"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _team_hex(constructor_id: Optional[str], constructor_name: Optional[str]) -> str:
    cid = (constructor_id or "").lower().strip()
    if cid in TEAM_COLOURS_FALLBACK:
        return TEAM_COLOURS_FALLBACK[cid]
    name = (constructor_name or "").lower()
    for key, val in TEAM_COLOURS_FALLBACK.items():
        if key.replace("_", " ") in name:
            return val
    return "64748B"


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


def _format_circuit_payload(c2d: Optional[Dict[str, Any]], circuit_id: Optional[str], name: str) -> Dict[str, Any]:
    if not c2d:
        return {"circuit_id": circuit_id, "name": name, "slug": None, "track_outline": None}
    sf = c2d.get("start_finish")
    return {
        "circuit_id": circuit_id or c2d.get("slug"),
        "slug": c2d.get("slug"),
        "name": c2d.get("short_name") or name,
        "rotation_deg": float(c2d.get("rotation_deg") or 0.0),
        "track_outline": [{"x": float(p["x"]), "y": float(p["y"])} for p in (c2d.get("points") or [])] or None,
        "sector_splits": [int(i) for i in (c2d.get("sector_splits") or [])] or None,
        "start_finish": (
            {"x": float(sf["x"]), "y": float(sf["y"])}
            if isinstance(sf, dict) and "x" in sf and "y" in sf
            else None
        ),
        "corners": [
            {
                "number": int(c["number"]),
                "letter": str(c.get("letter") or "") or None,
                "x": float(c["x"]),
                "y": float(c["y"]),
            }
            for c in (c2d.get("corners") or [])
        ] or None,
    }


def _build_session_best_from_openf1(
    laps: List[Dict[str, Any]],
    drivers_by_num: Dict[int, Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Compute overall fastest lap, S1/S2/S3 fastest sectors, and per-driver sector status."""
    tracker = SectorAndLapTracker()
    best_per_drv: Dict[int, Dict[str, Optional[float]]] = {}

    for lap in laps:
        dn = lap.get("driver_number")
        if dn is None:
            continue
        try:
            dn_int = int(dn)
        except (TypeError, ValueError):
            continue
        meta = drivers_by_num.get(dn_int, {})
        code = str(meta.get("code") or dn_int)
        team = str(meta.get("team_name") or "")
        colour = meta.get("team_colour")
        lap_num = int(lap.get("lap_number") or 0) or None

        s1 = lap.get("duration_sector_1")
        s2 = lap.get("duration_sector_2")
        s3 = lap.get("duration_sector_3")
        lt = lap.get("lap_duration")

        tracker.record_lap(
            driver_number=dn_int,
            driver_code=code,
            team_name=team,
            team_colour=colour,
            lap_number=lap_num,
            lap_time_s=float(lt) if isinstance(lt, (int, float)) else None,
            s1_s=float(s1) if isinstance(s1, (int, float)) else None,
            s2_s=float(s2) if isinstance(s2, (int, float)) else None,
            s3_s=float(s3) if isinstance(s3, (int, float)) else None,
        )

        rec = best_per_drv.setdefault(dn_int, {"s1": None, "s2": None, "s3": None, "lap": None})
        for key, val in (("s1", s1), ("s2", s2), ("s3", s3), ("lap", lt)):
            if isinstance(val, (int, float)) and val > 0:
                if rec[key] is None or float(val) < float(rec[key]):
                    rec[key] = float(val)

    sb = tracker.session_best_contract()
    overall_s1 = (sb.get("fastest_s1") or {}).get("time_s")
    overall_s2 = (sb.get("fastest_s2") or {}).get("time_s")
    overall_s3 = (sb.get("fastest_s3") or {}).get("time_s")

    sectors_out: Dict[int, Dict[str, Any]] = {}
    for dn_int, rec in best_per_drv.items():
        def _status(val: Optional[float], overall: Optional[float]) -> str:
            if val is None:
                return "NONE"
            if overall is not None and abs(val - overall) < 0.002:
                return "PURPLE"
            return "GREEN"

        sectors_out[dn_int] = {
            "s1_s": rec["s1"],
            "s2_s": rec["s2"],
            "s3_s": rec["s3"],
            "s1_status": _status(rec["s1"], overall_s1),
            "s2_status": _status(rec["s2"], overall_s2),
            "s3_status": _status(rec["s3"], overall_s3),
            "best_lap_s": rec["lap"],
        }

    return sb, sectors_out


def get_final_race_results(
    jolpica: JolpicaClient,
    openf1: OpenF1Client,
    *,
    season: int,
    round_number: Optional[int] = None,
    session_key: Optional[int] = None,
    sessions_list: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return final classification after the final lap for a completed GP."""
    resolved_season = int(season)
    resolved_round = int(round_number) if round_number is not None else None
    resolved_sk = int(session_key) if session_key is not None else None
    matched_session: Optional[Dict[str, Any]] = None

    if sessions_list:
        if resolved_sk is not None:
            matched_session = next((s for s in sessions_list if s.get("session_key") == resolved_sk), None)
            if matched_session and resolved_round is None:
                resolved_round = int(matched_session.get("round") or 1)
        elif resolved_round is not None:
            matched_session = next((s for s in sessions_list if int(s.get("round") or 0) == resolved_round), None)
            if matched_session and resolved_sk is None and matched_session.get("session_key"):
                resolved_sk = int(matched_session["session_key"])
        else:
            completed_sessions = [s for s in sessions_list if s.get("completed")]
            matched_session = completed_sessions[-1] if completed_sessions else sessions_list[-1]
            if matched_session:
                resolved_round = int(matched_session.get("round") or 1)
                if matched_session.get("session_key"):
                    resolved_sk = int(matched_session["session_key"])

    if resolved_round is None:
        latest = jolpica.get_latest_round_with_results(resolved_season)
        resolved_round = latest or 1

    # 1. Fetch official Jolpica final race classification
    race_obj: Optional[Dict[str, Any]] = None
    try:
        races = jolpica.get_race_results(resolved_season, resolved_round)
        if races and races[0].get("Results"):
            race_obj = races[0]
    except JolpicaError as exc:
        LOGGER.warning("Jolpica race results lookup failed for %s R%s: %s", resolved_season, resolved_round, exc)

    # Check schedule metadata if race has not been completed yet
    sched_race: Optional[Dict[str, Any]] = None
    if race_obj is None:
        try:
            sched_race = jolpica.get_race(resolved_season, resolved_round)
        except JolpicaError:
            sched_race = None

        gp_name = (
            (sched_race or {}).get("raceName")
            or (matched_session or {}).get("grand_prix")
            or f"Round {resolved_round}"
        )
        circ_info = (sched_race or {}).get("Circuit") or {}
        c2d = resolve_circuit_2d(
            circuit_id=circ_info.get("circuitId"),
            name=circ_info.get("circuitName") or (matched_session or {}).get("circuit_name"),
            grand_prix=gp_name,
        )
        return {
            "type": "final_race_results",
            "completed": False,
            "season": resolved_season,
            "round": resolved_round,
            "session_key": resolved_sk,
            "grand_prix": gp_name,
            "date": (sched_race or {}).get("date") or (matched_session or {}).get("date_start"),
            "lap_total": 0,
            "circuit": _format_circuit_payload(c2d, circ_info.get("circuitId"), circ_info.get("circuitName") or gp_name),
            "weather": None,
            "session_best": {"fastest_lap": None, "fastest_s1": None, "fastest_s2": None, "fastest_s3": None},
            "drivers": [],
            "race_control": [],
            "team_radio": [],
            "message": f"{gp_name} ({resolved_season} Round {resolved_round}) has not been completed yet.",
        }

    # Completed race! Parse Jolpica final classification
    gp_name = str(race_obj.get("raceName") or f"Round {resolved_round}")
    circ_info = race_obj.get("Circuit") or {}
    c2d = resolve_circuit_2d(
        circuit_id=circ_info.get("circuitId"),
        name=circ_info.get("circuitName") or (matched_session or {}).get("circuit_name"),
        grand_prix=gp_name,
    )
    results_rows = race_obj.get("Results") or []
    lap_total = max((int(r.get("laps") or 0) for r in results_rows), default=50) or 50

    drivers_out: List[Dict[str, Any]] = []
    drivers_by_num: Dict[int, Dict[str, Any]] = {}
    jolpica_fastest_lap: Optional[Dict[str, Any]] = None

    for idx, row in enumerate(results_rows, start=1):
        drv = row.get("Driver") or {}
        cons = row.get("Constructor") or {}
        pos = int(row.get("position") or idx)
        dn = int(row.get("number") or drv.get("permanentNumber") or idx)
        code = str(drv.get("code") or str(drv.get("familyName", ""))[:3].upper() or f"D{idx:02d}")
        full_name = f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip() or code
        team_name = str(cons.get("name") or "Unknown Team")
        team_colour = _team_hex(cons.get("constructorId"), team_name)
        grid_pos = int(row.get("grid") or 0) or None
        laps_done = int(row.get("laps") or 0)
        status = str(row.get("status") or "Finished")
        points = float(row.get("points") or 0.0)
        time_obj = row.get("Time") or {}
        time_text = time_obj.get("time")
        pos_text = str(row.get("positionText") or pos)

        retired = pos_text in {"R", "D", "E", "W", "F", "N"} or (
            status != "Finished" and not status.startswith("+") and "Lap" not in status
        )
        laps_down = max(0, lap_total - laps_done) if (not retired and "Lap" in status) else 0

        fl_obj = row.get("FastestLap") or {}
        fl_time_str = (fl_obj.get("Time") or {}).get("time")
        fl_time_s = parse_lap_time(fl_time_str)
        fl_lap_num = int(fl_obj.get("lap") or 0) or None
        fl_rank = int(fl_obj.get("rank") or 0) or None

        if fl_rank == 1 and fl_time_s:
            jolpica_fastest_lap = {
                "driver_number": dn,
                "driver_code": code,
                "team_name": team_name,
                "team_colour": team_colour,
                "time_s": fl_time_s,
                "formatted": fl_time_str,
                "lap_number": fl_lap_num,
            }

        if pos == 1:
            gap_display = time_text or "WINNER"
        elif retired:
            gap_display = "DNF" if status in {"Retired", "Collision", "Accident", "Mechanical"} else status[:12].upper()
        elif time_text:
            gap_display = str(time_text)
        else:
            gap_display = status

        d_entry: Dict[str, Any] = {
            "position": pos,
            "driver_number": dn,
            "code": code,
            "full_name": full_name,
            "team_name": team_name,
            "team_colour": team_colour,
            "grid_position": grid_pos,
            "laps_completed": laps_done,
            "laps_down": laps_down,
            "points": points,
            "status": status,
            "gap_display": gap_display,
            "retired": retired,
            "in_pit": False,
            "best_lap_time_s": fl_time_s,
            "best_lap_formatted": fl_time_str,
            "fastest_lap_rank": fl_rank,
            "tyre": {"compound": "UNKNOWN", "age_laps": None},
            "stints": [],
            "sectors": {"s1_status": "NONE", "s2_status": "NONE", "s3_status": "NONE"},
        }
        drivers_out.append(d_entry)
        drivers_by_num[dn] = d_entry

    # 2. Enrich with lightweight OpenF1 metadata (stints, laps/sectors, weather, FIA, team radio)
    session_best: Dict[str, Any] = {
        "fastest_lap": jolpica_fastest_lap,
        "fastest_s1": None,
        "fastest_s2": None,
        "fastest_s3": None,
    }
    weather_out: Optional[Dict[str, Any]] = None
    race_control_out: List[Dict[str, Any]] = []
    team_radio_out: List[Dict[str, Any]] = []

    if resolved_sk is not None:
        try:
            # Stints (complete full-race tyre strategy across laps 1..lap_total)
            raw_stints = openf1.get_stints(resolved_sk)
            stints_by_dn: Dict[int, List[Dict[str, Any]]] = {}
            for st in raw_stints:
                dn = st.get("driver_number")
                if dn is None:
                    continue
                dn_int = int(dn)
                stints_by_dn.setdefault(dn_int, []).append(
                    {
                        "stint_number": int(st.get("stint_number") or len(stints_by_dn.get(dn_int, [])) + 1),
                        "compound": _normalize_compound(st.get("compound")),
                        "lap_start": int(st.get("lap_start") or 1),
                        "lap_end": int(st["lap_end"]) if st.get("lap_end") is not None else None,
                        "tyre_age_at_start": int(st.get("tyre_age_at_start") or 0),
                    }
                )
            for dn_int, st_list in stints_by_dn.items():
                st_list.sort(key=lambda s: (s["stint_number"], s["lap_start"]))
                if dn_int in drivers_by_num:
                    drv_rec = drivers_by_num[dn_int]
                    drv_rec["stints"] = st_list
                    last_st = st_list[-1]
                    end_lap = last_st["lap_end"] or drv_rec["laps_completed"] or lap_total
                    age = max(1, end_lap - last_st["lap_start"] + 1 + last_st["tyre_age_at_start"])
                    drv_rec["tyre"] = {"compound": last_st["compound"], "age_laps": age}
        except OpenF1Error as exc:
            LOGGER.info("OpenF1 stints unavailable for sk=%s: %s", resolved_sk, exc)

        try:
            # Laps -> Fastest S1/S2/S3 & Overall Fastest Lap
            raw_laps = openf1.get_laps(resolved_sk)
            if raw_laps:
                sb, drv_sectors = _build_session_best_from_openf1(raw_laps, drivers_by_num)
                if sb.get("fastest_lap") and not session_best["fastest_lap"]:
                    session_best["fastest_lap"] = sb["fastest_lap"]
                session_best["fastest_s1"] = sb.get("fastest_s1")
                session_best["fastest_s2"] = sb.get("fastest_s2")
                session_best["fastest_s3"] = sb.get("fastest_s3")
                for dn_int, sec in drv_sectors.items():
                    if dn_int in drivers_by_num:
                        drivers_by_num[dn_int]["sectors"] = {
                            "s1_status": sec["s1_status"],
                            "s2_status": sec["s2_status"],
                            "s3_status": sec["s3_status"],
                        }
                        if drivers_by_num[dn_int]["best_lap_time_s"] is None and sec.get("best_lap_s"):
                            drivers_by_num[dn_int]["best_lap_time_s"] = sec["best_lap_s"]
        except OpenF1Error as exc:
            LOGGER.info("OpenF1 laps unavailable for sk=%s: %s", resolved_sk, exc)

        try:
            w_rows = openf1.get_weather(resolved_sk)
            if w_rows:
                w = w_rows[-1]
                weather_out = {
                    "air_temperature_c": w.get("air_temperature"),
                    "track_temperature_c": w.get("track_temperature"),
                    "humidity_pct": w.get("humidity"),
                    "rainfall": bool(w.get("rainfall")),
                }
        except OpenF1Error:
            pass

        try:
            rc_rows = openf1.get_race_control(resolved_sk)
            race_control_out = [
                {
                    "date": m.get("date"),
                    "lap_number": m.get("lap_number"),
                    "category": m.get("category"),
                    "flag": m.get("flag"),
                    "message": m.get("message") or "",
                }
                for m in rc_rows[-60:]
            ]
        except OpenF1Error:
            pass

        try:
            rad_rows = openf1.get("team_radio", {"session_key": resolved_sk})
            for r in rad_rows[-40:]:
                url = r.get("recording_url")
                if not url:
                    continue
                dn = int(r.get("driver_number") or 0)
                meta = drivers_by_num.get(dn, {})
                team_radio_out.append(
                    {
                        "date": r.get("date"),
                        "driver_number": dn,
                        "driver_code": meta.get("code") or str(dn),
                        "team_name": meta.get("team_name") or "Team Radio",
                        "team_colour": meta.get("team_colour") or "38BDF8",
                        "recording_url": url,
                    }
                )
        except OpenF1Error:
            pass

    return {
        "type": "final_race_results",
        "completed": True,
        "timestamp": _now_iso(),
        "season": resolved_season,
        "round": resolved_round,
        "session_key": resolved_sk,
        "grand_prix": gp_name,
        "date": race_obj.get("date"),
        "lap_total": lap_total,
        "circuit": _format_circuit_payload(c2d, circ_info.get("circuitId"), circ_info.get("circuitName") or gp_name),
        "weather": weather_out,
        "session_best": session_best,
        "drivers": drivers_out,
        "race_control": race_control_out,
        "team_radio": team_radio_out,
    }
