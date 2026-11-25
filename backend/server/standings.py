"""Pre-race Championship Standings (WDC & WCC) and Qualifying / Grid lookup.

Powers the dedicated "Before-Race Standings & Weekend Hub" screen via
``GET /api/standings`` using keyless Jolpica-F1 (cached on disk).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import config
from clients.jolpica_client import JolpicaClient, JolpicaError

TEAM_COLOURS_FALLBACK: Dict[str, str] = {
    "red_bull": "3671C6",
    "mclaren": "FF8000",
    "ferrari": "E80020",
    "mercedes": "27F4D2",
    "aston_martin": "229971",
    "alpine": "00A1E8",
    "williams": "1868DB",
    "rb": "6C98FF",
    "racing_bulls": "6C98FF",
    "haas": "9C9FA2",
    "sauber": "52E252",
    "audi": "F50537",
    "cadillac": "909090",
}

__all__ = ["get_pre_race_standings"]


def _team_hex(constructor_id: Optional[str], constructor_name: Optional[str]) -> str:
    cid = (constructor_id or "").lower().strip()
    if cid in TEAM_COLOURS_FALLBACK:
        return TEAM_COLOURS_FALLBACK[cid]
    name = (constructor_name or "").lower()
    for key, val in TEAM_COLOURS_FALLBACK.items():
        if key.replace("_", " ") in name:
            return val
    return "64748B"


def get_pre_race_standings(
    jolpica: JolpicaClient,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch WDC Driver Standings, WCC Constructor Standings, and Latest Qualifying Grid."""
    target_season = int(season or config.current_season())
    used_season = target_season

    drivers_out: List[Dict[str, Any]] = []
    constructors_out: List[Dict[str, Any]] = []
    standings_round: Optional[int] = None

    for cand in (target_season, target_season - 1):
        try:
            mr_drv = jolpica.get(f"{cand}/driverStandings", {"limit": 100}, season=cand)
            lists = mr_drv.get("StandingsTable", {}).get("StandingsLists", [])
            if lists:
                used_season = cand
                standings_round = int(lists[0].get("round") or 0) or None
                for item in lists[0].get("DriverStandings", []):
                    drv = item.get("Driver") or {}
                    cons_list = item.get("Constructors") or []
                    cons = cons_list[-1] if cons_list else {}
                    c_id = cons.get("constructorId")
                    c_name = cons.get("name") or "Unknown Team"
                    drivers_out.append(
                        {
                            "position": int(item.get("position") or len(drivers_out) + 1),
                            "driver_id": drv.get("driverId"),
                            "driver_code": drv.get("code") or str(drv.get("familyName", ""))[:3].upper(),
                            "driver_number": int(drv["permanentNumber"]) if str(drv.get("permanentNumber", "")).isdigit() else None,
                            "full_name": f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip(),
                            "nationality": drv.get("nationality"),
                            "team_name": c_name,
                            "team_colour": _team_hex(c_id, c_name),
                            "points": float(item.get("points") or 0.0),
                            "wins": int(item.get("wins") or 0),
                        }
                    )
                break
        except JolpicaError:
            continue

    try:
        mr_con = jolpica.get(f"{used_season}/constructorStandings", {"limit": 100}, season=used_season)
        c_lists = mr_con.get("StandingsTable", {}).get("StandingsLists", [])
        if c_lists:
            for item in c_lists[0].get("ConstructorStandings", []):
                cons = item.get("Constructor") or {}
                c_id = cons.get("constructorId")
                c_name = cons.get("name") or "Unknown Team"
                constructors_out.append(
                    {
                        "position": int(item.get("position") or len(constructors_out) + 1),
                        "constructor_id": c_id,
                        "team_name": c_name,
                        "nationality": cons.get("nationality"),
                        "team_colour": _team_hex(c_id, c_name),
                        "points": float(item.get("points") or 0.0),
                        "wins": int(item.get("wins") or 0),
                    }
                )
    except JolpicaError:
        pass

    # Latest qualifying / starting grid
    latest_quali: Optional[Dict[str, Any]] = None
    try:
        q_races = [r for r in jolpica.get_qualifying_results(used_season) if r.get("QualifyingResults")]
        if q_races:
            last_q = q_races[-1]
            grid_rows = []
            for q in last_q.get("QualifyingResults", []):
                drv = q.get("Driver") or {}
                cons = q.get("Constructor") or {}
                c_id = cons.get("constructorId")
                c_name = cons.get("name") or "Unknown Team"
                grid_rows.append(
                    {
                        "position": int(q.get("position") or len(grid_rows) + 1),
                        "driver_code": drv.get("code") or str(drv.get("familyName", ""))[:3].upper(),
                        "driver_number": int(q["number"]) if str(q.get("number", "")).isdigit() else None,
                        "full_name": f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip(),
                        "team_name": c_name,
                        "team_colour": _team_hex(c_id, c_name),
                        "q1": q.get("Q1"),
                        "q2": q.get("Q2"),
                        "q3": q.get("Q3"),
                    }
                )
            latest_quali = {
                "season": int(last_q.get("season") or used_season),
                "round": int(last_q.get("round") or 1),
                "grand_prix": last_q.get("raceName"),
                "circuit_name": (last_q.get("Circuit") or {}).get("circuitName"),
                "date": last_q.get("date"),
                "results": grid_rows,
            }
    except JolpicaError:
        pass

    return {
        "season": used_season,
        "after_round": standings_round,
        "drivers": drivers_out,
        "constructors": constructors_out,
        "latest_qualifying": latest_quali,
    }
