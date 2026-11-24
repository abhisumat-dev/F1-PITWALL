"""Upcoming Grand Prix schedule helper with IST, UK, and US timezone conversions.

Uses Jolpica-F1's cached season schedule (``get_schedule``) and Python's
standard ``zoneinfo`` (backed by ``tzdata`` on Windows) so Daylight Saving Time
transitions (BST/GMT, EDT/EST, CDT/CST, PDT/PST) are always accurate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

import config
from clients.jolpica_client import JolpicaClient, JolpicaError

TIMEZONE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("ist", "IST (India)", "Asia/Kolkata"),
    ("uk", "UK", "Europe/London"),
    ("us_eastern", "US Eastern", "America/New_York"),
    ("us_central", "US Central", "America/Chicago"),
    ("us_pacific", "US Pacific", "America/Los_Angeles"),
)

_SESSION_KEYS: Tuple[Tuple[str, str], ...] = (
    ("FirstPractice", "FP1"),
    ("SecondPractice", "FP2"),
    ("ThirdPractice", "FP3"),
    ("SprintQualifying", "Sprint Qualifying"),
    ("Sprint", "Sprint"),
    ("Qualifying", "Qualifying"),
)

__all__ = [
    "TIMEZONE_SPECS",
    "format_multi_timezone",
    "parse_jolpica_utc",
    "get_next_race_schedule",
]


def parse_jolpica_utc(date_str: Optional[str], time_str: Optional[str]) -> Tuple[Optional[datetime], bool]:
    """Parse Jolpica ``date`` (``YYYY-MM-DD``) and optional ``time`` (``HH:MM:SSZ``) into UTC datetime.

    Returns ``(dt_utc, time_confirmed)``. If ``time_str`` is missing, defaults to
    ``14:00:00Z`` and sets ``time_confirmed=False``.
    """
    if not date_str:
        return None, False
    cleaned_date = str(date_str).strip()
    cleaned_time = str(time_str).strip() if time_str else ""
    time_confirmed = bool(cleaned_time)
    if not cleaned_time:
        cleaned_time = "14:00:00Z"
    if cleaned_time.endswith("Z"):
        cleaned_time = cleaned_time[:-1] + "+00:00"
    elif "+" not in cleaned_time and "-" not in cleaned_time:
        cleaned_time += "+00:00"
    try:
        dt = datetime.fromisoformat(f"{cleaned_date}T{cleaned_time}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc), time_confirmed
    except ValueError:
        return None, False


def format_multi_timezone(dt_utc: datetime) -> Dict[str, Dict[str, str]]:
    """Format a UTC datetime into IST, UK, US Eastern, US Central, and US Pacific strings."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(timezone.utc)

    formatted: Dict[str, Dict[str, str]] = {}
    for key, label, iana in TIMEZONE_SPECS:
        tz = ZoneInfo(iana)
        local_dt = dt_utc.astimezone(tz)
        tz_abbr = local_dt.tzname() or label
        formatted[key] = {
            "label": label,
            "timezone": iana,
            "tz_abbr": tz_abbr,
            "iso": local_dt.isoformat(),
            "date": local_dt.strftime("%a, %d %b %Y"),
            "time_24h": local_dt.strftime("%H:%M"),
            "time_12h": local_dt.strftime("%I:%M %p").lstrip("0"),
            "display": f"{local_dt.strftime('%a, %d %b')} · {local_dt.strftime('%I:%M %p').lstrip('0')} {tz_abbr}",
        }
    return formatted


def _build_race_schedule_payload(
    race: Mapping[str, Any],
    race_dt_utc: datetime,
    time_confirmed: bool,
    now_utc: datetime,
    status: str,
) -> Dict[str, Any]:
    circuit = race.get("Circuit") or {}
    location = circuit.get("Location") or {}

    sessions: List[Dict[str, Any]] = []
    for j_key, display_name in _SESSION_KEYS:
        sess_obj = race.get(j_key)
        if not isinstance(sess_obj, Mapping):
            continue
        s_dt, s_confirmed = parse_jolpica_utc(sess_obj.get("date"), sess_obj.get("time"))
        if s_dt is None:
            continue
        sessions.append(
            {
                "name": display_name,
                "utc": s_dt.isoformat(),
                "time_confirmed": s_confirmed,
                "times": format_multi_timezone(s_dt),
            }
        )

    sessions.append(
        {
            "name": "Grand Prix",
            "utc": race_dt_utc.isoformat(),
            "time_confirmed": time_confirmed,
            "times": format_multi_timezone(race_dt_utc),
        }
    )

    delta_s = int((race_dt_utc - now_utc).total_seconds())
    return {
        "status": status,
        "season": int(race.get("season") or race_dt_utc.year),
        "round": int(race.get("round") or 0),
        "grand_prix": str(race.get("raceName") or "Grand Prix"),
        "circuit_name": str(circuit.get("circuitName") or ""),
        "locality": str(location.get("locality") or ""),
        "country": str(location.get("country") or ""),
        "race_utc": race_dt_utc.isoformat(),
        "time_confirmed": time_confirmed,
        "starts_in_seconds": max(0, delta_s),
        "times": format_multi_timezone(race_dt_utc),
        "sessions": sessions,
    }


def get_next_race_schedule(
    jolpica: JolpicaClient,
    *,
    season: Optional[int] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return the next upcoming F1 Grand Prix with IST, UK, and US times."""
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    base_season = int(season or now.year)

    last_seen: Optional[Tuple[Mapping[str, Any], datetime, bool]] = None
    for candidate_season in (base_season, base_season + 1):
        try:
            races = jolpica.get_schedule(candidate_season)
        except JolpicaError:
            races = []
        for race in races:
            race_dt, confirmed = parse_jolpica_utc(race.get("date"), race.get("time"))
            if race_dt is None:
                continue
            last_seen = (race, race_dt, confirmed)
            # Consider a race "live" for 3 hours after lights-out before advancing
            if race_dt + timedelta(hours=3) >= now:
                status = "live_now" if race_dt <= now else "upcoming"
                return _build_race_schedule_payload(race, race_dt, confirmed, now, status)

    if last_seen is not None:
        race, race_dt, confirmed = last_seen
        return _build_race_schedule_payload(race, race_dt, confirmed, now, "season_completed")

    raise ValueError(f"No F1 schedule available for {base_season} or {base_season + 1}.")
