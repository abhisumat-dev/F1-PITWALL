"""Build the 2018-2026 training datasets from Jolpica-F1 and OpenF1.

Two datasets are produced (both written to ``backend/data/``):

``pre_race_raw.parquet``
    One row per driver per Grand Prix (2018-2026, Jolpica): grid, qualifying
    times, constructor, finishing position, points, status and labels.

``in_race_raw.parquet``
    One row per driver per lap snapshot (2023-2026, OpenF1): position, gaps,
    pace, tyre stint, pit stops and track status at the moment the race
    leader completes each lap, labelled with whether the driver went on to
    win.  OpenF1 history starts in 2023, which bounds this dataset.

Everything downloaded is cached on disk by the API clients, so re-running is
cheap and the in-race build is incremental (already processed races are
skipped unless ``--force`` is given).

Usage (from the ``backend/`` directory)::

    python -m ml.ingest_history                    # 2018..current season
    python -m ml.ingest_history --skip-in-race     # Jolpica only (fast)
    python -m ml.ingest_history --start-season 2020 --end-season 2025
"""
from __future__ import annotations

import argparse
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import config
from clients.jolpica_client import JolpicaClient, JolpicaError, parse_lap_time
from clients.openf1_client import OpenF1Client, OpenF1Error
from ml.features import (
    IN_RACE_RAW_COLUMNS,
    RAW_PRE_RACE_COLUMNS,
    to_epoch_seconds,
    track_state_timeline,
)

LOGGER = logging.getLogger(__name__)

SNAPSHOT_COLUMNS: List[str] = list(IN_RACE_RAW_COLUMNS)


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def is_classified_status(status: Optional[str]) -> bool:
    """True when a Jolpica result status means the driver finished the race."""
    text = (status or "").strip()
    return text == "Finished" or text == "Lapped" or bool(re.match(r"^\+\d+\s+Laps?$", text))


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Pre-race dataset (Jolpica)
# --------------------------------------------------------------------------- #
def _make_row(
    base: Mapping[str, Any],
    driver: Mapping[str, Any],
    constructor: Mapping[str, Any],
    number: Any,
    result: Optional[Mapping[str, Any]],
    quali: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(base)
    family = str(driver.get("familyName") or "")
    row["driver_id"] = driver.get("driverId")
    row["driver_code"] = driver.get("code") or family[:3].upper()
    row["driver_number"] = _int_or_none(number) or _int_or_none(driver.get("permanentNumber"))
    row["driver_name"] = f"{driver.get('givenName', '')} {family}".strip()
    row["constructor_id"] = constructor.get("constructorId")
    row["constructor_name"] = constructor.get("name")

    q1 = q2 = q3 = None
    quali_position: Optional[int] = None
    if quali:
        q1 = parse_lap_time(quali.get("Q1"))
        q2 = parse_lap_time(quali.get("Q2"))
        q3 = parse_lap_time(quali.get("Q3"))
        quali_position = _int_or_none(quali.get("position"))
    times = [t for t in (q1, q2, q3) if t is not None]
    row["quali_position"] = quali_position
    row["q1_s"], row["q2_s"], row["q3_s"] = q1, q2, q3
    row["quali_best_s"] = min(times) if times else None

    if result is not None:
        status = str(result.get("status") or "")
        finish = _int_or_none(result.get("position"))
        row["grid"] = _int_or_none(result.get("grid"))
        row["finish_position"] = finish
        row["position_text"] = result.get("positionText")
        row["status"] = status
        row["points"] = _float_or_none(result.get("points"))
        row["laps_completed"] = _int_or_none(result.get("laps"))
        is_podium = 1 if finish is not None and finish <= 3 else 0
        row["won"] = 1 if finish == 1 else 0
        row["podium"] = is_podium
        row["is_podium"] = is_podium
        row["dnf"] = 0 if is_classified_status(status) else 1
        row["completed"] = True
    else:
        # Race not run yet: the starting grid is the qualifying order.
        row["grid"] = quali_position
        row["finish_position"] = None
        row["position_text"] = None
        row["status"] = None
        row["points"] = None
        row["laps_completed"] = None
        row["won"] = None
        row["podium"] = None
        row["is_podium"] = None
        row["dnf"] = None
        row["completed"] = False
    return row


def build_race_rows(
    race_meta: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    qualifying: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge one race's qualifying and result lists into driver rows."""
    circuit = race_meta.get("Circuit") or {}
    base = {
        "season": int(race_meta["season"]),
        "round": int(race_meta["round"]),
        "race_name": race_meta.get("raceName"),
        "circuit_id": circuit.get("circuitId"),
        "circuit_name": circuit.get("circuitName"),
        "race_date": race_meta.get("date"),
    }
    quali_by_driver = {
        (q.get("Driver") or {}).get("driverId"): q for q in (qualifying or [])
    }
    rows: List[Dict[str, Any]] = []
    if results:
        for result in results:
            driver = result.get("Driver") or {}
            quali = quali_by_driver.get(driver.get("driverId"))
            rows.append(
                _make_row(
                    base,
                    driver,
                    result.get("Constructor") or {},
                    result.get("number"),
                    result,
                    quali,
                )
            )
    else:
        for quali in qualifying or []:
            rows.append(
                _make_row(
                    base,
                    quali.get("Driver") or {},
                    quali.get("Constructor") or {},
                    quali.get("number"),
                    None,
                    quali,
                )
            )
    return rows


def fetch_race_rows(
    jolpica: JolpicaClient, season: int, round_number: int
) -> List[Dict[str, Any]]:
    """Driver rows for one race; works before *and* after the race is run."""
    meta = jolpica.get_race(season, round_number)
    results_races = jolpica.get_race_results(season, round_number)
    quali_races = jolpica.get_qualifying_results(season, round_number)
    if meta is None:
        meta = (results_races or quali_races or [None])[0]
    if meta is None:
        return []
    results = results_races[0].get("Results", []) if results_races else []
    qualifying = quali_races[0].get("QualifyingResults", []) if quali_races else []
    return build_race_rows(meta, results, qualifying)


def build_pre_race_dataset(
    jolpica: JolpicaClient,
    start_season: int,
    end_season: int,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Driver-race table for ``start_season..end_season`` from Jolpica."""
    rows: List[Dict[str, Any]] = []
    for season in range(int(start_season), int(end_season) + 1):
        try:
            schedule = {int(r["round"]): r for r in jolpica.get_schedule(season)}
            result_races = {int(r["round"]): r for r in jolpica.get_race_results(season)}
            quali_races = {int(r["round"]): r for r in jolpica.get_qualifying_results(season)}
        except JolpicaError as exc:
            LOGGER.warning("Skipping season %s: %s", season, exc)
            continue
        season_rows = 0
        for round_number in sorted(set(schedule) | set(result_races)):
            result_race = result_races.get(round_number)
            results = result_race.get("Results", []) if result_race else []
            if not results and not include_incomplete:
                continue
            meta = schedule.get(round_number) or result_race
            if meta is None:
                continue
            qualifying = (quali_races.get(round_number) or {}).get("QualifyingResults", [])
            built = build_race_rows(meta, results, qualifying)
            rows.extend(built)
            season_rows += len(built)
        LOGGER.info("Season %s: %s driver-race rows", season, season_rows)
    return pd.DataFrame(rows, columns=RAW_PRE_RACE_COLUMNS)


# --------------------------------------------------------------------------- #
# OpenF1 <-> Jolpica race matching
# --------------------------------------------------------------------------- #
def match_session_to_race_date(
    sessions: Sequence[Mapping[str, Any]],
    race_date: Any,
    used: Optional[Set[int]] = None,
    tolerance_days: int = 1,
) -> Optional[Mapping[str, Any]]:
    """Pick the OpenF1 race session closest to a Jolpica race date.

    A one day tolerance covers races that start after midnight UTC (for
    example Las Vegas), where Jolpica lists the local date.
    """
    target = _parse_date(race_date)
    if target is None:
        return None
    best: Optional[Mapping[str, Any]] = None
    best_gap: Optional[int] = None
    for session in sessions:
        key = _int_or_none(session.get("session_key"))
        if used is not None and key in used:
            continue
        start = _parse_date(session.get("date_start"))
        if start is None:
            continue
        gap = abs((start - target).days)
        if gap <= tolerance_days and (best_gap is None or gap < best_gap):
            best, best_gap = session, gap
    return best


# --------------------------------------------------------------------------- #
# In-race snapshots (OpenF1)
# --------------------------------------------------------------------------- #
def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _prepare_laps(
    laps: Sequence[Mapping[str, Any]], pits: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    df = pd.DataFrame(list(laps or []))
    if df.empty or not {"driver_number", "lap_number"} <= set(df.columns):
        return pd.DataFrame()
    df = df.dropna(subset=["driver_number", "lap_number"]).copy()
    df["driver_number"] = df["driver_number"].astype(int)
    df["lap_number"] = df["lap_number"].astype(int)
    df["lap_duration"] = pd.to_numeric(_column(df, "lap_duration"), errors="coerce")
    df["start_s"] = to_epoch_seconds(_column(df, "date_start"))
    df = df.sort_values(["driver_number", "lap_number"]).drop_duplicates(
        ["driver_number", "lap_number"], keep="last"
    ).reset_index(drop=True)

    next_start = df.groupby("driver_number")["start_s"].shift(-1)
    df["lap_end_s"] = (df["start_s"] + df["lap_duration"]).fillna(next_start)

    pit_laps: Set[Tuple[int, int]] = set()
    for pit in pits or []:
        driver = _int_or_none(pit.get("driver_number"))
        lap = _int_or_none(pit.get("lap_number"))
        if driver is not None and lap is not None:
            pit_laps.add((driver, lap))
    if "is_pit_out_lap" in df.columns:
        out_lap = df["is_pit_out_lap"].map(lambda v: bool(v) if v is not None and v == v else False)
    else:
        out_lap = pd.Series(False, index=df.index)
    in_or_out = [
        ((d, l) in pit_laps) or ((d, l - 1) in pit_laps)
        for d, l in zip(df["driver_number"], df["lap_number"])
    ]
    excluded = out_lap | pd.Series(in_or_out, index=df.index) | (df["lap_number"] <= 1)

    racing = df.loc[~excluded & df["lap_duration"].notna(), "lap_duration"]
    median = float(racing.median()) if len(racing) else float("nan")
    clean = df["lap_duration"].where(~excluded)
    if not np.isnan(median):
        clean = clean.where(clean <= 1.25 * median)  # drop SC / red-flag crawl laps
    df["clean_lap_s"] = clean
    df["pace_3_s"] = df.groupby("driver_number")["clean_lap_s"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    return df


def _result_lookup(session_result: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for row in session_result or []:
        number = _int_or_none(row.get("driver_number"))
        if number is None:
            continue
        lookup[number] = {
            "position": _int_or_none(row.get("position")),
            "laps": _int_or_none(row.get("number_of_laps")),
            "retired": bool(row.get("dnf") or row.get("dns") or row.get("dsq")),
        }
    return lookup


def _stint_table(stints: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    table: Dict[int, List[Dict[str, Any]]] = {}
    for stint in stints or []:
        number = _int_or_none(stint.get("driver_number"))
        if number is None:
            continue
        table.setdefault(number, []).append(
            {
                "stint_number": _int_or_none(stint.get("stint_number")),
                "lap_start": _int_or_none(stint.get("lap_start")),
                "compound": str(stint.get("compound") or "UNKNOWN").upper(),
                "age0": _int_or_none(stint.get("tyre_age_at_start")) or 0,
            }
        )
    for items in table.values():
        items.sort(key=lambda s: s["lap_start"] if s["lap_start"] is not None else 0)
    return table


def _stint_at(
    table: Mapping[int, Sequence[Mapping[str, Any]]], driver: int, completed: int
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    chosen: Optional[Mapping[str, Any]] = None
    for stint in table.get(driver, []):
        start = stint["lap_start"] if stint["lap_start"] is not None else 1
        if start <= completed:
            chosen = stint
        else:
            break
    if chosen is None:
        return None, None, None
    start = chosen["lap_start"] if chosen["lap_start"] is not None else 1
    age = chosen["age0"] + (completed - start + 1)
    return chosen["compound"], age, chosen["stint_number"]


def build_race_snapshots(
    *,
    season: int,
    round_number: int,
    session_key: int,
    drivers: Sequence[Mapping[str, Any]],
    laps: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    stints: Sequence[Mapping[str, Any]],
    pits: Sequence[Mapping[str, Any]],
    race_control: Sequence[Mapping[str, Any]],
    session_result: Sequence[Mapping[str, Any]],
    min_lap: int = config.FIRST_LIVE_SNAPSHOT_LAP,
) -> pd.DataFrame:
    """Turn raw OpenF1 endpoint rows into per-lap training snapshots.

    A snapshot is taken at the moment the race leader completes lap ``L``.  For
    every car still running, the state at that instant is read from the
    ``position`` / ``intervals`` streams, the car's own completed-lap count,
    its stint (compound/age), its pit stops so far and the track status.
    """
    empty = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    laps_df = _prepare_laps(laps, pits)
    if laps_df.empty:
        return empty

    results = _result_lookup(session_result)
    code_by_driver: Dict[int, str] = {}
    for driver in drivers or []:
        number = _int_or_none(driver.get("driver_number"))
        if number is not None:
            code_by_driver[number] = str(driver.get("name_acronym") or number)

    total_laps = int(laps_df["lap_number"].max())
    result_laps = [info["laps"] for info in results.values() if info["laps"]]
    if result_laps:
        total_laps = max(total_laps, max(result_laps))

    valid = laps_df.dropna(subset=["lap_end_s"]).sort_values(["driver_number", "lap_end_s"])
    if valid.empty:
        return empty

    winners = {number for number, info in results.items() if info["position"] == 1}
    podium_finishers = {
        number for number, info in results.items() if info["position"] is not None and info["position"] <= 3
    }
    if not winners:
        last = valid.sort_values(["lap_number", "lap_end_s"], ascending=[False, True])
        winners = {int(last.iloc[0]["driver_number"])}
    if not podium_finishers:
        last_by_driver = (
            valid.sort_values(["lap_number", "lap_end_s"], ascending=[False, True])
            .drop_duplicates("driver_number", keep="first")
        )
        podium_finishers = {int(n) for n in last_by_driver.iloc[:3]["driver_number"].tolist()}
    retired = {number for number, info in results.items() if info["retired"]}

    per_driver: Dict[int, Dict[str, np.ndarray]] = {}
    for number, group in valid.groupby("driver_number"):
        per_driver[int(number)] = {
            "ends": group["lap_end_s"].to_numpy(dtype=float),
            "laps": group["lap_number"].to_numpy(dtype=int),
            "durs": group["lap_duration"].to_numpy(dtype=float),
            "pace": group["pace_3_s"].to_numpy(dtype=float),
        }

    first_end = valid.groupby("lap_number")["lap_end_s"].min()
    clean = laps_df.dropna(subset=["clean_lap_s"]).sort_values("lap_number")
    overall_ref = float(clean["clean_lap_s"].median()) if len(clean) else float("nan")
    ref_by_lap: Dict[int, float] = {}
    for lap_no in first_end.index:
        subset = clean.loc[clean["lap_number"] <= lap_no, "clean_lap_s"]
        ref_by_lap[int(lap_no)] = float(subset.median()) if len(subset) else overall_ref

    stint_table = _stint_table(stints)
    pit_by_driver: Dict[int, np.ndarray] = {}
    for pit in pits or []:
        number = _int_or_none(pit.get("driver_number"))
        lap = _int_or_none(pit.get("lap_number"))
        if number is not None and lap is not None:
            pit_by_driver.setdefault(number, []).append(lap)  # type: ignore[arg-type]
    pit_arrays = {n: np.sort(np.asarray(v, dtype=int)) for n, v in pit_by_driver.items()}
    timeline = track_state_timeline(race_control)
    timeline_times = np.asarray([t for t, _ in timeline], dtype=float)
    timeline_states = np.asarray([s for _, s in timeline], dtype=int)

    records: List[Tuple[Any, ...]] = []
    for lap_no, t_s in first_end.items():
        lap_no = int(lap_no)
        if lap_no < min_lap or lap_no >= total_laps:
            continue
        if timeline_times.size:
            idx = int(np.searchsorted(timeline_times, t_s, side="right")) - 1
            state = int(timeline_states[idx]) if idx >= 0 else 0
        else:
            state = 0
        ref = ref_by_lap.get(lap_no, overall_ref)
        for number, info in per_driver.items():
            idx = int(np.searchsorted(info["ends"], t_s, side="right"))
            if idx == 0:
                continue
            if number in retired and t_s > info["ends"][-1]:
                continue
            completed = int(info["laps"][idx - 1])
            compound, age, stint_no = _stint_at(stint_table, number, completed)
            pit_count = (
                int(np.searchsorted(pit_arrays[number], completed, side="right"))
                if number in pit_arrays
                else 0
            )
            records.append(
                (
                    lap_no,
                    float(t_s),
                    number,
                    completed,
                    float(info["durs"][idx - 1]),
                    float(info["pace"][idx - 1]),
                    ref,
                    compound,
                    age,
                    stint_no,
                    pit_count,
                    state,
                )
            )
    if not records:
        return empty

    request = pd.DataFrame(
        records,
        columns=[
            "lap",
            "t_s",
            "driver_number",
            "laps_completed",
            "last_lap_s",
            "pace_3_s",
            "ref_lap_s",
            "compound",
            "tyre_age_laps",
            "stint_number",
            "pit_stops_done",
            "track_state",
        ],
    )
    request["driver_number"] = request["driver_number"].astype("int64")
    request = request.sort_values("t_s", kind="mergesort").reset_index(drop=True)

    pos_df = pd.DataFrame(list(positions or []))
    if not pos_df.empty and {"driver_number", "position", "date"} <= set(pos_df.columns):
        pos_df = pos_df.dropna(subset=["driver_number", "position"]).copy()
        pos_df["driver_number"] = pos_df["driver_number"].astype("int64")
        pos_df["t_s"] = to_epoch_seconds(pos_df["date"])
        pos_df["position"] = pd.to_numeric(pos_df["position"], errors="coerce")
        pos_df = pos_df.dropna(subset=["t_s", "position"])[["t_s", "driver_number", "position"]]
        pos_df = pos_df.sort_values("t_s", kind="mergesort")
        request = pd.merge_asof(request, pos_df, on="t_s", by="driver_number", direction="backward")
    else:
        request["position"] = np.nan

    int_df = pd.DataFrame(list(intervals or []))
    if not int_df.empty and {"driver_number", "date"} <= set(int_df.columns):
        int_df = int_df.dropna(subset=["driver_number"]).copy()
        int_df["driver_number"] = int_df["driver_number"].astype("int64")
        int_df["t_s"] = to_epoch_seconds(int_df["date"])
        int_df["gap_to_leader_s"] = pd.to_numeric(_column(int_df, "gap_to_leader"), errors="coerce")
        int_df["interval_s"] = pd.to_numeric(_column(int_df, "interval"), errors="coerce")
        int_df = int_df.dropna(subset=["t_s"])[["t_s", "driver_number", "gap_to_leader_s", "interval_s"]]
        int_df = int_df.sort_values("t_s", kind="mergesort")
        request = pd.merge_asof(request, int_df, on="t_s", by="driver_number", direction="backward")
    else:
        request["gap_to_leader_s"] = np.nan
        request["interval_s"] = np.nan

    request = request.dropna(subset=["position"]).copy()
    if request.empty:
        return empty
    request["position"] = request["position"].astype(int)
    # When the leader completes lap L, cars on the lead lap have finished L-1 or L
    # laps; only cars with fewer than L-1 completed laps are genuinely lapped.
    request["laps_down"] = (request["lap"] - request["laps_completed"] - 1).clip(lower=0).astype(int)
    leader = request["position"] == 1
    request.loc[leader, "gap_to_leader_s"] = 0.0
    request.loc[leader, "interval_s"] = 0.0

    request["season"] = int(season)
    request["round"] = int(round_number)
    request["session_key"] = int(session_key)
    request["total_laps"] = int(total_laps)
    request["driver_code"] = request["driver_number"].map(lambda n: code_by_driver.get(int(n), str(n)))
    request["final_position"] = request["driver_number"].map(
        lambda n: (results.get(int(n)) or {}).get("position")
    )
    request["won"] = request["driver_number"].isin(winners).astype(int)
    podium_col = np.where(
        request["final_position"].notna(),
        (pd.to_numeric(request["final_position"], errors="coerce") <= 3).astype(int),
        request["driver_number"].isin(podium_finishers).astype(int),
    )
    request["podium"] = podium_col
    request["is_podium"] = podium_col
    return (
        request[SNAPSHOT_COLUMNS]
        .sort_values(["lap", "position"], kind="mergesort")
        .reset_index(drop=True)
    )


def _fetch_race_snapshots(
    openf1: OpenF1Client, season: int, round_number: int, session_key: int
) -> pd.DataFrame:
    bundle = {
        name: openf1.get(name, {"session_key": session_key})
        for name in (
            "drivers",
            "laps",
            "position",
            "intervals",
            "stints",
            "pit",
            "race_control",
            "session_result",
        )
    }
    return build_race_snapshots(
        season=season,
        round_number=round_number,
        session_key=session_key,
        drivers=bundle["drivers"],
        laps=bundle["laps"],
        positions=bundle["position"],
        intervals=bundle["intervals"],
        stints=bundle["stints"],
        pits=bundle["pit"],
        race_control=bundle["race_control"],
        session_result=bundle["session_result"],
    )


def build_in_race_dataset(
    openf1: OpenF1Client,
    pre_race_raw: pd.DataFrame,
    start_season: int,
    end_season: int,
    existing: Optional[pd.DataFrame] = None,
    force: bool = False,
    save_each_season: bool = True,
) -> pd.DataFrame:
    """Lap-snapshot table for every completed race OpenF1 covers."""
    frames: List[pd.DataFrame] = []
    done: Set[Tuple[int, int]] = set()
    if existing is not None and not existing.empty and not force:
        frames.append(existing)
        done = {(int(s), int(r)) for s, r in zip(existing["season"], existing["round"])}

    completed = pre_race_raw[pre_race_raw["completed"].astype(bool)]
    races = completed.drop_duplicates(["season", "round"])[
        ["season", "round", "race_date", "race_name"]
    ]
    for season in range(int(start_season), int(end_season) + 1):
        season_races = races[races["season"] == season]
        if season_races.empty:
            continue
        try:
            meeting_keys = {int(m["meeting_key"]) for m in openf1.list_races(season)}
            sessions = [
                s
                for s in openf1.get_sessions(year=season, session_name="Race")
                if _int_or_none(s.get("meeting_key")) in meeting_keys
            ]
        except (OpenF1Error, KeyError) as exc:
            LOGGER.warning("OpenF1 season %s unavailable: %s", season, exc)
            continue

        used: Set[int] = set()
        season_new = 0
        for _, race in season_races.sort_values("round").iterrows():
            key = (int(race["season"]), int(race["round"]))
            session = match_session_to_race_date(sessions, race["race_date"], used)
            if session is None:
                LOGGER.warning("No OpenF1 race session for %s round %s (%s)", key[0], key[1], race["race_name"])
                continue
            session_key = int(session["session_key"])
            used.add(session_key)
            if key in done:
                continue
            try:
                snapshots = _fetch_race_snapshots(openf1, key[0], key[1], session_key)
            except OpenF1Error as exc:
                LOGGER.warning("Skipping %s round %s: %s", key[0], key[1], exc)
                continue
            if snapshots.empty:
                LOGGER.warning("No usable lap data for %s round %s (session %s)", key[0], key[1], session_key)
                continue
            frames.append(snapshots)
            season_new += 1
            LOGGER.info("%s R%02d %s: %s snapshots", key[0], key[1], race["race_name"], len(snapshots))
        if save_each_season and frames and season_new:
            save_dataset(pd.concat(frames, ignore_index=True), config.IN_RACE_RAW_PATH)

    if not frames:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Persistence + CLI
# --------------------------------------------------------------------------- #
def save_dataset(frame: pd.DataFrame, path: Any) -> None:
    """Atomically write a dataset as parquet."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(target)


def load_pre_race_raw(path: Any = None) -> pd.DataFrame:
    target = Path(path or config.PRE_RACE_RAW_PATH)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run `python -m ml.ingest_history` from backend/ first."
        )
    return pd.read_parquet(target)


def load_in_race_raw(path: Any = None) -> pd.DataFrame:
    target = Path(path or config.IN_RACE_RAW_PATH)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run `python -m ml.ingest_history` from backend/ first."
        )
    return pd.read_parquet(target)


def ingest(
    start_season: int,
    end_season: int,
    in_race_start: int,
    skip_in_race: bool = False,
    force: bool = False,
    jolpica: Optional[JolpicaClient] = None,
    openf1: Optional[OpenF1Client] = None,
) -> Dict[str, int]:
    """Run the full ingestion and return row counts."""
    config.ensure_directories()
    jolpica = jolpica or JolpicaClient()
    pre = build_pre_race_dataset(jolpica, start_season, end_season)
    save_dataset(pre, config.PRE_RACE_RAW_PATH)
    summary = {
        "pre_race_rows": int(len(pre)),
        "pre_race_races": int(pre.drop_duplicates(["season", "round"]).shape[0]) if len(pre) else 0,
        "in_race_rows": 0,
    }
    LOGGER.info("Pre-race dataset: %s rows over %s races", summary["pre_race_rows"], summary["pre_race_races"])
    if skip_in_race or pre.empty:
        return summary

    openf1 = openf1 or OpenF1Client()
    existing = None
    if config.IN_RACE_RAW_PATH.exists() and not force:
        existing = pd.read_parquet(config.IN_RACE_RAW_PATH)
    in_race = build_in_race_dataset(
        openf1,
        pre,
        max(start_season, in_race_start),
        end_season,
        existing=existing,
        force=force,
    )
    if not in_race.empty:
        save_dataset(in_race, config.IN_RACE_RAW_PATH)
    summary["in_race_rows"] = int(len(in_race))
    LOGGER.info("In-race dataset: %s snapshot rows", summary["in_race_rows"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the F1 Pit-Wall training datasets.")
    parser.add_argument("--start-season", type=int, default=config.TRAIN_START_SEASON)
    parser.add_argument("--end-season", type=int, default=config.current_season())
    parser.add_argument("--in-race-start", type=int, default=config.IN_RACE_START_SEASON)
    parser.add_argument("--skip-in-race", action="store_true", help="Only build the Jolpica pre-race dataset.")
    parser.add_argument("--force", action="store_true", help="Rebuild in-race snapshots for every race.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    config.configure_logging(args.verbose)
    summary = ingest(
        args.start_season,
        args.end_season,
        args.in_race_start,
        skip_in_race=args.skip_in_race,
        force=args.force,
    )
    LOGGER.info("Done: %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
