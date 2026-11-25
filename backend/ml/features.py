"""Feature engineering for the dual-stage race-winner models.

Two completely separate feature sets are built here:

* **Pre-Race (post-qualifying)** - one row per driver per Grand Prix, using
  only information that is known once qualifying is over (grid, quali pace,
  rolling driver/team form, circuit history, championship state).  Every
  rolling statistic is shifted by one race so a row never sees its own race.
* **In-Race (live)** - one row per driver per lap snapshot, describing the
  state of the race when the leader completes a lap (position, gaps, pace,
  tyres, pit stops, track status) plus a few leakage-free pre-race priors.

The same functions are used for training and for live inference so the two
can never drift apart.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
RAW_PRE_RACE_COLUMNS: List[str] = [
    "season",
    "round",
    "race_name",
    "circuit_id",
    "circuit_name",
    "race_date",
    "driver_id",
    "driver_code",
    "driver_number",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "grid",
    "quali_position",
    "q1_s",
    "q2_s",
    "q3_s",
    "quali_best_s",
    "finish_position",
    "position_text",
    "status",
    "points",
    "laps_completed",
    "won",
    "podium",
    "is_podium",
    "dnf",
    "completed",
]

IN_RACE_RAW_COLUMNS: List[str] = [
    "season",
    "round",
    "session_key",
    "lap",
    "total_laps",
    "driver_number",
    "driver_code",
    "position",
    "laps_completed",
    "laps_down",
    "gap_to_leader_s",
    "interval_s",
    "last_lap_s",
    "pace_3_s",
    "ref_lap_s",
    "compound",
    "tyre_age_laps",
    "stint_number",
    "pit_stops_done",
    "track_state",
    "final_position",
    "won",
    "podium",
    "is_podium",
]

PRE_RACE_FEATURES: List[str] = [
    "grid_position",
    "pit_lane_start",
    "quali_pos",
    "grid_penalty_places",
    "quali_gap_to_pole_pct",
    "is_pole",
    "front_row",
    "field_size",
    "driver_avg_finish_5",
    "driver_avg_grid_5",
    "driver_win_rate_10",
    "driver_podium_rate_10",
    "driver_dnf_rate_10",
    "driver_points_per_race_5",
    "driver_career_wins",
    "driver_career_starts",
    "driver_season_points_before",
    "driver_points_share_before",
    "driver_season_rank_before",
    "team_points_avg_5",
    "team_win_rate_10",
    "team_best_finish_avg_5",
    "team_quali_gap_avg_5",
    "team_quali_improvement_slope",
    "team_points_momentum",
    "team_finish_improvement",
    "current_quali_vs_team_baseline",
    "driver_finish_momentum",
    "driver_circuit_avg_finish",
    "driver_circuit_wins",
    "driver_circuit_starts",
    "driver_form_rank",
    "team_form_rank",
    "grid_vs_team_form_rank",
    "grid_vs_driver_form_rank",
]

#: Leakage-free pre-race context carried into the in-race model.
PRIOR_FEATURES: List[str] = [
    "grid_position",
    "driver_win_rate_10",
    "driver_podium_rate_10",
    "driver_avg_finish_5",
    "team_points_avg_5",
    "team_win_rate_10",
    "team_points_momentum",
    "team_quali_improvement_slope",
]

IN_RACE_FEATURES: List[str] = [
    "lap_fraction",
    "laps_remaining",
    "position",
    "log_position",
    "laps_down",
    "gap_to_leader_s",
    "gap_to_leader_laps",
    "gap_to_leader_per_remaining",
    "interval_ahead_s",
    "interval_ahead_laps",
    "gap_behind_s",
    "pace_delta_field_s",
    "pace_delta_leader_s",
    "tyre_age_laps",
    "stint_number",
    "pit_stops_done",
    "pit_stops_vs_field",
    "compound_soft",
    "compound_medium",
    "compound_hard",
    "compound_wet",
    "track_state",
    "is_sc_or_vsc",
    "field_remaining",
    "cars_on_lead_lap",
    "grid_position",
    "positions_gained",
    "driver_win_rate_10",
    "driver_podium_rate_10",
    "driver_avg_finish_5",
    "team_points_avg_5",
    "team_win_rate_10",
    "team_points_momentum",
    "team_quali_improvement_slope",
]

#: Numeric track-state codes used by the in-race model.
TRACK_STATE_CODES: Dict[str, int] = {"GREEN": 0, "VSC": 1, "SC": 2, "RED": 3}
TRACK_STATE_NAMES: Dict[int, str] = {v: k for k, v in TRACK_STATE_CODES.items()}

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def to_epoch_seconds(values: Any) -> pd.Series:
    """Convert ISO-8601 strings / datetimes to float seconds since the epoch."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    series = series.astype(object)
    parsed = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    return (parsed - _EPOCH).dt.total_seconds()


def epoch_seconds(value: Any) -> Optional[float]:
    """Single-value variant of :func:`to_epoch_seconds` (``None`` on failure)."""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return float(stamp.timestamp())


def normalize_to_percent(values: Iterable[float], decimals: int = 2) -> List[float]:
    """Scale non-negative scores so they sum to exactly 100 percent.

    Uses the largest-remainder method on integer units of ``10**-decimals`` so
    the rounded numbers still add up to 100.  An all-zero input becomes a
    uniform distribution.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return []
    array = np.where(np.isfinite(array) & (array > 0), array, 0.0)
    total = float(array.sum())
    if total <= 0:
        array = np.ones_like(array)
        total = float(array.sum())
    scale = 10 ** decimals
    target = 100 * scale
    exact = array / total * target
    floors = np.floor(exact).astype(np.int64)
    remainder = int(target - floors.sum())
    if remainder > 0:
        order = np.argsort(-(exact - floors), kind="stable")
        floors[order[:remainder]] += 1
    return [round(float(units) / scale, decimals) for units in floors]


def normalize_podium_to_percent(
    values: Iterable[float],
    active_mask: Optional[Iterable[bool]] = None,
    decimals: int = 2,
    target_sum: float = 300.0,
    max_per_driver: float = 100.0,
) -> List[float]:
    """Iterative clipping-and-normalization for Top 3 (Podium) probabilities.

    * Inactive / DNF / OUT drivers (``active_mask == False``) receive ``0.0``.
    * Active drivers are scaled and clipped iteratively so each stays within
      ``[0.0, 100.0]`` and the sum across active drivers equals ``300.0%``
      (or ``100.0 * n_active`` when fewer than 3 active drivers exist).
    """
    raw = np.asarray(list(values), dtype=float)
    n = raw.size
    if n == 0:
        return []
    if active_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(list(active_mask), dtype=bool)
        if active.size != n:
            raise ValueError("active_mask length must match values length")

    scores = np.where(np.isfinite(raw) & (raw > 0) & active, raw, 0.0)
    n_active = int(active.sum())
    if n_active == 0:
        return [0.0] * n

    effective_target = min(float(target_sum), float(n_active) * float(max_per_driver))
    if scores[active].sum() <= 0:
        scores[active] = 1.0

    # Iterative water-filling / clipping to [0, max_per_driver] summing to effective_target
    probs = np.zeros(n, dtype=float)
    free = active.copy()
    remaining_target = effective_target

    for _ in range(n_active + 2):
        free_sum = float(scores[free].sum())
        if free_sum <= 0:
            n_free = int(free.sum())
            if n_free > 0:
                probs[free] = min(max_per_driver, remaining_target / n_free)
            break
        scaled = scores[free] / free_sum * remaining_target
        over = scaled > max_per_driver + 1e-12
        if not np.any(over):
            probs[free] = scaled
            break
        free_indices = np.flatnonzero(free)
        capped_indices = free_indices[over]
        probs[capped_indices] = max_per_driver
        free[capped_indices] = False
        remaining_target = max(0.0, effective_target - float(probs[~free & active].sum()))
        if not np.any(free) or remaining_target <= 0:
            break

    probs = np.clip(probs, 0.0, max_per_driver)
    probs[~active] = 0.0

    # Largest-remainder rounding while respecting [0, max_per_driver * scale]
    scale = 10 ** decimals
    max_units = int(round(max_per_driver * scale))
    target_units = int(round(effective_target * scale))
    exact_units = probs * scale
    floors = np.floor(exact_units).astype(np.int64)
    floors = np.clip(floors, 0, max_units)
    floors[~active] = 0

    diff = int(target_units - floors.sum())
    frac = exact_units - floors
    if diff > 0:
        eligible = np.flatnonzero(active & (floors < max_units))
        order = eligible[np.argsort(-frac[eligible], kind="stable")]
        for idx in order:
            if diff <= 0:
                break
            add = min(diff, int(max_units - floors[idx]))
            if add > 0:
                floors[idx] += 1
                diff -= 1
        # If still remaining due to multiple units gap
        while diff > 0:
            eligible = np.flatnonzero(active & (floors < max_units))
            if eligible.size == 0:
                break
            for idx in eligible:
                if diff <= 0:
                    break
                floors[idx] += 1
                diff -= 1
    elif diff < 0:
        need = -diff
        eligible = np.flatnonzero(active & (floors > 0))
        order = eligible[np.argsort(frac[eligible], kind="stable")]
        for idx in order:
            if need <= 0:
                break
            floors[idx] -= 1
            need -= 1

    return [round(float(u) / scale, decimals) for u in floors]


def normalize_within_groups(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    scores: Sequence[float],
    target_sum: float = 3.0,
    max_prob: float = 1.0,
) -> np.ndarray:
    """Normalise each group's scores to ``target_sum`` (3.0 for podium, 1.0 for win) with clipping at ``max_prob``."""
    data = frame[list(group_cols)].copy().reset_index(drop=True)
    raw = np.clip(np.asarray(scores, dtype=float), 1e-9, None)
    out = np.zeros_like(raw, dtype=float)
    for _, group_idx in data.groupby(list(group_cols), sort=False).groups.items():
        idx = np.asarray(list(group_idx), dtype=int)
        g_scores = raw[idx].copy()
        n = len(idx)
        eff_target = min(float(target_sum), float(n) * float(max_prob))
        probs = np.zeros(n, dtype=float)
        free = np.ones(n, dtype=bool)
        rem = eff_target
        for _ in range(n + 1):
            s = float(g_scores[free].sum())
            if s <= 0:
                if free.any():
                    probs[free] = min(max_prob, rem / int(free.sum()))
                break
            scaled = g_scores[free] / s * rem
            over = scaled > max_prob + 1e-12
            if not np.any(over):
                probs[free] = scaled
                break
            f_idx = np.flatnonzero(free)
            c_idx = f_idx[over]
            probs[c_idx] = max_prob
            free[c_idx] = False
            rem = max(0.0, eff_target - float(probs[~free].sum()))
            if not free.any() or rem <= 0:
                break
        out[idx] = np.clip(probs, 1e-6, max_prob - 1e-6 if max_prob >= 1.0 else max_prob)
    return out


def recency_weights(seasons: Sequence[float], half_life: float) -> np.ndarray:
    """Exponential recency weights: a season ``half_life`` years old counts half."""
    values = np.asarray(seasons, dtype=float)
    if values.size == 0 or half_life <= 0:
        return np.ones_like(values)
    return 0.5 ** ((values.max() - values) / half_life)


def _prior_rolling_mean(
    series: pd.Series, groups: Any, window: int, min_periods: int = 1
) -> pd.Series:
    """Rolling mean over each group's *previous* ``window`` rows."""
    return series.groupby(groups, sort=False).transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
    )


def _prior_cumsum(series: pd.Series, groups: Any) -> pd.Series:
    """Cumulative sum over each group's previous rows (excludes the current)."""
    return series.fillna(0.0).groupby(groups, sort=False).transform(
        lambda s: s.shift(1, fill_value=0.0).cumsum()
    )


# --------------------------------------------------------------------------- #
# Track status from race-control messages (shared by ingestion and live)
# --------------------------------------------------------------------------- #
def _next_track_state(state: int, message: Mapping[str, Any]) -> int:
    text = str(message.get("message") or "").upper()
    flag = str(message.get("flag") or "").upper()
    scope = str(message.get("scope") or "").upper()
    if "VIRTUAL SAFETY CAR DEPLOYED" in text:
        return TRACK_STATE_CODES["VSC"]
    if "VIRTUAL SAFETY CAR ENDING" in text:
        return TRACK_STATE_CODES["GREEN"]
    if "SAFETY CAR DEPLOYED" in text and "VIRTUAL" not in text:
        return TRACK_STATE_CODES["SC"]
    if "SAFETY CAR IN THIS LAP" in text:
        return TRACK_STATE_CODES["GREEN"]
    if flag == "RED" and scope in {"", "TRACK"}:
        return TRACK_STATE_CODES["RED"]
    if state == TRACK_STATE_CODES["RED"] and (
        "GREEN LIGHT" in text or "RACE WILL RESUME" in text or flag in {"GREEN", "CLEAR"}
    ):
        return TRACK_STATE_CODES["GREEN"]
    return state


def track_state_timeline(messages: Iterable[Mapping[str, Any]]) -> List[Tuple[float, int]]:
    """Chronological ``(epoch_seconds, state)`` change points.

    State codes: 0 green, 1 virtual safety car, 2 safety car, 3 red flag.
    Messages use OpenF1 ``race_control`` keys (``date``, ``message``, ``flag``,
    ``scope``); the live SignalR provider normalises to the same keys.
    """
    events: List[Tuple[float, Mapping[str, Any]]] = []
    for message in messages or []:
        stamp = epoch_seconds(message.get("date"))
        if stamp is not None:
            events.append((stamp, message))
    events.sort(key=lambda item: item[0])
    state = TRACK_STATE_CODES["GREEN"]
    timeline: List[Tuple[float, int]] = []
    for stamp, message in events:
        new_state = _next_track_state(state, message)
        if new_state != state:
            state = new_state
            timeline.append((stamp, state))
    return timeline


def track_state_at(timeline: Sequence[Tuple[float, int]], t_s: float) -> int:
    """State in force at time ``t_s`` given a :func:`track_state_timeline`."""
    state = TRACK_STATE_CODES["GREEN"]
    for stamp, value in timeline:
        if stamp <= t_s:
            state = value
        else:
            break
    return state


def current_track_state(messages: Iterable[Mapping[str, Any]]) -> int:
    """Latest state after replaying every message (used by the live feed)."""
    timeline = track_state_timeline(messages)
    return timeline[-1][1] if timeline else TRACK_STATE_CODES["GREEN"]


# --------------------------------------------------------------------------- #
# Pre-race (post-qualifying) features
# --------------------------------------------------------------------------- #
def build_pre_race_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add engineered pre-race features to the raw driver-race table.

    ``raw`` may contain not-yet-run races (``completed == False``); those rows
    receive features computed from history only, which is exactly what is
    needed for post-qualifying inference.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=RAW_PRE_RACE_COLUMNS + PRE_RACE_FEATURES)

    df = raw.copy()
    for column in RAW_PRE_RACE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df["season"] = pd.to_numeric(df["season"]).astype(int)
    df["round"] = pd.to_numeric(df["round"]).astype(int)
    df["completed"] = df["completed"].astype("boolean").fillna(False).astype(bool)
    df = df.sort_values(
        ["season", "round", "quali_position", "driver_id"], kind="mergesort"
    ).reset_index(drop=True)

    completed = df["completed"]
    race_keys = [df["season"], df["round"]]
    race_group = df.groupby(["season", "round"], sort=False)

    grid = pd.to_numeric(df["grid"], errors="coerce")
    quali_pos_raw = pd.to_numeric(df["quali_position"], errors="coerce")
    best_time = pd.to_numeric(df["quali_best_s"], errors="coerce")
    finish = pd.to_numeric(df["finish_position"], errors="coerce").where(completed)
    points = pd.to_numeric(df["points"], errors="coerce").where(completed)
    won = pd.to_numeric(df["won"], errors="coerce").where(completed)
    podium = pd.to_numeric(df["podium"], errors="coerce").where(completed)
    if "is_podium" in df.columns and df["is_podium"].notna().any():
        podium = podium.fillna(pd.to_numeric(df["is_podium"], errors="coerce").where(completed))
    podium = podium.fillna((finish <= 3).astype(float).where(finish.notna()))
    df["podium"] = podium
    df["is_podium"] = podium
    dnf = pd.to_numeric(df["dnf"], errors="coerce").where(completed)

    field_size = race_group["driver_id"].transform("count").astype(float)

    # Grid / qualifying -------------------------------------------------------
    pit_lane = grid == 0
    grid_pos = grid.where(grid > 0)
    grid_pos = grid_pos.fillna(quali_pos_raw.where(~pit_lane))
    grid_pos = grid_pos.where(~pit_lane, field_size + 1)
    grid_pos = grid_pos.fillna(field_size + 1)
    quali_pos = quali_pos_raw.fillna(grid_pos)

    pole_time = best_time.groupby(race_keys, sort=False).transform("min")
    quali_gap = (best_time / pole_time - 1.0) * 100.0

    df["grid_position"] = grid_pos
    df["pit_lane_start"] = pit_lane.astype(float)
    df["quali_pos"] = quali_pos
    df["grid_penalty_places"] = grid_pos - quali_pos
    df["quali_gap_to_pole_pct"] = quali_gap
    df["is_pole"] = (grid_pos == 1).astype(float)
    df["front_row"] = (grid_pos <= 2).astype(float)
    df["field_size"] = field_size

    # Helper columns (history only) --------------------------------------------
    df["_finish"] = finish
    df["_points"] = points
    df["_won"] = won
    df["_grid_hist"] = grid_pos.where(completed)
    df["_started"] = completed.astype(float)

    # Driver form ---------------------------------------------------------------
    by_driver = df["driver_id"]
    df["driver_avg_finish_5"] = _prior_rolling_mean(df["_finish"], by_driver, 5)
    drv_finish_3 = _prior_rolling_mean(df["_finish"], by_driver, 3)
    drv_finish_8 = _prior_rolling_mean(df["_finish"], by_driver, 8)
    df["driver_finish_momentum"] = (drv_finish_8 - drv_finish_3).fillna(0.0)
    df["driver_avg_grid_5"] = _prior_rolling_mean(df["_grid_hist"], by_driver, 5)
    df["driver_win_rate_10"] = _prior_rolling_mean(won, by_driver, 10)
    df["driver_podium_rate_10"] = _prior_rolling_mean(podium, by_driver, 10)
    df["driver_dnf_rate_10"] = _prior_rolling_mean(dnf, by_driver, 10)
    df["driver_points_per_race_5"] = _prior_rolling_mean(df["_points"], by_driver, 5)
    df["driver_career_wins"] = _prior_cumsum(won, by_driver)
    df["driver_career_starts"] = _prior_cumsum(df["_started"], by_driver)

    # Championship state before the race -------------------------------------------
    season_points = _prior_cumsum(df["_points"], [df["season"], df["driver_id"]])
    df["driver_season_points_before"] = season_points
    max_points = season_points.groupby(race_keys, sort=False).transform("max")
    df["driver_points_share_before"] = np.where(
        max_points > 0, season_points / max_points.where(max_points > 0, 1.0), 0.0
    )
    df["driver_season_rank_before"] = season_points.groupby(race_keys, sort=False).rank(
        method="min", ascending=False
    )

    # Circuit history ------------------------------------------------------------------
    by_circuit = [df["driver_id"], df["circuit_id"]]
    df["driver_circuit_avg_finish"] = df["_finish"].groupby(by_circuit, sort=False).transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["driver_circuit_wins"] = _prior_cumsum(won, by_circuit)
    df["driver_circuit_starts"] = _prior_cumsum(df["_started"], by_circuit)

    # Team (constructor) form ------------------------------------------------------------
    df["_race_index"] = race_group.ngroup()
    df["_quali_gap"] = quali_gap
    team = (
        df.groupby(["constructor_id", "season", "round"], sort=False)
        .agg(
            race_index=("_race_index", "first"),
            team_points=("_points", lambda s: s.sum(min_count=1)),
            team_wins=("_won", lambda s: s.sum(min_count=1)),
            team_best_finish=("_finish", "min"),
            team_gap=("_quali_gap", "mean"),
        )
        .reset_index()
        .sort_values(["constructor_id", "race_index"], kind="mergesort")
    )
    team_group = team.groupby("constructor_id", sort=False)
    team["team_points_avg_5"] = team_group["team_points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    team_pts_3 = team_group["team_points"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    team_pts_8 = team_group["team_points"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    team["team_points_momentum"] = (team_pts_3 - team_pts_8).fillna(0.0)

    team["team_win_rate_10"] = team_group["team_wins"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    team["team_best_finish_avg_5"] = team_group["team_best_finish"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    team_fin_3 = team_group["team_best_finish"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    team_fin_8 = team_group["team_best_finish"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    team["team_finish_improvement"] = (team_fin_8 - team_fin_3).fillna(0.0)

    team["team_quali_gap_avg_5"] = team_group["team_gap"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    team_gap_3 = team_group["team_gap"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    team_gap_8 = team_group["team_gap"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    team["team_quali_improvement_slope"] = (team_gap_8 - team_gap_3).fillna(0.0)

    df = df.merge(
        team[
            [
                "constructor_id",
                "season",
                "round",
                "team_points_avg_5",
                "team_win_rate_10",
                "team_best_finish_avg_5",
                "team_quali_gap_avg_5",
                "team_quali_improvement_slope",
                "team_points_momentum",
                "team_finish_improvement",
            ]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
    )
    df["current_quali_vs_team_baseline"] = (
        df["team_quali_gap_avg_5"] - df["quali_gap_to_pole_pct"]
    ).fillna(0.0)

    # Within-race relative strength ------------------------------------------------------------
    race_group = df.groupby(["season", "round"], sort=False)
    df["driver_form_rank"] = race_group["driver_avg_finish_5"].rank(method="min", ascending=True)
    df["team_form_rank"] = race_group["team_points_avg_5"].rank(method="min", ascending=False)
    df["grid_vs_team_form_rank"] = df["grid_position"] - df["team_form_rank"]
    df["grid_vs_driver_form_rank"] = df["grid_position"] - df["driver_form_rank"]

    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def make_pre_race_x(features: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix for the pre-race model."""
    return (
        features.reindex(columns=PRE_RACE_FEATURES)
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# In-race (live) features
# --------------------------------------------------------------------------- #
_IN_RACE_NUMERIC_INPUTS = [
    "season",
    "round",
    "lap",
    "total_laps",
    "driver_number",
    "position",
    "laps_completed",
    "laps_down",
    "gap_to_leader_s",
    "interval_s",
    "last_lap_s",
    "pace_3_s",
    "ref_lap_s",
    "tyre_age_laps",
    "stint_number",
    "pit_stops_done",
    "track_state",
]


def _attach_prior(df: pd.DataFrame, prior: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Join leakage-free pre-race features onto in-race rows."""
    if prior is None or prior.empty:
        for column in PRIOR_FEATURES:
            df[column] = np.nan
        return df

    keep = ["season", "round", "driver_number", "driver_code"] + PRIOR_FEATURES
    table = prior.reindex(columns=keep).copy()
    table["season"] = pd.to_numeric(table["season"], errors="coerce")
    table["round"] = pd.to_numeric(table["round"], errors="coerce")
    table["driver_number"] = pd.to_numeric(table["driver_number"], errors="coerce")

    by_number = table.dropna(subset=["season", "round", "driver_number"]).copy()
    by_number[["season", "round", "driver_number"]] = by_number[
        ["season", "round", "driver_number"]
    ].astype(int)
    by_number = by_number.drop_duplicates(["season", "round", "driver_number"]).drop(
        columns=["driver_code"]
    )
    merged = df.merge(by_number, on=["season", "round", "driver_number"], how="left")

    if "driver_code" in df.columns:
        missing = merged["grid_position"].isna()
        if missing.any():
            by_code = table.dropna(subset=["season", "round", "driver_code"]).copy()
            by_code[["season", "round"]] = by_code[["season", "round"]].astype(int)
            by_code = by_code.drop_duplicates(["season", "round", "driver_code"]).drop(
                columns=["driver_number"]
            )
            fallback = df.loc[missing, ["season", "round", "driver_code"]].merge(
                by_code, on=["season", "round", "driver_code"], how="left"
            )
            for column in PRIOR_FEATURES:
                merged.loc[missing, column] = fallback[column].to_numpy()
    return merged


def build_in_race_features(
    snapshots: pd.DataFrame, prior: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Engineer the live in-race feature set from lap snapshots.

    ``snapshots`` needs the columns in :data:`IN_RACE_RAW_COLUMNS` (``won`` and
    ``final_position`` are only needed for training).  ``prior`` is the
    pre-race feature table (``build_pre_race_features`` output or the race
    context used by the predictor).
    """
    df = snapshots.copy().reset_index(drop=True)
    for column in _IN_RACE_NUMERIC_INPUTS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "compound" not in df.columns:
        df["compound"] = None
    if "driver_code" not in df.columns:
        df["driver_code"] = None

    valid = df["season"].notna() & df["round"].notna() & df["driver_number"].notna()
    df = df.loc[valid].reset_index(drop=True)
    df[["season", "round", "driver_number"]] = df[["season", "round", "driver_number"]].astype(int)
    df = _attach_prior(df, prior)

    group = ["season", "round", "lap"]
    ref = df["ref_lap_s"].where(df["ref_lap_s"] > 0)
    total_laps = df["total_laps"].where(df["total_laps"] > 0)
    laps_down = df["laps_down"].fillna(0.0)
    position = df["position"]
    is_leader = position == 1

    df["lap_fraction"] = df["lap"] / total_laps
    df["laps_remaining"] = (total_laps - df["lap"]).clip(lower=0)
    df["log_position"] = np.log(position.where(position > 0))
    df["laps_down"] = laps_down

    gap = df["gap_to_leader_s"]
    lapped_without_gap = gap.isna() & (laps_down > 0)
    gap = gap.where(~lapped_without_gap, laps_down * ref)
    gap = gap.where(~is_leader, 0.0)
    df["gap_to_leader_s"] = gap
    df["gap_to_leader_laps"] = gap / ref
    df["gap_to_leader_per_remaining"] = gap / ((df["laps_remaining"] + 1.0) * ref)

    interval = df["interval_s"].where(~is_leader, 0.0)
    df["interval_ahead_s"] = interval
    df["interval_ahead_laps"] = interval / ref

    df = df.sort_values(group + ["position"], kind="mergesort").reset_index(drop=True)
    df["gap_behind_s"] = df.groupby(group, sort=False)["interval_ahead_s"].shift(-1)

    pace = df["pace_3_s"]
    df["pace_delta_field_s"] = pace - df.groupby(group, sort=False)["pace_3_s"].transform("median")
    leader_pace = pace.where(df["position"] == 1)
    df["pace_delta_leader_s"] = pace - leader_pace.groupby(
        [df[c] for c in group], sort=False
    ).transform("max")

    compound = df["compound"].astype(str).str.upper()
    df["compound_soft"] = (compound == "SOFT").astype(float)
    df["compound_medium"] = (compound == "MEDIUM").astype(float)
    df["compound_hard"] = (compound == "HARD").astype(float)
    df["compound_wet"] = compound.isin(["INTERMEDIATE", "WET"]).astype(float)

    df["pit_stops_vs_field"] = df["pit_stops_done"] - df.groupby(group, sort=False)[
        "pit_stops_done"
    ].transform("median")
    df["track_state"] = df["track_state"].fillna(0.0)
    df["is_sc_or_vsc"] = (df["track_state"] >= 1).astype(float)
    df["field_remaining"] = df.groupby(group, sort=False)["driver_number"].transform("count").astype(float)
    df["cars_on_lead_lap"] = (
        (df["laps_down"] == 0).astype(float).groupby([df[c] for c in group], sort=False).transform("sum")
    )
    df["positions_gained"] = df["grid_position"] - df["position"]
    if "final_position" in df.columns:
        fin = pd.to_numeric(df["final_position"], errors="coerce")
        if "is_podium" not in df.columns or df["is_podium"].isna().all():
            df["is_podium"] = np.where(fin.notna(), (fin <= 3).astype(int), np.nan)
        if "podium" not in df.columns or df["podium"].isna().all():
            df["podium"] = df["is_podium"]
    return df


def make_in_race_x(features: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix for the in-race model."""
    return (
        features.reindex(columns=IN_RACE_FEATURES)
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .reset_index(drop=True)
    )
