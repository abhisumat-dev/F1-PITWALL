"""Post-qualifying and live in-race Top-3 (Podium) inference.

``RacePredictor`` loads ``pre_race_model.joblib`` and ``live_race_model.joblib``
and produces contract-ready objects:

* Every driver receives both ``podium_prob_pct`` (primary Top 3 finish
  probability, ``0.0`` to ``100.0``) and ``win_prob_pct`` (``0.0`` to ``100.0``).
* ``DNF`` / ``OUT`` / retired drivers receive ``podium_prob_pct = 0.0`` and
  ``win_prob_pct = 0.0``.
* Across active drivers, iterative clipping-and-normalization ensures every
  driver stays within ``[0.0, 100.0]`` and ``podium_prob_pct`` sums to
  ``300.0%`` (3 podium spots per race), while ``win_prob_pct`` sums to ``100.0%``.
* Leaderboard items are sorted descending by ``podium_prob_pct``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import config
from clients.jolpica_client import JolpicaClient
from clients.openf1_client import OpenF1Client
from ml.features import (
    TRACK_STATE_CODES,
    build_in_race_features,
    build_pre_race_features,
    make_in_race_x,
    make_pre_race_x,
    normalize_podium_to_percent,
    normalize_to_percent,
)
from ml.ingest_history import (
    build_pre_race_dataset,
    fetch_race_rows,
    match_session_to_race_date,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ModelBundle",
    "ModelNotAvailableError",
    "PredictionError",
    "RaceContext",
    "RacePredictor",
    "is_driver_out",
    "normalize_podium_to_percent",
    "normalize_to_percent",
]

_OUT_STATUSES = frozenset({"DNF", "OUT", "RET", "RETIRED", "DNS", "DSQ", "NC", "WD", "EXCLUDED"})


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced."""


class ModelNotAvailableError(PredictionError):
    """Raised when a trained model artifact is missing."""


@dataclass
class ModelBundle:
    stage: str
    model: Any
    features: List[str]
    name: str
    version: str
    trained_at: Optional[str]
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RaceContext:
    """Pre-race information for one Grand Prix, reused by live inference."""

    season: int
    round_number: int
    race_name: str
    circuit_id: Optional[str]
    circuit_name: Optional[str]
    race_date: Optional[str]
    grid_source: str
    features: pd.DataFrame
    meeting_key: Optional[int] = None
    session_key: Optional[int] = None
    rotation_deg: Optional[float] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_driver_out(driver: Mapping[str, Any]) -> bool:
    """True when a driver is DNF, OUT, retired, or lacks a valid running position."""
    if bool(driver.get("retired")) or bool(driver.get("dnf")) or bool(driver.get("out")):
        return True
    status = str(driver.get("status") or "").strip().upper()
    if status in _OUT_STATUSES or status.startswith("DNF") or status.startswith("RET"):
        return True
    pos = _int_or_none(driver.get("position"))
    return pos is None or pos <= 0


def load_bundle(path: Path, expected_stage: str) -> Optional[ModelBundle]:
    """Load a joblib artifact, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        # Check fallback alias between live_race_model.joblib and in_race_model.joblib
        if path.name == "live_race_model.joblib":
            alt = path.parent / "in_race_model.joblib"
            if alt.exists():
                path = alt
            else:
                return None
        elif path.name == "in_race_model.joblib":
            alt = path.parent / "live_race_model.joblib"
            if alt.exists():
                path = alt
            else:
                return None
        else:
            return None
    artifact = joblib.load(path)
    stage = artifact.get("stage")
    valid_stages = {expected_stage}
    if expected_stage in {"in_race", "live_race"}:
        valid_stages = {"in_race", "live_race"}
    if stage not in valid_stages:
        raise PredictionError(f"{path} holds a {stage!r} model, expected {expected_stage!r}")
    return ModelBundle(
        stage=expected_stage,
        model=artifact["model"],
        features=list(artifact["features"]),
        name=str(artifact.get("model_name", "unknown")),
        version=str(artifact.get("version", "unknown")),
        trained_at=artifact.get("trained_at"),
        validation=dict(artifact.get("validation") or {}),
    )


class RacePredictor:
    """Dual-stage (pre-race + live) Top-3 Podium and race-winner predictor."""

    def __init__(
        self,
        *,
        jolpica: Optional[JolpicaClient] = None,
        openf1: Optional[OpenF1Client] = None,
        pre_race_path: Optional[Path] = None,
        in_race_path: Optional[Path] = None,
        live_race_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
    ) -> None:
        self.jolpica = jolpica or JolpicaClient()
        self.openf1 = openf1 or OpenF1Client()
        self.pre_race_path = Path(pre_race_path or config.PRE_RACE_MODEL_PATH)
        self.in_race_path = Path(live_race_path or in_race_path or config.LIVE_RACE_MODEL_PATH)
        self.history_path = Path(history_path or config.PRE_RACE_RAW_PATH)
        self._pre: Optional[ModelBundle] = None
        self._in: Optional[ModelBundle] = None
        self._history: Optional[pd.DataFrame] = None
        self.reload_models()

    # ------------------------------------------------------------------ #
    # Models / history
    # ------------------------------------------------------------------ #
    def reload_models(self) -> None:
        self._pre = load_bundle(self.pre_race_path, "pre_race")
        self._in = load_bundle(self.in_race_path, "in_race")

    @property
    def pre_race_available(self) -> bool:
        return self._pre is not None

    @property
    def in_race_available(self) -> bool:
        return self._in is not None

    def model_info(self) -> Dict[str, Any]:
        def describe(bundle: Optional[ModelBundle]) -> Optional[Dict[str, Any]]:
            if bundle is None:
                return None
            return {
                "name": bundle.name,
                "version": bundle.version,
                "target": "is_podium",
                "trained_at": bundle.trained_at,
                "validation": bundle.validation,
            }

        return {"pre_race": describe(self._pre), "in_race": describe(self._in)}

    def _load_history(self) -> pd.DataFrame:
        if self._history is None:
            if self.history_path.exists():
                self._history = pd.read_parquet(self.history_path)
            else:
                season = config.current_season()
                LOGGER.warning("%s missing; building a 4-season history from Jolpica", self.history_path)
                self._history = build_pre_race_dataset(self.jolpica, season - 3, season)
        return self._history

    @staticmethod
    def _require(bundle: Optional[ModelBundle], label: str) -> ModelBundle:
        if bundle is None:
            raise ModelNotAvailableError(
                f"The {label} model is not trained yet. Run `python -m ml.train` from backend/."
            )
        return bundle

    @staticmethod
    def _score(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
        return bundle.model.predict_proba(X.reindex(columns=bundle.features))[:, 1]

    @staticmethod
    def _win_scores_from_podium(podium_scores: np.ndarray) -> np.ndarray:
        """Sharpen podium probabilities into relative win scores summing to 100%."""
        clipped = np.clip(np.asarray(podium_scores, dtype=float), 1e-6, 1.0 - 1e-6)
        # Odds sharpening: higher podium likelihood concentrates more heavily for P1
        odds = clipped / (1.0 - clipped)
        return np.power(odds, 1.35)

    # ------------------------------------------------------------------ #
    # Race lookup
    # ------------------------------------------------------------------ #
    def latest_qualified_round(self, season: Optional[int] = None) -> Tuple[int, int]:
        """Most recent ``(season, round)`` that has qualifying results."""
        start = season or config.current_season()
        for candidate in (start, start - 1):
            races = [r for r in self.jolpica.get_qualifying_results(candidate) if r.get("QualifyingResults")]
            if races:
                return candidate, int(races[-1]["round"])
        raise PredictionError("No qualifying results found for the current or previous season.")

    def _resolve_openf1(self, season: int, race_date: Optional[str]) -> Dict[str, Any]:
        """Best-effort OpenF1 meeting/session/rotation lookup (never raises)."""
        info: Dict[str, Any] = {}
        try:
            meeting_keys = {int(m["meeting_key"]) for m in self.openf1.list_races(season)}
            sessions = [
                s
                for s in self.openf1.get_sessions(year=season, session_name="Race")
                if _int_or_none(s.get("meeting_key")) in meeting_keys
            ]
            session = match_session_to_race_date(sessions, race_date)
            if session is None:
                return info
            info["session_key"] = _int_or_none(session.get("session_key"))
            info["meeting_key"] = _int_or_none(session.get("meeting_key"))
            if info["meeting_key"] is not None:
                info["rotation_deg"] = self.openf1.get_circuit_rotation(info["meeting_key"])
        except Exception as exc:  # noqa: BLE001 - OpenF1 is an optional enrichment
            LOGGER.warning("OpenF1 enrichment unavailable: %s", exc)
        return info

    def _openf1_grid(self, session_key: Optional[int]) -> Dict[int, int]:
        """Official starting grid (with penalties) once OpenF1 publishes it."""
        if session_key is None:
            return {}
        try:
            rows = self.openf1.get_starting_grid(session_key)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OpenF1 starting grid unavailable: %s", exc)
            return {}
        grid: Dict[int, int] = {}
        for row in rows:
            number, position = _int_or_none(row.get("driver_number")), _int_or_none(row.get("position"))
            if number is not None and position is not None:
                grid[number] = position
        return grid

    # ------------------------------------------------------------------ #
    # Pre-race (post-qualifying)
    # ------------------------------------------------------------------ #
    def build_race_context(
        self, season: int, round_number: int, *, use_openf1: bool = True
    ) -> RaceContext:
        """Assemble pre-race features for every driver on the grid."""
        rows = fetch_race_rows(self.jolpica, season, round_number)
        if not rows:
            raise PredictionError(f"No qualifying or result data for {season} round {round_number}.")
        race = pd.DataFrame(rows)
        completed = bool(race["completed"].astype(bool).any())
        grid_source = "jolpica_results" if completed else "jolpica_qualifying"

        extra: Dict[str, Any] = {}
        if use_openf1:
            extra = self._resolve_openf1(season, str(race["race_date"].iloc[0]))
            if not completed:
                official = self._openf1_grid(extra.get("session_key"))
                if official:
                    numbers = pd.to_numeric(race["driver_number"], errors="coerce")
                    mapped = numbers.map(lambda n: official.get(int(n)) if pd.notna(n) else None)
                    if mapped.notna().sum() >= max(2, len(race) // 2):
                        race["grid"] = mapped.where(mapped.notna(), race["grid"])
                        grid_source = "openf1_starting_grid"

        history = self._load_history()
        previous = history[~((history["season"] == season) & (history["round"] == round_number))]
        combined = pd.concat([previous, race], ignore_index=True, sort=False)
        featured = build_pre_race_features(combined)
        target = featured[(featured["season"] == season) & (featured["round"] == round_number)]
        target = target.reset_index(drop=True)
        if target.empty:
            raise PredictionError(f"Feature build produced no rows for {season} round {round_number}.")

        first = target.iloc[0]
        return RaceContext(
            season=int(season),
            round_number=int(round_number),
            race_name=str(first.get("race_name") or f"Round {round_number}"),
            circuit_id=first.get("circuit_id") if pd.notna(first.get("circuit_id")) else None,
            circuit_name=first.get("circuit_name") if pd.notna(first.get("circuit_name")) else None,
            race_date=str(first.get("race_date")) if pd.notna(first.get("race_date")) else None,
            grid_source=grid_source,
            features=target,
            meeting_key=extra.get("meeting_key"),
            session_key=extra.get("session_key"),
            rotation_deg=extra.get("rotation_deg"),
        )

    def predict_post_quali(
        self,
        season: Optional[int] = None,
        round_number: Optional[int] = None,
        *,
        context: Optional[RaceContext] = None,
        sequence: int = 0,
    ) -> Dict[str, Any]:
        """Return a ``post_quali_prediction`` packet (contract-valid, Top 3 Podium sorted)."""
        bundle = self._require(self._pre, "pre-race")
        if context is None:
            if season is None or round_number is None:
                season, round_number = self.latest_qualified_round(season)
            context = self.build_race_context(season, round_number)

        table = context.features
        raw_scores = self._score(bundle, make_pre_race_x(table))
        podium_pcts = normalize_podium_to_percent(
            raw_scores,
            active_mask=[True] * len(table),
            decimals=config.PROBABILITY_DECIMALS,
            target_sum=300.0,
            max_per_driver=100.0,
        )
        win_pcts = normalize_to_percent(
            self._win_scores_from_podium(raw_scores),
            decimals=config.PROBABILITY_DECIMALS,
        )

        entries: List[Dict[str, Any]] = []
        for (_, row), podium_pct, win_pct in zip(table.iterrows(), podium_pcts, win_pcts):
            entries.append(
                {
                    "driver_number": _int_or_none(row.get("driver_number")) or 0,
                    "driver_code": str(row.get("driver_code") or ""),
                    "full_name": str(row.get("driver_name") or row.get("driver_code") or ""),
                    "team_name": str(row.get("constructor_name") or ""),
                    "grid_position": max(1, _int_or_none(row.get("grid_position")) or 1),
                    "podium_prob_pct": float(podium_pct),
                    "win_prob_pct": float(win_pct),
                    "win_probability_pct": float(win_pct),
                }
            )
        entries.sort(key=lambda e: (-e["podium_prob_pct"], -e["win_prob_pct"], e["grid_position"]))
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

        sum_podium = round(float(sum(e["podium_prob_pct"] for e in entries)), config.PROBABILITY_DECIMALS)
        sum_win = round(float(sum(e["win_prob_pct"] for e in entries)), config.PROBABILITY_DECIMALS)

        return {
            "schema_version": config.SCHEMA_VERSION,
            "type": "post_quali_prediction",
            "timestamp": _now_iso(),
            "sequence": int(sequence),
            "meeting_key": context.meeting_key,
            "session_key": context.session_key,
            "season": context.season,
            "round": context.round_number,
            "grand_prix": context.race_name,
            "circuit": {
                "circuit_id": context.circuit_id,
                "name": context.circuit_name,
                "rotation_deg": context.rotation_deg,
            },
            "grid_source": context.grid_source,
            "predictions": {
                "stage": "pre_race",
                "target": "top3_podium",
                "model_name": bundle.name,
                "model_version": bundle.version,
                "trained_at": bundle.trained_at,
                "generated_at": _now_iso(),
                "validation": {k: v for k, v in bundle.validation.items() if isinstance(v, (int, float)) or v is None},
                "items": entries,
                "sum_podium_pct": sum_podium,
                "sum_win_pct": sum_win,
                "sum_pct": sum_podium,
            },
        }

    # ------------------------------------------------------------------ #
    # Live in-race
    # ------------------------------------------------------------------ #
    def predict_live(
        self,
        context: RaceContext,
        drivers: Sequence[Mapping[str, Any]],
        *,
        lap: int,
        total_laps: Optional[int] = None,
        track_state: Any = "GREEN",
        pace_by_driver: Optional[Mapping[int, float]] = None,
        ref_lap_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Top-3 Podium & Win probabilities for the current race state.

        ``drivers`` are contract ``DriverState`` dictionaries.  Retired / DNF /
        OUT cars (or cars without a position) get ``0.0%``; active drivers are
        clipped to ``[0.0, 100.0]`` and normalized so ``podium_prob_pct`` sums
        to ``300.0%`` and ``win_prob_pct`` sums to ``100.0%``.
        """
        bundle = self._require(self._in, "in-race")
        state_code = (
            TRACK_STATE_CODES.get(str(track_state).upper(), 0)
            if not isinstance(track_state, (int, np.integer))
            else int(track_state)
        )

        active: List[Mapping[str, Any]] = []
        for driver in drivers:
            if is_driver_out(driver):
                continue
            active.append(driver)
        if not active:
            raise PredictionError("No active drivers with a position to predict.")

        paces: List[Optional[float]] = []
        for driver in active:
            number = _int_or_none(driver.get("driver_number"))
            pace = (pace_by_driver or {}).get(number) if number is not None else None
            paces.append(pace if pace is not None else driver.get("last_lap_time_s"))
        valid_paces = [float(p) for p in paces if p is not None and np.isfinite(p) and p > 0]
        reference = ref_lap_s or (float(np.median(valid_paces)) if valid_paces else 90.0)
        laps_total = total_laps or max(int(lap) + 1, 57)

        rows: List[Dict[str, Any]] = []
        for driver, pace in zip(active, paces):
            tyre = driver.get("tyre") or {}
            rows.append(
                {
                    "season": context.season,
                    "round": context.round_number,
                    "lap": int(lap),
                    "total_laps": laps_total,
                    "driver_number": _int_or_none(driver.get("driver_number")),
                    "driver_code": driver.get("code"),
                    "position": _int_or_none(driver.get("position")),
                    "laps_down": driver.get("laps_down") or 0,
                    "gap_to_leader_s": driver.get("gap_to_leader_s"),
                    "interval_s": driver.get("interval_s"),
                    "pace_3_s": pace,
                    "ref_lap_s": reference,
                    "compound": tyre.get("compound"),
                    "tyre_age_laps": tyre.get("age_laps"),
                    "stint_number": tyre.get("stint_number"),
                    "pit_stops_done": len(driver.get("pit_stops") or []),
                    "track_state": state_code,
                }
            )
        snapshot = pd.DataFrame(rows)
        features = build_in_race_features(snapshot, context.features)
        raw_scores = self._score(bundle, make_in_race_x(features))

        podium_pcts = normalize_podium_to_percent(
            raw_scores,
            active_mask=[True] * len(features),
            decimals=config.PROBABILITY_DECIMALS,
            target_sum=300.0,
            max_per_driver=100.0,
        )
        win_pcts = normalize_to_percent(
            self._win_scores_from_podium(raw_scores),
            decimals=config.PROBABILITY_DECIMALS,
        )

        podium_by_number = {int(n): float(pct) for n, pct in zip(features["driver_number"], podium_pcts)}
        win_by_number = {int(n): float(pct) for n, pct in zip(features["driver_number"], win_pcts)}
        code_by_number = {int(n): c for n, c in zip(features["driver_number"], features["driver_code"])}

        items: List[Dict[str, Any]] = []
        for driver in drivers:
            number = _int_or_none(driver.get("driver_number"))
            if number is None:
                continue
            out = is_driver_out(driver)
            p_pct = 0.0 if out else podium_by_number.get(number, 0.0)
            w_pct = 0.0 if out else win_by_number.get(number, 0.0)
            item: Dict[str, Any] = {
                "driver_number": number,
                "driver_code": str(driver.get("code") or code_by_number.get(number) or number),
                "podium_prob_pct": p_pct,
                "win_prob_pct": w_pct,
                "win_probability_pct": w_pct,
            }
            if driver.get("full_name"):
                item["full_name"] = str(driver["full_name"])
            if driver.get("team_name"):
                item["team_name"] = str(driver["team_name"])
            if driver.get("team_colour"):
                item["team_colour"] = str(driver["team_colour"])
            items.append(item)

        items.sort(key=lambda item: (-item["podium_prob_pct"], -item["win_prob_pct"], item["driver_number"]))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank

        sum_podium = round(float(sum(i["podium_prob_pct"] for i in items)), config.PROBABILITY_DECIMALS)
        sum_win = round(float(sum(i["win_prob_pct"] for i in items)), config.PROBABILITY_DECIMALS)

        return {
            "stage": "in_race",
            "target": "top3_podium",
            "model_version": bundle.version,
            "generated_at": _now_iso(),
            "lap": int(lap),
            "items": items,
            "sum_podium_pct": sum_podium,
            "sum_win_pct": sum_win,
            "sum_pct": sum_podium,
        }
