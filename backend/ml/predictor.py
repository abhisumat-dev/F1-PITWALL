"""Post-qualifying and live in-race inference.

``RacePredictor`` loads the two trained artifacts and produces contract-ready
objects.  Every probability set is normalised so it sums to exactly 100 %.

* :meth:`RacePredictor.predict_post_quali` builds a complete
  ``post_quali_prediction`` packet (see ``contract/live_packet.json``).
* :meth:`RacePredictor.predict_live` takes the current driver states of a
  ``live_race_update`` / ``replay_frame`` and returns the ``predictions``
  block of that packet.
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
    "normalize_to_percent",
]


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


def load_bundle(path: Path, expected_stage: str) -> Optional[ModelBundle]:
    """Load a joblib artifact, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    artifact = joblib.load(path)
    if artifact.get("stage") != expected_stage:
        raise PredictionError(f"{path} holds a {artifact.get('stage')!r} model, expected {expected_stage!r}")
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
    """Dual-stage (pre-race + live) race-winner predictor."""

    def __init__(
        self,
        *,
        jolpica: Optional[JolpicaClient] = None,
        openf1: Optional[OpenF1Client] = None,
        pre_race_path: Optional[Path] = None,
        in_race_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
    ) -> None:
        self.jolpica = jolpica or JolpicaClient()
        self.openf1 = openf1 or OpenF1Client()
        self.pre_race_path = Path(pre_race_path or config.PRE_RACE_MODEL_PATH)
        self.in_race_path = Path(in_race_path or config.IN_RACE_MODEL_PATH)
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
        """Return a ``post_quali_prediction`` packet (contract-valid)."""
        bundle = self._require(self._pre, "pre-race")
        if context is None:
            if season is None or round_number is None:
                season, round_number = self.latest_qualified_round(season)
            context = self.build_race_context(season, round_number)

        table = context.features
        scores = self._score(bundle, make_pre_race_x(table))
        percents = normalize_to_percent(scores, config.PROBABILITY_DECIMALS)

        entries: List[Dict[str, Any]] = []
        for (_, row), pct in zip(table.iterrows(), percents):
            entries.append(
                {
                    "driver_number": _int_or_none(row.get("driver_number")) or 0,
                    "driver_code": str(row.get("driver_code") or ""),
                    "full_name": str(row.get("driver_name") or row.get("driver_code") or ""),
                    "team_name": str(row.get("constructor_name") or ""),
                    "grid_position": max(1, _int_or_none(row.get("grid_position")) or 1),
                    "win_probability_pct": pct,
                }
            )
        entries.sort(key=lambda e: (-e["win_probability_pct"], e["grid_position"]))
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

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
                "model_name": bundle.name,
                "model_version": bundle.version,
                "trained_at": bundle.trained_at,
                "generated_at": _now_iso(),
                "validation": {k: v for k, v in bundle.validation.items() if isinstance(v, (int, float)) or v is None},
                "items": entries,
                "sum_pct": 100.0,
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
        """Win probabilities for the current race state.

        ``drivers`` are contract ``DriverState`` dictionaries.  Retired cars (or
        cars without a position) get 0 %; the rest sum to exactly 100 %.
        ``pace_by_driver`` optionally supplies the rolling clean 3-lap pace in
        seconds (falls back to ``last_lap_time_s``).
        """
        bundle = self._require(self._in, "in-race")
        state_code = (
            TRACK_STATE_CODES.get(str(track_state).upper(), 0)
            if not isinstance(track_state, (int, np.integer))
            else int(track_state)
        )

        active: List[Mapping[str, Any]] = []
        for driver in drivers:
            if driver.get("retired") or _int_or_none(driver.get("position")) is None:
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
        scores = self._score(bundle, make_in_race_x(features))
        percents = normalize_to_percent(scores, config.PROBABILITY_DECIMALS)

        by_number = {int(n): pct for n, pct in zip(features["driver_number"], percents)}
        code_by_number = {int(n): c for n, c in zip(features["driver_number"], features["driver_code"])}
        items: List[Dict[str, Any]] = []
        for driver in drivers:
            number = _int_or_none(driver.get("driver_number"))
            if number is None:
                continue
            items.append(
                {
                    "driver_number": number,
                    "driver_code": str(driver.get("code") or code_by_number.get(number) or number),
                    "win_probability_pct": by_number.get(number, 0.0),
                }
            )
        items.sort(key=lambda item: -item["win_probability_pct"])
        return {
            "stage": "in_race",
            "model_version": bundle.version,
            "generated_at": _now_iso(),
            "lap": int(lap),
            "items": items,
            "sum_pct": 100.0,
        }
