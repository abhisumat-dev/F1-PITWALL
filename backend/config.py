"""Central configuration for the F1 Pit-Wall backend.

Every tunable lives here so the clients, ML pipeline, stream providers and
server share one source of truth.  Values can be overridden with environment
variables (or a ``backend/.env`` file when ``python-dotenv`` is installed).

Run every module from the ``backend/`` directory, for example::

    cd backend
    python -m ml.ingest_history
    python -m ml.train
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

try:  # Optional convenience; never required.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    pass


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

CONTRACT_PATH: Path = PROJECT_ROOT / "contract" / "live_packet.json"
SCHEMA_VERSION: str = "1.0"

CACHE_DIR: Path = Path(_env_str("F1_CACHE_DIR", str(BACKEND_DIR / "cache")))
JOLPICA_CACHE_DIR: Path = CACHE_DIR / "jolpica"
OPENF1_CACHE_DIR: Path = CACHE_DIR / "openf1"

DATA_DIR: Path = Path(_env_str("F1_DATA_DIR", str(BACKEND_DIR / "data")))
PRE_RACE_RAW_PATH: Path = DATA_DIR / "pre_race_raw.parquet"
IN_RACE_RAW_PATH: Path = DATA_DIR / "in_race_raw.parquet"

ARTIFACT_DIR: Path = Path(
    _env_str("F1_ARTIFACT_DIR", str(BACKEND_DIR / "ml" / "artifacts"))
)
PRE_RACE_MODEL_PATH: Path = ARTIFACT_DIR / "pre_race_model.joblib"
LIVE_RACE_MODEL_PATH: Path = ARTIFACT_DIR / "live_race_model.joblib"
IN_RACE_MODEL_PATH: Path = LIVE_RACE_MODEL_PATH
METRICS_PATH: Path = ARTIFACT_DIR / "metrics.json"


# --------------------------------------------------------------------------- #
# HTTP / free data stack
# --------------------------------------------------------------------------- #
USER_AGENT: str = "F1PitWall/1.0"
HTTP_TIMEOUT_S: float = _env_float("F1_HTTP_TIMEOUT_S", 30.0)
HTTP_MAX_RETRIES: int = _env_int("F1_HTTP_MAX_RETRIES", 6)

# Jolpica-F1 (Ergast-compatible, keyless). Burst limit is 4 req/s and the
# sustained limit is far lower, so we space calls out and cache everything.
JOLPICA_BASE_URL: str = _env_str(
    "JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1"
).rstrip("/")
JOLPICA_MIN_INTERVAL_S: float = _env_float("JOLPICA_MIN_INTERVAL_S", 0.4)
JOLPICA_BACKOFF_BASE_S: float = _env_float("JOLPICA_BACKOFF_BASE_S", 2.0)
JOLPICA_PAGE_LIMIT: int = 100
# Completed seasons are cached forever; the current season is refreshed.
JOLPICA_CURRENT_SEASON_TTL_S: float = _env_float("JOLPICA_CURRENT_TTL_S", 900.0)

# OpenF1 (free, unauthenticated historical REST API, 2023-2026).
# Free tier: 3 req/s and 30 req/min -> one uncached call every 2.1 s.
OPENF1_BASE_URL: str = _env_str("OPENF1_BASE_URL", "https://api.openf1.org/v1").rstrip("/")
OPENF1_MIN_INTERVAL_S: float = _env_float("OPENF1_MIN_INTERVAL_S", 2.1)
OPENF1_BACKOFF_BASE_S: float = _env_float("OPENF1_BACKOFF_BASE_S", 2.1)
OPENF1_SCHEDULE_TTL_S: float = _env_float("OPENF1_SCHEDULE_TTL_S", 6 * 3600.0)
OPENF1_EMPTY_TTL_S: float = _env_float("OPENF1_EMPTY_TTL_S", 1800.0)
OPENF1_TELEMETRY_CHUNK_S: int = _env_int("OPENF1_TELEMETRY_CHUNK_S", 600)

# Free live race feed (unauthenticated SignalR). Used by Stage 2.
SIGNALR_URL: str = _env_str("F1_SIGNALR_URL", "wss://livetiming.formula1.com/signalrcore")
SIGNALR_NEGOTIATE_URL: str = _env_str(
    "F1_SIGNALR_NEGOTIATE_URL", "https://livetiming.formula1.com/signalrcore"
)
LIVE_TOPICS: Tuple[str, ...] = (
    "TimingData",
    "TimingAppData",
    "TimingStats",
    "LapCount",
    "WeatherData",
    "RaceControlMessages",
    "TeamRadio",
    "DriverList",
)
# Optional credential hook: when set, Position.z / CarData.z unlock.
F1_LIVE_TOKEN: str = _env_str("F1_LIVE_TOKEN", "")
LIVE_TOKEN_TOPICS: Tuple[str, ...] = ("Position.z", "CarData.z")


# --------------------------------------------------------------------------- #
# ML pipeline
# --------------------------------------------------------------------------- #
TRAIN_START_SEASON: int = _env_int("F1_TRAIN_START_SEASON", 2018)
# OpenF1 historical coverage starts in 2023, which bounds the in-race model.
IN_RACE_START_SEASON: int = _env_int("F1_IN_RACE_START_SEASON", 2023)
RANDOM_SEED: int = 42
WALK_FORWARD_SPLITS: int = _env_int("F1_WALK_FORWARD_SPLITS", 3)
MIN_TRAIN_SEASONS: int = _env_int("F1_MIN_TRAIN_SEASONS", 2)
RECENCY_HALF_LIFE_SEASONS: float = _env_float("F1_RECENCY_HALF_LIFE", 3.0)
PROBABILITY_DECIMALS: int = 2
FIRST_LIVE_SNAPSHOT_LAP: int = 2


# --------------------------------------------------------------------------- #
# Tyre degradation (Stage 2)
# --------------------------------------------------------------------------- #
TYRE_DEG_MIN_LAPS: int = _env_int("F1_TYRE_DEG_MIN_LAPS", 4)
TYRE_DEG_WINDOW_LAPS: int = _env_int("F1_TYRE_DEG_WINDOW_LAPS", 8)
FUEL_BURN_CORRECTION_S_PER_LAP: float = _env_float("F1_FUEL_BURN_CORRECTION_S_PER_LAP", 0.035)
FUEL_START_KG: float = _env_float("F1_FUEL_START_KG", 100.0)
FUEL_EFFECT_S_PER_KG: float = _env_float("F1_FUEL_EFFECT_S_PER_KG", 0.035)


# --------------------------------------------------------------------------- #
# Server / replay (Stage 2 + 3)
# --------------------------------------------------------------------------- #
SERVER_HOST: str = _env_str("F1_SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = _env_int("F1_SERVER_PORT", 8000)
REPLAY_DEFAULT_SPEED: float = _env_float("F1_REPLAY_SPEED", 1.0)
REPLAY_TICK_S: float = _env_float("F1_REPLAY_TICK_S", 0.25)
RACE_CONTROL_TAIL: int = _env_int("F1_RACE_CONTROL_TAIL", 15)
ALLOW_LIVE_NETWORK: bool = _env_bool("F1_ALLOW_LIVE_NETWORK", True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def current_season() -> int:
    """Calendar year in UTC, used as the "current" F1 season."""
    return datetime.now(timezone.utc).year


def ensure_directories() -> None:
    """Create every directory the backend writes to."""
    for path in (
        JOLPICA_CACHE_DIR,
        OPENF1_CACHE_DIR,
        DATA_DIR,
        ARTIFACT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for CLI entry points."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
