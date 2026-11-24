"""Cached, rate-limited Jolpica-F1 (Ergast-compatible) REST client.

* Keyless and free: ``https://api.jolpi.ca/ergast/f1/``.
* Always sends the custom ``User-Agent: F1PitWall/1.0`` header.
* Every response is cached on disk in ``backend/cache/jolpica/`` so each URL
  is downloaded once.  Completed seasons are cached forever; the current
  season uses a short TTL so fresh qualifying/race results appear quickly.
* Uncached HTTP calls are spaced out and retried with exponential backoff on
  HTTP 429 / 5xx, honouring ``Retry-After`` when present.

The client is synchronous.  Async callers (the FastAPI server) should wrap
calls in ``asyncio.to_thread``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import requests

import config

LOGGER = logging.getLogger(__name__)

_UNSET: Any = object()
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class JolpicaError(RuntimeError):
    """Raised when Jolpica cannot satisfy a request."""


def parse_lap_time(value: Any) -> Optional[float]:
    """Convert ``"1:23.456"`` / ``"83.456"`` / ``"1:02:03.4"`` to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-", "None", "null"}:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        return None
    return None


class JolpicaClient:
    """Disk-cached Jolpica-F1 client."""

    PAGE_LIMIT: int = config.JOLPICA_PAGE_LIMIT

    def __init__(
        self,
        base_url: str = config.JOLPICA_BASE_URL,
        cache_dir: Path = config.JOLPICA_CACHE_DIR,
        user_agent: str = config.USER_AGENT,
        min_interval_s: float = config.JOLPICA_MIN_INTERVAL_S,
        max_retries: int = config.HTTP_MAX_RETRIES,
        backoff_base_s: float = config.JOLPICA_BACKOFF_BASE_S,
        timeout_s: float = config.HTTP_TIMEOUT_S,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.timeout_s = float(timeout_s)
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_request_at = float("-inf")
        self.http_calls = 0  # uncached requests actually sent (useful in tests)

        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Caching
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_ttl(season: Optional[int]) -> Optional[float]:
        """``None`` means "cache forever" (completed seasons)."""
        if season is not None and int(season) < config.current_season():
            return None
        return config.JOLPICA_CURRENT_SEASON_TTL_S

    def _cache_path(self, endpoint: str, params: Mapping[str, Any]) -> Path:
        stem = endpoint[:-5] if endpoint.endswith(".json") else endpoint
        pieces = [stem.replace("/", "__")]
        for key in sorted(params):
            pieces.append(f"{key}-{params[key]}")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", "__".join(pieces))
        return self.cache_dir / f"{name}.json"

    @staticmethod
    def _read_cache(path: Path, ttl_s: Optional[float]) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        if ttl_s is not None and (time.time() - path.stat().st_mtime) > ttl_s:
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            LOGGER.warning("Corrupt Jolpica cache file %s; refetching", path)
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _write_cache(path: Path, payload: Mapping[str, Any]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp, path)

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request_at + self.min_interval_s - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last_request_at = self._clock()

    def _backoff_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return max(float(retry_after), self.min_interval_s)
            except ValueError:
                pass
        return min(self.backoff_base_s * (2 ** attempt), 120.0)

    def _http_get(self, url: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        attempt = 0
        while True:
            self._throttle()
            self.http_calls += 1
            try:
                response = self.session.get(
                    url,
                    params=dict(params),
                    headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise JolpicaError(f"Network error calling {url}: {exc}") from exc
                delay = self._backoff_delay(attempt, None)
                LOGGER.warning("Jolpica network error (%s); retry in %.1fs", exc, delay)
                self._sleep(delay)
                attempt += 1
                continue

            status = response.status_code
            if status == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise JolpicaError(f"Invalid JSON from {url}") from exc
                if not isinstance(payload, dict):
                    raise JolpicaError(f"Unexpected payload type from {url}")
                return payload
            if status in _RETRY_STATUSES and attempt < self.max_retries:
                delay = self._backoff_delay(attempt, response.headers.get("Retry-After"))
                LOGGER.warning("Jolpica HTTP %s for %s; retry in %.1fs", status, url, delay)
                self._sleep(delay)
                attempt += 1
                continue
            raise JolpicaError(f"HTTP {status} for {url}: {response.text[:200]}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        season: Optional[int] = None,
        ttl_s: Any = _UNSET,
    ) -> Dict[str, Any]:
        """GET an Ergast-style endpoint and return its ``MRData`` object."""
        endpoint = endpoint.strip("/")
        if not endpoint.endswith(".json"):
            endpoint += ".json"
        query = dict(params or {})
        if ttl_s is _UNSET:
            ttl_s = self._default_ttl(season)

        cache_path = self._cache_path(endpoint, query)
        cached = self._read_cache(cache_path, ttl_s)
        if cached is not None:
            return cached

        payload = self._http_get(f"{self.base_url}/{endpoint}", query)
        mrdata = payload.get("MRData")
        if not isinstance(mrdata, dict):
            raise JolpicaError(f"Missing MRData in response for {endpoint}")
        # Never persist empty answers: the data may simply not exist *yet*.
        if str(mrdata.get("total", "0")) != "0":
            self._write_cache(cache_path, mrdata)
        return mrdata

    def _paginated_races(
        self, endpoint: str, list_key: str, season: int
    ) -> List[Dict[str, Any]]:
        """Fetch every page of a season-level endpoint and merge by round.

        Ergast paginates by *result rows*, so one race can be split across two
        pages; rows are merged back together here.
        """
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        offset = 0
        while True:
            mrdata = self.get(
                endpoint,
                {"limit": self.PAGE_LIMIT, "offset": offset},
                season=season,
            )
            races = mrdata.get("RaceTable", {}).get("Races", [])
            total = int(mrdata.get("total", 0) or 0)
            for race in races:
                key = (str(race.get("season")), str(race.get("round")))
                if key not in merged:
                    merged[key] = {**race, list_key: list(race.get(list_key, []))}
                else:
                    merged[key][list_key].extend(race.get(list_key, []))
            offset += self.PAGE_LIMIT
            if not races or offset >= total:
                break
        return sorted(merged.values(), key=lambda race: int(race["round"]))

    def get_schedule(self, season: int) -> List[Dict[str, Any]]:
        """Race calendar for a season (each race has ``Circuit`` and ``date``)."""
        mrdata = self.get(f"{season}", {"limit": self.PAGE_LIMIT}, season=season)
        races = mrdata.get("RaceTable", {}).get("Races", [])
        return sorted(races, key=lambda race: int(race["round"]))

    def get_race_results(
        self, season: int, round_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Race results; each returned race carries a ``Results`` list."""
        if round_number is not None:
            mrdata = self.get(
                f"{season}/{round_number}/results",
                {"limit": self.PAGE_LIMIT},
                season=season,
            )
            return mrdata.get("RaceTable", {}).get("Races", [])
        return self._paginated_races(f"{season}/results", "Results", season)

    def get_qualifying_results(
        self, season: int, round_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Qualifying results; each race carries a ``QualifyingResults`` list."""
        if round_number is not None:
            mrdata = self.get(
                f"{season}/{round_number}/qualifying",
                {"limit": self.PAGE_LIMIT},
                season=season,
            )
            return mrdata.get("RaceTable", {}).get("Races", [])
        return self._paginated_races(f"{season}/qualifying", "QualifyingResults", season)

    def get_race(self, season: int, round_number: int) -> Optional[Dict[str, Any]]:
        """Schedule entry for one round, or ``None`` if it does not exist."""
        for race in self.get_schedule(season):
            if int(race["round"]) == int(round_number):
                return race
        return None

    def get_latest_round_with_results(self, season: int) -> Optional[int]:
        """Highest round of ``season`` that already has race results."""
        races = [r for r in self.get_race_results(season) if r.get("Results")]
        return int(races[-1]["round"]) if races else None
