"""Cached, rate-limited OpenF1 REST client (free, unauthenticated, 2023-2026).

Behaviour required by the project spec:

* Every downloaded endpoint is cached in
  ``backend/cache/openf1/{session_key}/`` as ``.json`` (or ``.parquet`` for the
  large telemetry endpoints) so each session is fetched only once.  Requests
  that are not tied to a session (``meetings``/``sessions`` listings) live in
  ``backend/cache/openf1/_global/``.
* A ``2.1 s`` delay is enforced between *uncached* HTTP calls (free tier:
  3 req/s and 30 req/min) with exponential backoff on HTTP ``429``.
* 2026 API rules: cancelled rounds (``is_cancelled``) are filtered out when
  listing races, the deprecated ``country_code`` field is never queried, and
  ``car_data.drs`` is treated as nullable (always present, ``None`` in 2026).
* ``meetings[0]["circuit_info_url"]`` is fetched to extract the circuit's
  ``rotation`` angle for TV map orientation.

The client is synchronous; async callers should use ``asyncio.to_thread``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote

import requests

import config

LOGGER = logging.getLogger(__name__)

_UNSET: Any = object()
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: All 18 OpenF1 v1 endpoints.
ENDPOINTS = frozenset(
    {
        "car_data",
        "championship_drivers",
        "championship_teams",
        "drivers",
        "intervals",
        "laps",
        "location",
        "meetings",
        "overtakes",
        "pit",
        "position",
        "race_control",
        "session_result",
        "sessions",
        "starting_grid",
        "stints",
        "team_radio",
        "weather",
    }
)
#: Large, flat, homogeneous endpoints stored as parquet when pyarrow exists.
PARQUET_ENDPOINTS = frozenset({"car_data", "location", "position"})
#: Endpoints downloaded by :meth:`OpenF1Client.prefetch_session`.
DEFAULT_PREFETCH_ENDPOINTS = (
    "drivers",
    "session_result",
    "starting_grid",
    "laps",
    "position",
    "intervals",
    "pit",
    "stints",
    "weather",
    "race_control",
)

_KEY_RE = re.compile(r"^[a-z_]+(?:[<>]=?)?$")
_SIMPLE_VALUE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class OpenF1Error(RuntimeError):
    """Raised when OpenF1 cannot satisfy a request."""


def parse_iso_datetime(value: Any) -> datetime:
    """Parse an OpenF1 timestamp into an aware UTC ``datetime``."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        # Normalise fractional seconds to exactly 6 digits for old Pythons.
        text = re.sub(
            r"\.(\d+)",
            lambda m: "." + (m.group(1) + "000000")[:6],
            text,
            count=1,
        )
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


class OpenF1Client:
    """Disk-cached OpenF1 client."""

    def __init__(
        self,
        base_url: str = config.OPENF1_BASE_URL,
        cache_dir: Path = config.OPENF1_CACHE_DIR,
        user_agent: str = config.USER_AGENT,
        min_interval_s: float = config.OPENF1_MIN_INTERVAL_S,
        max_retries: int = config.HTTP_MAX_RETRIES,
        backoff_base_s: float = config.OPENF1_BACKOFF_BASE_S,
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
        self.http_calls = 0  # uncached requests actually sent

        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _session_cache_dir(self, session_key: Any) -> Path:
        if session_key is None:
            path = self.cache_dir / "_global"
        else:
            path = self.cache_dir / str(int(session_key))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _cache_stem(endpoint: str, params: Mapping[str, Any]) -> str:
        extras = {k: v for k, v in params.items() if k != "session_key" and v is not None}
        if not extras:
            return endpoint
        simple = all(
            re.fullmatch(r"[a-z_]+", k) and _SIMPLE_VALUE_RE.match(str(v))
            for k, v in extras.items()
        )
        if simple:
            slug = "__".join(f"{k}-{extras[k]}" for k in sorted(extras))
            if len(slug) <= 80:
                return f"{endpoint}__{slug}"
        digest = hashlib.sha1(
            json.dumps(sorted((k, str(v)) for k, v in extras.items())).encode("utf-8")
        ).hexdigest()[:12]
        return f"{endpoint}__{digest}"

    @staticmethod
    def _default_ttl(endpoint: str, params: Mapping[str, Any]) -> Optional[float]:
        """``None`` caches forever (finished sessions never change)."""
        if endpoint in {"meetings", "sessions"}:
            year = params.get("year")
            try:
                if year is not None and int(year) < config.current_season():
                    return None
            except (TypeError, ValueError):
                pass
            return config.OPENF1_SCHEDULE_TTL_S
        return None

    @staticmethod
    def _load_file(path: Path) -> Optional[List[Dict[str, Any]]]:
        try:
            if path.suffix == ".parquet":
                import pandas as pd  # local import: pandas is optional for JSON-only use

                frame = pd.read_parquet(path)
                frame = frame.astype(object).where(frame.notna(), None)
                return frame.to_dict("records")
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else None
        except Exception as exc:  # noqa: BLE001 - any unreadable cache is a miss
            LOGGER.warning("Unreadable OpenF1 cache file %s (%s); refetching", path, exc)
            return None

    def _read_cache(
        self, directory: Path, stem: str, ttl_s: Optional[float]
    ) -> Optional[List[Dict[str, Any]]]:
        for suffix in (".parquet", ".json"):
            path = directory / f"{stem}{suffix}"
            if not path.exists():
                continue
            age = time.time() - path.stat().st_mtime
            if ttl_s is not None and age > ttl_s:
                continue
            rows = self._load_file(path)
            if rows is None:
                continue
            # Empty answers may just mean "not published yet": expire quickly.
            if not rows and age > config.OPENF1_EMPTY_TTL_S:
                continue
            return rows
        return None

    @staticmethod
    def _write_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, separators=(",", ":"))
        os.replace(tmp, path)

    def _write_cache(
        self,
        directory: Path,
        stem: str,
        endpoint: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        json_path = directory / f"{stem}.json"
        parquet_path = directory / f"{stem}.parquet"
        if rows and endpoint in PARQUET_ENDPOINTS:
            try:
                import pandas as pd

                tmp = parquet_path.with_name(parquet_path.name + ".tmp")
                pd.DataFrame.from_records(rows).to_parquet(tmp, index=False)
                os.replace(tmp, parquet_path)
                if json_path.exists():
                    json_path.unlink()
                return
            except Exception as exc:  # noqa: BLE001 - fall back to JSON
                LOGGER.debug("Parquet cache write failed for %s (%s); using JSON", stem, exc)
        self._write_json(json_path, rows)
        if parquet_path.exists():
            parquet_path.unlink()

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

    @staticmethod
    def _build_query(params: Mapping[str, Any]) -> str:
        parts: List[str] = []
        for key, value in params.items():
            if value is None:
                continue
            if not _KEY_RE.match(key):
                raise ValueError(f"Invalid OpenF1 parameter name: {key!r}")
            if isinstance(value, bool):
                value = str(value).lower()
            text = quote(str(value), safe=":-._~")
            if key[-1] in "<>=":
                parts.append(f"{key}{text}")  # operator filters: date>=..., speed>=...
            else:
                parts.append(f"{key}={text}")
        return "&".join(parts)

    def _fetch_json(self, url: str, *, empty_on_404: bool) -> Any:
        attempt = 0
        while True:
            self._throttle()
            self.http_calls += 1
            try:
                response = self.session.get(
                    url,
                    headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise OpenF1Error(f"Network error calling {url}: {exc}") from exc
                delay = self._backoff_delay(attempt, None)
                LOGGER.warning("OpenF1 network error (%s); retry in %.1fs", exc, delay)
                self._sleep(delay)
                attempt += 1
                continue

            status = response.status_code
            if status == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise OpenF1Error(f"Invalid JSON from {url}") from exc
            if status == 404 and empty_on_404:
                return []  # OpenF1 answers 404 "No results found" for empty queries
            if status in _RETRY_STATUSES and attempt < self.max_retries:
                delay = self._backoff_delay(attempt, response.headers.get("Retry-After"))
                LOGGER.warning("OpenF1 HTTP %s for %s; retry in %.1fs", status, url, delay)
                self._sleep(delay)
                attempt += 1
                continue
            raise OpenF1Error(f"HTTP {status} for {url}: {response.text[:200]}")

    # ------------------------------------------------------------------ #
    # Generic access
    # ------------------------------------------------------------------ #
    def get(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        ttl_s: Any = _UNSET,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """GET one OpenF1 endpoint and return a list of row dictionaries."""
        if endpoint not in ENDPOINTS:
            raise ValueError(f"Unknown OpenF1 endpoint: {endpoint!r}")
        query: Dict[str, Any] = {k: v for k, v in (params or {}).items() if v is not None}
        if "country_code" in query:
            raise ValueError("country_code is deprecated in the 2026 OpenF1 API; do not query it")

        if ttl_s is _UNSET:
            ttl_s = self._default_ttl(endpoint, query)
        directory = self._session_cache_dir(query.get("session_key"))
        stem = self._cache_stem(endpoint, query)

        rows: Optional[List[Dict[str, Any]]] = None
        if use_cache:
            rows = self._read_cache(directory, stem, ttl_s)
        if rows is None:
            url = f"{self.base_url}/{endpoint}"
            qs = self._build_query(query)
            if qs:
                url = f"{url}?{qs}"
            payload = self._fetch_json(url, empty_on_404=True)
            if isinstance(payload, dict):
                detail = str(payload.get("detail", "")).lower()
                if "no results" in detail:
                    payload = []
                else:
                    raise OpenF1Error(f"Unexpected OpenF1 payload for {endpoint}: {payload}")
            if not isinstance(payload, list):
                raise OpenF1Error(f"Unexpected OpenF1 payload type for {endpoint}")
            rows = payload
            self._write_cache(directory, stem, endpoint, rows)

        if endpoint == "car_data":
            for row in rows:
                row.setdefault("drs", None)  # nullable in 2026 sessions
        return rows

    def get_dataframe(self, endpoint: str, params: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        """Same as :meth:`get` but returns a ``pandas.DataFrame``."""
        import pandas as pd

        return pd.DataFrame.from_records(self.get(endpoint, params, **kwargs))

    # ------------------------------------------------------------------ #
    # Meetings / sessions
    # ------------------------------------------------------------------ #
    def get_meetings(
        self,
        year: Optional[int] = None,
        meeting_key: Optional[int] = None,
        include_cancelled: bool = False,
    ) -> List[Dict[str, Any]]:
        rows = self.get("meetings", {"year": year, "meeting_key": meeting_key})
        if not include_cancelled:
            rows = [m for m in rows if not _truthy(m.get("is_cancelled", False))]
        return sorted(rows, key=lambda m: str(m.get("date_start", "")))

    def list_races(self, year: int) -> List[Dict[str, Any]]:
        """Non-cancelled Grand Prix meetings of ``year`` (testing excluded)."""
        meetings = self.get_meetings(year=year)
        return [
            m
            for m in meetings
            if "testing" not in str(m.get("meeting_name", "")).lower()
            and "testing" not in str(m.get("meeting_official_name", "")).lower()
        ]

    def get_sessions(
        self,
        *,
        session_key: Optional[int] = None,
        meeting_key: Optional[int] = None,
        year: Optional[int] = None,
        session_name: Optional[str] = None,
        session_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = self.get(
            "sessions",
            {
                "session_key": session_key,
                "meeting_key": meeting_key,
                "year": year,
                "session_name": session_name,
                "session_type": session_type,
            },
        )
        return sorted(rows, key=lambda s: str(s.get("date_start", "")))

    def get_race_session(self, meeting_key: int) -> Optional[Dict[str, Any]]:
        sessions = self.get_sessions(meeting_key=meeting_key, session_name="Race")
        return sessions[0] if sessions else None

    def get_qualifying_session(self, meeting_key: int) -> Optional[Dict[str, Any]]:
        sessions = self.get_sessions(meeting_key=meeting_key, session_name="Qualifying")
        return sessions[0] if sessions else None

    # ------------------------------------------------------------------ #
    # Session-scoped endpoints
    # ------------------------------------------------------------------ #
    def _by_session(
        self, endpoint: str, session_key: int, **filters: Any
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"session_key": session_key}
        params.update(filters)
        return self.get(endpoint, params)

    def get_drivers(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("drivers", session_key)

    def get_starting_grid(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("starting_grid", session_key)

    def get_session_result(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("session_result", session_key)

    def get_laps(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._by_session("laps", session_key, driver_number=driver_number)

    def get_intervals(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("intervals", session_key)

    def get_position(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("position", session_key)

    def get_pit(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("pit", session_key)

    def get_stints(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("stints", session_key)

    def get_weather(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("weather", session_key)

    def get_race_control(self, session_key: int) -> List[Dict[str, Any]]:
        return self._by_session("race_control", session_key)

    def get_car_data(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        date_start: Optional[Any] = None,
        date_end: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Car telemetry; ``drs`` is always present and nullable."""
        return self._by_session(
            "car_data",
            session_key,
            driver_number=driver_number,
            **{
                "date>=": None if date_start is None else parse_iso_datetime(date_start).isoformat(),
                "date<": None if date_end is None else parse_iso_datetime(date_end).isoformat(),
            },
        )

    def get_location(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        date_start: Optional[Any] = None,
        date_end: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return self._by_session(
            "location",
            session_key,
            driver_number=driver_number,
            **{
                "date>=": None if date_start is None else parse_iso_datetime(date_start).isoformat(),
                "date<": None if date_end is None else parse_iso_datetime(date_end).isoformat(),
            },
        )

    def get_telemetry_chunked(
        self,
        endpoint: str,
        session_key: int,
        driver_number: int,
        start: Any,
        end: Any,
        chunk_s: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Download ``car_data``/``location`` for one driver in time windows."""
        if endpoint not in {"car_data", "location"}:
            raise ValueError("Chunked download supports car_data and location only")
        step = timedelta(seconds=int(chunk_s or config.OPENF1_TELEMETRY_CHUNK_S))
        cursor = parse_iso_datetime(start)
        stop = parse_iso_datetime(end)
        rows: List[Dict[str, Any]] = []
        while cursor < stop:
            window_end = min(cursor + step, stop)
            rows.extend(
                self.get(
                    endpoint,
                    {
                        "session_key": session_key,
                        "driver_number": driver_number,
                        "date>=": cursor.isoformat(),
                        "date<": window_end.isoformat(),
                    },
                )
            )
            cursor = window_end
        return rows

    # ------------------------------------------------------------------ #
    # Circuit orientation
    # ------------------------------------------------------------------ #
    def get_circuit_info(self, meeting_key: int) -> Optional[Dict[str, Any]]:
        """Fetch ``meetings[0]["circuit_info_url"]`` and cache the JSON."""
        meetings = self.get_meetings(meeting_key=meeting_key, include_cancelled=True)
        exact = [m for m in meetings if str(m.get("meeting_key")) == str(meeting_key)]
        meetings = exact or meetings
        if not meetings:
            return None
        url = meetings[0].get("circuit_info_url")
        if not url:
            return None
        path = self._session_cache_dir(None) / f"circuit_info__meeting-{int(meeting_key)}.json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                if isinstance(cached, dict):
                    return cached
            except (OSError, ValueError):
                pass
        try:
            payload = self._fetch_json(str(url), empty_on_404=False)
        except OpenF1Error as exc:
            LOGGER.warning("Could not fetch circuit info for meeting %s: %s", meeting_key, exc)
            return None
        if not isinstance(payload, dict):
            return None
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp, path)
        return payload

    def get_circuit_rotation(self, meeting_key: int) -> Optional[float]:
        """Track-map rotation angle in degrees, or ``None`` if unavailable."""
        info = self.get_circuit_info(meeting_key)
        if not info:
            return None
        try:
            return float(info["rotation"])
        except (KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # Bulk download
    # ------------------------------------------------------------------ #
    def prefetch_session(
        self,
        session_key: int,
        endpoints: Iterable[str] = DEFAULT_PREFETCH_ENDPOINTS,
        telemetry: bool = False,
    ) -> Dict[str, int]:
        """Download (and cache) everything a replay needs for one session.

        Returns ``{endpoint: row_count}``.  With ``telemetry=True`` the full
        ``car_data`` and ``location`` streams are downloaded per driver in
        time windows, which is slow on the free tier but only happens once.
        """
        counts: Dict[str, int] = {}
        for endpoint in endpoints:
            counts[endpoint] = len(self._by_session(endpoint, session_key))
        if telemetry:
            sessions = self.get_sessions(session_key=session_key)
            if not sessions:
                return counts
            start = sessions[0].get("date_start")
            end = sessions[0].get("date_end")
            if not start or not end:
                return counts
            drivers = self.get_drivers(session_key)
            for kind in ("car_data", "location"):
                total = 0
                for driver in drivers:
                    number = driver.get("driver_number")
                    if number is None:
                        continue
                    total += len(
                        self.get_telemetry_chunked(kind, session_key, int(number), start, end)
                    )
                counts[kind] = total
        return counts
