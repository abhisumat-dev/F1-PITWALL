"""Sector 1 / 2 / 3 fastest times and overall fastest lap tracker.

Tracks both per-driver latest/best sector times (with PURPLE / GREEN / YELLOW
telemetry status) and session-wide overall fastest Sector 1, Sector 2, Sector 3,
and Fastest Lap (including who set each and on which lap).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "BestRecord",
    "DriverSectorState",
    "SectorAndLapTracker",
    "format_sector_time",
    "format_lap_clock",
]


def format_sector_time(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return None
    return f"{float(seconds):.3f}"


def format_lap_clock(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return None
    total = float(seconds)
    mins = int(total // 60)
    rem = total - mins * 60
    if mins > 0:
        return f"{mins}:{rem:06.3f}"
    return f"{rem:.3f}"


@dataclass
class BestRecord:
    driver_number: int
    driver_code: str
    team_name: Optional[str]
    team_colour: Optional[str]
    time_s: float
    formatted: str
    lap_number: Optional[int] = None

    def to_contract(self) -> Dict[str, Any]:
        return {
            "driver_number": int(self.driver_number),
            "driver_code": str(self.driver_code),
            "team_name": self.team_name,
            "team_colour": self.team_colour,
            "time_s": round(float(self.time_s), 3),
            "formatted": self.formatted,
            "lap_number": self.lap_number,
        }


@dataclass
class DriverSectorState:
    last_s1_s: Optional[float] = None
    last_s2_s: Optional[float] = None
    last_s3_s: Optional[float] = None
    best_s1_s: Optional[float] = None
    best_s2_s: Optional[float] = None
    best_s3_s: Optional[float] = None
    best_lap_s: Optional[float] = None
    best_lap_number: Optional[int] = None
    seen_laps: set[int] = field(default_factory=set)


class SectorAndLapTracker:
    """Maintains live/replay S1, S2, S3 fastest records and overall fastest lap."""

    def __init__(self) -> None:
        self._drivers: Dict[int, DriverSectorState] = {}
        self.fastest_lap: Optional[BestRecord] = None
        self.fastest_s1: Optional[BestRecord] = None
        self.fastest_s2: Optional[BestRecord] = None
        self.fastest_s3: Optional[BestRecord] = None

    def reset(self) -> None:
        self._drivers.clear()
        self.fastest_lap = None
        self.fastest_s1 = None
        self.fastest_s2 = None
        self.fastest_s3 = None

    @staticmethod
    def _valid(val: Optional[float], min_s: float = 8.0, max_s: float = 300.0) -> Optional[float]:
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f) or f < min_s or f > max_s:
            return None
        return f

    def record_lap(
        self,
        driver_number: int,
        driver_code: str,
        *,
        lap_number: Optional[int] = None,
        lap_time_s: Optional[float] = None,
        s1_s: Optional[float] = None,
        s2_s: Optional[float] = None,
        s3_s: Optional[float] = None,
        team_name: Optional[str] = None,
        team_colour: Optional[str] = None,
    ) -> None:
        dn = int(driver_number)
        st = self._drivers.setdefault(dn, DriverSectorState())
        if lap_number is not None and int(lap_number) in st.seen_laps:
            return
        if lap_number is not None:
            st.seen_laps.add(int(lap_number))

        v_s1 = self._valid(s1_s, min_s=8.0, max_s=120.0)
        v_s2 = self._valid(s2_s, min_s=8.0, max_s=120.0)
        v_s3 = self._valid(s3_s, min_s=8.0, max_s=120.0)
        v_lap = self._valid(lap_time_s, min_s=40.0, max_s=240.0)

        if v_s1 is not None:
            st.last_s1_s = v_s1
            if st.best_s1_s is None or v_s1 < st.best_s1_s:
                st.best_s1_s = v_s1
            if self.fastest_s1 is None or v_s1 < self.fastest_s1.time_s:
                self.fastest_s1 = BestRecord(
                    driver_number=dn,
                    driver_code=driver_code,
                    team_name=team_name,
                    team_colour=team_colour,
                    time_s=v_s1,
                    formatted=format_sector_time(v_s1) or "",
                    lap_number=lap_number,
                )

        if v_s2 is not None:
            st.last_s2_s = v_s2
            if st.best_s2_s is None or v_s2 < st.best_s2_s:
                st.best_s2_s = v_s2
            if self.fastest_s2 is None or v_s2 < self.fastest_s2.time_s:
                self.fastest_s2 = BestRecord(
                    driver_number=dn,
                    driver_code=driver_code,
                    team_name=team_name,
                    team_colour=team_colour,
                    time_s=v_s2,
                    formatted=format_sector_time(v_s2) or "",
                    lap_number=lap_number,
                )

        if v_s3 is not None:
            st.last_s3_s = v_s3
            if st.best_s3_s is None or v_s3 < st.best_s3_s:
                st.best_s3_s = v_s3
            if self.fastest_s3 is None or v_s3 < self.fastest_s3.time_s:
                self.fastest_s3 = BestRecord(
                    driver_number=dn,
                    driver_code=driver_code,
                    team_name=team_name,
                    team_colour=team_colour,
                    time_s=v_s3,
                    formatted=format_sector_time(v_s3) or "",
                    lap_number=lap_number,
                )

        if v_lap is not None:
            if st.best_lap_s is None or v_lap < st.best_lap_s:
                st.best_lap_s = v_lap
                st.best_lap_number = lap_number
            if self.fastest_lap is None or v_lap < self.fastest_lap.time_s:
                self.fastest_lap = BestRecord(
                    driver_number=dn,
                    driver_code=driver_code,
                    team_name=team_name,
                    team_colour=team_colour,
                    time_s=v_lap,
                    formatted=format_lap_clock(v_lap) or "",
                    lap_number=lap_number,
                )

    def update_live_sectors(
        self,
        driver_number: int,
        driver_code: str,
        *,
        s1_s: Optional[float] = None,
        s2_s: Optional[float] = None,
        s3_s: Optional[float] = None,
        best_lap_s: Optional[float] = None,
        best_lap_number: Optional[int] = None,
        team_name: Optional[str] = None,
        team_colour: Optional[str] = None,
    ) -> None:
        """Update from SignalR live rolling sector/best-lap fields (idempotent)."""
        dn = int(driver_number)
        st = self._drivers.setdefault(dn, DriverSectorState())
        for idx, (raw_val, attr_last, attr_best, overall_attr) in enumerate(
            [
                (s1_s, "last_s1_s", "best_s1_s", "fastest_s1"),
                (s2_s, "last_s2_s", "best_s2_s", "fastest_s2"),
                (s3_s, "last_s3_s", "best_s3_s", "fastest_s3"),
            ],
            start=1,
        ):
            val = self._valid(raw_val, min_s=8.0, max_s=120.0)
            if val is None:
                continue
            setattr(st, attr_last, val)
            cur_best = getattr(st, attr_best)
            if cur_best is None or val < cur_best:
                setattr(st, attr_best, val)
            overall: Optional[BestRecord] = getattr(self, overall_attr)
            if overall is None or val < overall.time_s:
                setattr(
                    self,
                    overall_attr,
                    BestRecord(
                        driver_number=dn,
                        driver_code=driver_code,
                        team_name=team_name,
                        team_colour=team_colour,
                        time_s=val,
                        formatted=format_sector_time(val) or "",
                        lap_number=best_lap_number,
                    ),
                )

        v_best = self._valid(best_lap_s, min_s=40.0, max_s=240.0)
        if v_best is not None:
            if st.best_lap_s is None or v_best < st.best_lap_s:
                st.best_lap_s = v_best
                st.best_lap_number = best_lap_number
            if self.fastest_lap is None or v_best < self.fastest_lap.time_s:
                self.fastest_lap = BestRecord(
                    driver_number=dn,
                    driver_code=driver_code,
                    team_name=team_name,
                    team_colour=team_colour,
                    time_s=v_best,
                    formatted=format_lap_clock(v_best) or "",
                    lap_number=best_lap_number,
                )

    @staticmethod
    def _sector_colour(
        last_val: Optional[float],
        personal_best: Optional[float],
        overall_best: Optional[BestRecord],
    ) -> str:
        if last_val is None:
            return "NONE"
        if overall_best is not None and last_val <= overall_best.time_s + 1e-4:
            return "PURPLE"
        if personal_best is not None and last_val <= personal_best + 1e-4:
            return "GREEN"
        return "YELLOW"

    def driver_sectors_contract(self, driver_number: int) -> Dict[str, Any]:
        st = self._drivers.get(int(driver_number))
        if st is None:
            return {
                "s1_s": None,
                "s2_s": None,
                "s3_s": None,
                "s1_status": "NONE",
                "s2_status": "NONE",
                "s3_status": "NONE",
            }
        return {
            "s1_s": round(st.last_s1_s, 3) if st.last_s1_s is not None else None,
            "s2_s": round(st.last_s2_s, 3) if st.last_s2_s is not None else None,
            "s3_s": round(st.last_s3_s, 3) if st.last_s3_s is not None else None,
            "s1_status": self._sector_colour(st.last_s1_s, st.best_s1_s, self.fastest_s1),
            "s2_status": self._sector_colour(st.last_s2_s, st.best_s2_s, self.fastest_s2),
            "s3_status": self._sector_colour(st.last_s3_s, st.best_s3_s, self.fastest_s3),
        }

    def session_best_contract(self) -> Dict[str, Any]:
        return {
            "fastest_lap": self.fastest_lap.to_contract() if self.fastest_lap else None,
            "fastest_s1": self.fastest_s1.to_contract() if self.fastest_s1 else None,
            "fastest_s2": self.fastest_s2.to_contract() if self.fastest_s2 else None,
            "fastest_s3": self.fastest_s3.to_contract() if self.fastest_s3 else None,
        }
