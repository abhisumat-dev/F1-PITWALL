"""Fuel-corrected tyre degradation tracker for live and replayed F1 stints.

Rules enforced per Stage 2 spec:
- Excludes Lap 1 (standing start / traffic distortion).
- Excludes pit-in laps and pit-out laps (``is_pit_out_lap`` or ``is_pit_in_lap``).
- Excludes non-green track flag laps (``YELLOW``, ``SC``, ``VSC``, ``RED``).
- Applies a fuel-burn correction of ``-0.035 s`` per lap burned so lighter
  fuel later in the race does not mask real tyre wear:
      corrected_lap_time = lap_time_s + 0.035 * (lap_number - 1)
  (equivalently subtracting ``0.035 * laps_remaining``, which gives the exact
  same positive slope shift ``+0.035 s/lap`` over increasing lap numbers).
- Fits an ordinary least-squares linear slope (``sec/lap``) over the last
  4 to 8 clean laps of the current stint.
- Classifies ``tyre_health``:
    * ``OPTIMAL``   : slope < 0.08 s/lap
    * ``DEGRADING`` : 0.08 <= slope <= 0.16 s/lap
    * ``CLIFF``     : slope > 0.16 s/lap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import config

__all__ = [
    "CleanLapSample",
    "TyreDegResult",
    "TyreDegTracker",
    "classify_tyre_health",
    "NON_GREEN_FLAGS",
]

NON_GREEN_FLAGS = frozenset({"YELLOW", "SC", "VSC", "RED", "CHEQUERED"})


def classify_tyre_health(slope_s_per_lap: Optional[float]) -> str:
    """Classify tyre degradation slope (s/lap) into OPTIMAL, DEGRADING, CLIFF, or UNKNOWN."""
    if slope_s_per_lap is None or not np.isfinite(slope_s_per_lap):
        return "UNKNOWN"
    if slope_s_per_lap < 0.08:
        return "OPTIMAL"
    if slope_s_per_lap <= 0.16:
        return "DEGRADING"
    return "CLIFF"


@dataclass(frozen=True)
class CleanLapSample:
    lap_number: int
    stint_number: int
    raw_lap_time_s: float
    corrected_lap_time_s: float


@dataclass
class TyreDegResult:
    slope_s_per_lap: Optional[float]
    tyre_health: str
    fuel_corrected: bool
    window_laps: int
    sample_laps: int
    stint_number: Optional[int]
    r_squared: Optional[float]

    def to_contract(self) -> Dict[str, Any]:
        return {
            "slope_s_per_lap": (
                round(float(self.slope_s_per_lap), 4)
                if self.slope_s_per_lap is not None and np.isfinite(self.slope_s_per_lap)
                else None
            ),
            "tyre_health": self.tyre_health,
            "fuel_corrected": bool(self.fuel_corrected),
            "window_laps": int(self.window_laps),
            "sample_laps": int(self.sample_laps),
            "stint_number": self.stint_number,
            "r_squared": (
                round(float(self.r_squared), 4)
                if self.r_squared is not None and np.isfinite(self.r_squared)
                else None
            ),
        }


@dataclass
class _DriverStintBuffer:
    stint_number: int = 1
    samples: List[CleanLapSample] = field(default_factory=list)
    seen_laps: set[int] = field(default_factory=set)


class TyreDegTracker:
    """Tracks clean fuel-corrected lap times per driver stint and fits degradation slope."""

    def __init__(
        self,
        *,
        min_laps: int = config.TYRE_DEG_MIN_LAPS,
        window_laps: int = config.TYRE_DEG_WINDOW_LAPS,
        fuel_correction_s_per_lap: float = config.FUEL_BURN_CORRECTION_S_PER_LAP,
        total_laps: Optional[int] = None,
    ) -> None:
        self.min_laps = max(2, int(min_laps))
        self.window_laps = max(self.min_laps, int(window_laps))
        self.fuel_correction_s_per_lap = float(fuel_correction_s_per_lap)
        self.total_laps = int(total_laps) if total_laps is not None and total_laps > 0 else None
        self._buffers: Dict[int, _DriverStintBuffer] = {}

    def reset(self) -> None:
        self._buffers.clear()

    def set_total_laps(self, total_laps: Optional[int]) -> None:
        if total_laps is not None and int(total_laps) > 0:
            self.total_laps = int(total_laps)

    def _fuel_corrected_time(self, lap_number: int, lap_time_s: float) -> float:
        """Apply fuel-burn correction (``-0.035s`` per lap burned -> normalize to lap 1 fuel).

        Each lap burned makes the car ~0.035s faster purely from burning fuel.
        Adding ``+0.035 * (lap_number - 1)`` (equivalently subtracting
        ``0.035 * remaining_fuel_laps`` from a baseline) removes the artificial
        speedup so the linear slope measures true tyre degradation.
        """
        burned_laps = max(0, int(lap_number) - 1)
        return float(lap_time_s) + (self.fuel_correction_s_per_lap * burned_laps)

    @staticmethod
    def _is_green_flag(track_status: Optional[str]) -> bool:
        if track_status is None:
            return True
        status = str(track_status).strip().upper()
        if not status or status in {"GREEN", "ALLCLEAR", "CLEAR", "1"}:
            return True
        return status not in NON_GREEN_FLAGS

    def record_lap(
        self,
        driver_number: int,
        lap_number: int,
        lap_time_s: Optional[float],
        *,
        stint_number: int = 1,
        is_pit_out_lap: bool = False,
        is_pit_in_lap: bool = False,
        track_status: Optional[str] = "GREEN",
    ) -> Optional[TyreDegResult]:
        """Ingest one completed lap for ``driver_number`` and return updated degradation state."""
        driver_number = int(driver_number)
        lap_number = int(lap_number)
        stint_number = max(1, int(stint_number or 1))

        buf = self._buffers.get(driver_number)
        if buf is None or buf.stint_number != stint_number:
            buf = _DriverStintBuffer(stint_number=stint_number)
            self._buffers[driver_number] = buf

        # Exclusion checks: Lap 1, pit-in/pit-out laps, non-green flag laps, invalid times
        if lap_number <= 1:
            return self.get_driver_deg(driver_number)
        if is_pit_out_lap or is_pit_in_lap:
            return self.get_driver_deg(driver_number)
        if not self._is_green_flag(track_status):
            return self.get_driver_deg(driver_number)
        if lap_time_s is None or not np.isfinite(lap_time_s) or lap_time_s <= 0:
            return self.get_driver_deg(driver_number)
        if lap_number in buf.seen_laps:
            return self.get_driver_deg(driver_number)

        # Reject extreme outlier laps (e.g., spin/traffic > 18% off current stint median)
        if len(buf.samples) >= 2:
            median_raw = float(np.median([s.raw_lap_time_s for s in buf.samples]))
            if lap_time_s > median_raw * 1.18 or lap_time_s < median_raw * 0.82:
                buf.seen_laps.add(lap_number)
                return self.get_driver_deg(driver_number)

        corrected = self._fuel_corrected_time(lap_number, float(lap_time_s))
        buf.samples.append(
            CleanLapSample(
                lap_number=lap_number,
                stint_number=stint_number,
                raw_lap_time_s=float(lap_time_s),
                corrected_lap_time_s=corrected,
            )
        )
        buf.seen_laps.add(lap_number)
        buf.samples.sort(key=lambda s: s.lap_number)
        return self.get_driver_deg(driver_number)

    def get_driver_deg(self, driver_number: int) -> Optional[TyreDegResult]:
        """Compute fuel-corrected degradation slope over the last 4-8 clean stint laps."""
        buf = self._buffers.get(int(driver_number))
        if buf is None or not buf.samples:
            return None

        recent = buf.samples[-self.window_laps :]
        n = len(recent)
        if n < self.min_laps:
            return TyreDegResult(
                slope_s_per_lap=None,
                tyre_health="UNKNOWN",
                fuel_corrected=True,
                window_laps=self.window_laps,
                sample_laps=n,
                stint_number=buf.stint_number,
                r_squared=None,
            )

        x = np.asarray([s.lap_number for s in recent], dtype=float)
        y = np.asarray([s.corrected_lap_time_s for s in recent], dtype=float)
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        dx = x - x_mean
        denom = float(np.dot(dx, dx))
        if denom <= 0:
            slope = 0.0
            r_sq: Optional[float] = 0.0
        else:
            dy = y - y_mean
            slope = float(np.dot(dx, dy) / denom)
            ss_tot = float(np.dot(dy, dy))
            if ss_tot <= 1e-12:
                r_sq = 1.0
            else:
                y_hat = y_mean + slope * dx
                ss_res = float(np.dot(y - y_hat, y - y_hat))
                r_sq = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

        health = classify_tyre_health(slope)
        return TyreDegResult(
            slope_s_per_lap=slope,
            tyre_health=health,
            fuel_corrected=True,
            window_laps=self.window_laps,
            sample_laps=n,
            stint_number=buf.stint_number,
            r_squared=r_sq,
        )

    def get_clean_pace_3(self, driver_number: int) -> Optional[float]:
        """Return rolling 3-lap clean raw lap-time average (for ML live inference)."""
        buf = self._buffers.get(int(driver_number))
        if buf is None or not buf.samples:
            return None
        recent = buf.samples[-3:]
        return float(np.mean([s.raw_lap_time_s for s in recent]))

    def bulk_ingest_openf1_laps(
        self,
        laps: Sequence[Mapping[str, Any]],
        stints: Sequence[Mapping[str, Any]],
        pit_laps_by_driver: Mapping[int, set[int]],
        track_status_fn: Any,
    ) -> None:
        """Helper to feed historical laps (used by replay seek/rebuild)."""
        self.reset()
        stint_map: Dict[int, List[Mapping[str, Any]]] = {}
        for s in stints or []:
            dn = s.get("driver_number")
            if dn is None:
                continue
            stint_map.setdefault(int(dn), []).append(s)
        for lst in stint_map.values():
            lst.sort(key=lambda item: int(item.get("lap_start") or 1))

        def _stint_for(dn: int, lap_no: int) -> int:
            chosen = 1
            for st in stint_map.get(dn, []):
                start = int(st.get("lap_start") or 1)
                if start <= lap_no:
                    chosen = int(st.get("stint_number") or 1)
                else:
                    break
            return chosen

        sorted_laps = sorted(
            laps,
            key=lambda l: (int(l.get("lap_number") or 0), int(l.get("driver_number") or 0)),
        )
        for row in sorted_laps:
            dn = row.get("driver_number")
            lap_no = row.get("lap_number")
            if dn is None or lap_no is None:
                continue
            dn_i = int(dn)
            lap_i = int(lap_no)
            dur = row.get("lap_duration")
            pit_out = bool(row.get("is_pit_out_lap"))
            pit_in = lap_i in pit_laps_by_driver.get(dn_i, set())
            t_status = track_status_fn(row) if callable(track_status_fn) else "GREEN"
            self.record_lap(
                dn_i,
                lap_i,
                float(dur) if dur is not None else None,
                stint_number=_stint_for(dn_i, lap_i),
                is_pit_out_lap=pit_out,
                is_pit_in_lap=pit_in,
                track_status=t_status,
            )
