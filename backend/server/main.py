"""FastAPI server & WebSocket hub for F1 Pit-Wall ($0 Architecture).

Run from workspace root::

    uvicorn backend.server.main:app --host 0.0.0.0 --port 8000

Endpoints:
- ``WS  /ws/live``                : Streams live/replay packets & post-quali broadcasts.
- ``GET /api/health``             : Server health, active mode, provider & ML model status.
- ``GET /api/sessions``           : Non-cancelled 2023-2026 OpenF1 Race sessions.
- ``GET /api/next-race``          : Next upcoming GP dates & times in IST, UK, and US timezones.
- ``POST /api/predict/post-quali``: Saturday Top-3 Podium prediction (300% sum) + WS broadcast.
- ``POST /api/mode``              : Switch active mode between ``"replay"`` and ``"free_live"``.
- ``POST /api/replay/load``       : Load & cache a 2023-2026 OpenF1 race session.
- ``POST /api/replay/control``    : ``play``, ``pause``, ``set_speed`` (1x-10x), ``seek_lap``.
- ``GET /``                       : Serves ``frontend/`` PWA when present.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

# Ensure backend/ is importable whether launched as `backend.server.main:app` or `server.main:app`
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from analytics.radio_transcriber import RadioTranscriber
from clients.jolpica_client import JolpicaClient, JolpicaError
from clients.openf1_client import OpenF1Client, OpenF1Error
from ml.features import normalize_podium_to_percent, normalize_to_percent
from ml.predictor import ModelNotAvailableError, PredictionError, RacePredictor
from server.circuits import list_circuits_summary, resolve_circuit_2d
from server.race_results import get_final_race_results
from server.schedule import get_next_race_schedule
from server.standings import get_pre_race_standings
from streams.base_provider import BaseStreamProvider
from streams.replay_engine import ReplayEngine
from streams.signalr_live import SignalRLiveProvider

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #
class ModeRequest(BaseModel):
    mode: Literal["replay", "free_live"]


class ReplayLoadRequest(BaseModel):
    session_key: Optional[int] = Field(default=None, description="OpenF1 race session_key")
    season: Optional[int] = Field(default=2026, ge=2023, le=2030)
    round: Optional[int] = Field(default=None, ge=1, le=30)
    grand_prix: Optional[str] = Field(default=None, description="Substring match e.g. 'Las Vegas'")
    include_telemetry: bool = Field(default=True)
    telemetry_window_s: Optional[int] = Field(default=None, ge=60)
    autoplay: bool = Field(default=True)


class ReplayControlRequest(BaseModel):
    action: Literal["play", "pause", "set_speed", "seek_lap"]
    speed: Optional[float] = Field(default=None, ge=0.5, le=20.0)
    lap: Optional[int] = Field(default=None, ge=1, le=200)


class PostQualiRequest(BaseModel):
    season: Optional[int] = Field(default=None, ge=2018, le=2030)
    round: Optional[int] = Field(default=None, ge=1, le=30)


# --------------------------------------------------------------------------- #
# Runtime coordinator (manages providers + WebSocket broadcast hub)
# --------------------------------------------------------------------------- #
class PitWallRuntime:
    """Coordinates active stream provider, ML predictor, and WebSocket fan-out."""

    def __init__(
        self,
        *,
        jolpica: Optional[JolpicaClient] = None,
        openf1: Optional[OpenF1Client] = None,
        predictor: Optional[RacePredictor] = None,
        transcriber: Optional[RadioTranscriber] = None,
        replay: Optional[ReplayEngine] = None,
        live: Optional[SignalRLiveProvider] = None,
        initial_mode: Literal["replay", "free_live"] = "replay",
    ) -> None:
        config.ensure_directories()
        self.jolpica = jolpica or JolpicaClient()
        self.openf1 = openf1 or OpenF1Client()
        self.predictor = predictor or RacePredictor(jolpica=self.jolpica, openf1=self.openf1)
        self.transcriber = transcriber or RadioTranscriber()
        self.replay = replay or ReplayEngine(
            openf1=self.openf1, predictor=self.predictor, transcriber=self.transcriber
        )
        self.live = live or SignalRLiveProvider(
            predictor=self.predictor, transcriber=self.transcriber
        )

        self.mode: Literal["replay", "free_live"] = initial_mode
        self._clients: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._latest_packet: Optional[Dict[str, Any]] = None
        self._last_post_quali: Optional[Dict[str, Any]] = None
        self._pump_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    @property
    def active_provider(self) -> BaseStreamProvider:
        return self.replay if self.mode == "replay" else self.live

    async def start(self) -> None:
        await self.active_provider.start()
        self._start_pump()

    async def stop(self) -> None:
        await self._stop_pump()
        await self.replay.stop()
        await self.live.stop()

    def _start_pump(self) -> None:
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
        provider = self.active_provider
        self._pump_task = asyncio.create_task(self._pump_provider(provider), name=f"ws-pump-{provider.mode}")

    async def _stop_pump(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None

    async def _pump_provider(self, provider: BaseStreamProvider) -> None:
        queue = provider.subscribe(maxsize=8)
        try:
            while True:
                packet = await queue.get()
                await self.broadcast(packet)
        except asyncio.CancelledError:
            raise
        finally:
            provider.unsubscribe(queue)

    def register_client(self) -> asyncio.Queue[Dict[str, Any]]:
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=16)
        self._clients.add(q)
        if self._latest_packet is not None:
            try:
                q.put_nowait(self._latest_packet)
            except asyncio.QueueFull:
                pass
        return q

    def unregister_client(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        self._clients.discard(queue)

    async def broadcast(self, packet: Dict[str, Any]) -> None:
        self._latest_packet = packet
        dead: List[asyncio.Queue[Dict[str, Any]]] = []
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(packet)
            except Exception:  # noqa: BLE001
                dead.append(q)
        for q in dead:
            self._clients.discard(q)

    async def switch_mode(self, new_mode: Literal["replay", "free_live"]) -> Dict[str, Any]:
        async with self._lock:
            if new_mode == self.mode and self.active_provider.running:
                if self.active_provider.latest_packet is not None:
                    await self.broadcast(self.active_provider.latest_packet)
                return self.health()

            prev = self.active_provider
            await self._stop_pump()
            await prev.stop()

            self.mode = new_mode
            nxt = self.active_provider
            await nxt.start()
            self._start_pump()

            if nxt.latest_packet is not None:
                await self.broadcast(nxt.latest_packet)
            elif new_mode == "free_live":
                pkt = self.live.build_packet()
                await self.broadcast(pkt)
            elif new_mode == "replay" and self.replay.status()["loaded"]:
                pkt = self.replay.build_frame()
                await self.broadcast(pkt)
            return self.health()

    def _fallback_post_quali_packet(
        self,
        season: Optional[int],
        round_number: Optional[int],
    ) -> Dict[str, Any]:
        """Contract-valid Top-3 Podium fallback when joblib artifacts are not yet trained."""
        entries_raw: List[Dict[str, Any]] = []
        gp_name = "Upcoming Grand Prix"
        circuit_id: Optional[str] = None
        circuit_name: Optional[str] = "Circuit"
        rotation_deg: Optional[float] = None
        grid_source = "jolpica_qualifying"
        resolved_season = int(season or config.current_season())
        resolved_round = int(round_number or 1)
        meeting_key: Optional[int] = None
        session_key: Optional[int] = None

        # 1. Try Jolpica qualifying / race rows first
        try:
            if season is None or round_number is None:
                resolved_season, resolved_round = self.predictor.latest_qualified_round(season)
            from ml.ingest_history import fetch_race_rows

            rows = fetch_race_rows(self.jolpica, resolved_season, resolved_round)
            if rows:
                first = rows[0]
                gp_name = str(first.get("race_name") or f"Round {resolved_round}")
                circuit_id = first.get("circuit_id")
                circuit_name = first.get("circuit_name") or "Circuit"
                if any(bool(r.get("completed")) for r in rows):
                    grid_source = "jolpica_results"
                for idx, row in enumerate(rows, start=1):
                    grid_pos = int(row.get("grid") or row.get("quali_position") or idx)
                    if grid_pos <= 0:
                        grid_pos = len(rows)
                    entries_raw.append(
                        {
                            "driver_number": int(row.get("driver_number") or idx),
                            "driver_code": str(row.get("driver_code") or f"D{idx:02d}"),
                            "full_name": str(row.get("driver_name") or row.get("driver_code") or f"Driver {idx}"),
                            "team_name": str(row.get("constructor_name") or "Unknown Team"),
                            "grid_position": grid_pos,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("Jolpica fallback grid unavailable (%s); checking active replay/live state", exc)

        # 2. Only fall back to active replay/live packet when no explicit round was requested
        if not entries_raw and round_number is None and self._latest_packet is not None:
            race = self._latest_packet.get("race") or {}
            drivers = race.get("drivers") or []
            if drivers:
                gp_name = str(self._latest_packet.get("grand_prix") or gp_name)
                circuit = self._latest_packet.get("circuit") or {}
                circuit_id = circuit.get("circuit_id")
                circuit_name = circuit.get("name") or circuit_name
                rotation_deg = circuit.get("rotation_deg")
                meeting_key = self._latest_packet.get("meeting_key")
                session_key = self._latest_packet.get("session_key")
                resolved_season = int(self._latest_packet.get("season") or resolved_season)
                resolved_round = int(self._latest_packet.get("round") or resolved_round)
                grid_source = "openf1_starting_grid"
                for idx, d in enumerate(drivers, start=1):
                    entries_raw.append(
                        {
                            "driver_number": int(d.get("driver_number") or idx),
                            "driver_code": str(d.get("code") or f"D{idx:02d}"),
                            "full_name": str(d.get("full_name") or d.get("code") or f"Driver {idx}"),
                            "team_name": str(d.get("team_name") or "Unknown Team"),
                            "team_colour": d.get("team_colour"),
                            "grid_position": int(d.get("grid_position") or d.get("position") or idx),
                        }
                    )

        if not entries_raw:
            raise PredictionError(
                "No qualifying data available from Jolpica and no replay session loaded."
            )

        entries_raw.sort(key=lambda e: (e["grid_position"], e["driver_number"]))
        raw_podium = [math.exp(-0.28 * (e["grid_position"] - 1)) for e in entries_raw]
        raw_win = [math.exp(-0.46 * (e["grid_position"] - 1)) for e in entries_raw]

        podium_pcts = normalize_podium_to_percent(raw_podium, decimals=config.PROBABILITY_DECIMALS, target_sum=300.0)
        win_pcts = normalize_to_percent(raw_win, decimals=config.PROBABILITY_DECIMALS)

        items: List[Dict[str, Any]] = []
        for entry, p_pct, w_pct in zip(entries_raw, podium_pcts, win_pcts):
            item: Dict[str, Any] = {
                "driver_number": entry["driver_number"],
                "driver_code": entry["driver_code"],
                "full_name": entry["full_name"],
                "team_name": entry["team_name"],
                "grid_position": max(1, entry["grid_position"]),
                "podium_prob_pct": float(p_pct),
                "win_prob_pct": float(w_pct),
                "win_probability_pct": float(w_pct),
            }
            if entry.get("team_colour"):
                item["team_colour"] = entry["team_colour"]
            items.append(item)

        items.sort(key=lambda e: (-e["podium_prob_pct"], -e["win_prob_pct"], e["grid_position"]))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank

        sum_podium = round(float(sum(i["podium_prob_pct"] for i in items)), config.PROBABILITY_DECIMALS)
        sum_win = round(float(sum(i["win_prob_pct"] for i in items)), config.PROBABILITY_DECIMALS)
        return {
            "schema_version": config.SCHEMA_VERSION,
            "type": "post_quali_prediction",
            "timestamp": _now_iso(),
            "sequence": 0,
            "meeting_key": meeting_key,
            "session_key": session_key,
            "season": resolved_season,
            "round": resolved_round,
            "grand_prix": gp_name,
            "circuit": {
                "circuit_id": circuit_id,
                "name": circuit_name,
                "rotation_deg": rotation_deg,
            },
            "grid_source": grid_source,
            "predictions": {
                "stage": "pre_race",
                "target": "top3_podium",
                "model_name": "grid-prior-fallback",
                "model_version": "pre_race-grid-fallback-v1",
                "trained_at": None,
                "generated_at": _now_iso(),
                "validation": {},
                "items": items,
                "sum_podium_pct": sum_podium,
                "sum_win_pct": sum_win,
                "sum_pct": sum_podium,
            },
        }

    def generate_post_quali(
        self,
        season: Optional[int] = None,
        round_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            packet = self.predictor.predict_post_quali(season=season, round_number=round_number)
        except (ModelNotAvailableError, PredictionError, JolpicaError) as exc:
            LOGGER.info("Using post-quali fallback (%s)", exc)
            packet = self._fallback_post_quali_packet(season, round_number)

        # Enrich circuit block with official 2D geometry if matched
        circ = packet.get("circuit") or {}
        c2d = resolve_circuit_2d(
            circuit_id=circ.get("circuit_id"),
            name=circ.get("name"),
            grand_prix=packet.get("grand_prix"),
        )
        if c2d:
            circ["slug"] = c2d.get("slug")
            if circ.get("rotation_deg") is None and c2d.get("rotation_deg") is not None:
                circ["rotation_deg"] = float(c2d["rotation_deg"])
            circ["track_outline"] = [{"x": float(p["x"]), "y": float(p["y"])} for p in (c2d.get("points") or [])] or None
            circ["sector_splits"] = [int(i) for i in (c2d.get("sector_splits") or [])] or None
            sf = c2d.get("start_finish")
            circ["start_finish"] = (
                {"x": float(sf["x"]), "y": float(sf["y"])}
                if isinstance(sf, dict) and "x" in sf and "y" in sf
                else None
            )
            circ["corners"] = [
                {
                    "number": int(c["number"]),
                    "letter": str(c.get("letter") or "") or None,
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                }
                for c in (c2d.get("corners") or [])
            ] or None
            packet["circuit"] = circ

        self._last_post_quali = packet
        return packet

    def health(self) -> Dict[str, Any]:
        is_serverless = bool(os.environ.get("VERCEL"))
        return {
            "status": "ok",
            "schema_version": config.SCHEMA_VERSION,
            "server_time_utc": _now_iso(),
            "serverless": is_serverless,
            "ws_supported": not is_serverless,
            "mode": self.mode,
            "ws_clients": len(self._clients),
            "active_provider": self.active_provider.status(),
            "replay": self.replay.status(),
            "free_live": self.live.status(),
            "models": self.predictor.model_info(),
        }


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    runtime: Optional[PitWallRuntime] = None,
    *,
    mount_frontend: bool = True,
) -> FastAPI:
    rt = runtime or PitWallRuntime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await rt.start()
        try:
            yield
        finally:
            await rt.stop()

    app = FastAPI(
        title="F1 Pit-Wall Telemetry & Top-3 Podium API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = rt

    # CORS for Chrome Side Panel extension, Vercel hybrid UI, localhost dev, and mobile LAN PWA
    default_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "https://f1-pitwall-steel.vercel.app",
        *config.CORS_EXTRA_ORIGINS,
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(dict.fromkeys(default_origins)),
        allow_origin_regex=(
            r"^(chrome-extension://.*"
            r"|https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?"
            r"|https://[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.(vercel\.app|onrender\.com|up\.railway\.app|fly\.dev))$"
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # REST API
    # ------------------------------------------------------------------ #
    @app.get("/api/health")
    async def api_health() -> Dict[str, Any]:
        return rt.health()

    @app.get("/api/sessions")
    async def api_sessions(
        year: int = Query(default=2026, ge=2023, le=2030),
    ) -> Dict[str, Any]:
        try:
            sessions = await asyncio.to_thread(rt.replay.list_available_sessions, year)
        except OpenF1Error as exc:
            raise HTTPException(status_code=502, detail=f"OpenF1 unavailable: {exc}") from exc
        # Cross-check completion against official Jolpica race results when available
        try:
            latest_done = await asyncio.to_thread(rt.jolpica.get_latest_round_with_results, year)
            if latest_done is not None:
                for s in sessions:
                    if int(s.get("round") or 0) > latest_done:
                        s["completed"] = False
        except JolpicaError:
            pass
        return {"year": year, "count": len(sessions), "sessions": sessions}

    @app.get("/api/race-results")
    async def api_race_results(
        season: int = Query(default=2026, ge=2018, le=2030),
        round: Optional[int] = Query(default=None, ge=1, le=30),
        session_key: Optional[int] = Query(default=None),
    ) -> Dict[str, Any]:
        sessions_list: Optional[List[Dict[str, Any]]] = None
        if season >= 2023:
            try:
                sessions_list = await asyncio.to_thread(rt.replay.list_available_sessions, season)
            except OpenF1Error:
                sessions_list = None
        try:
            return await asyncio.to_thread(
                get_final_race_results,
                rt.jolpica,
                rt.openf1,
                season=season,
                round_number=round,
                session_key=session_key,
                sessions_list=sessions_list,
            )
        except (JolpicaError, OpenF1Error, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/next-race")
    async def api_next_race(
        season: Optional[int] = Query(default=None, ge=2018, le=2030),
    ) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(get_next_race_schedule, rt.jolpica, season=season)
        except (JolpicaError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/standings")
    async def api_standings(
        season: Optional[int] = Query(default=None, ge=2018, le=2030),
    ) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(get_pre_race_standings, rt.jolpica, season=season)
        except JolpicaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/circuits")
    async def api_circuits() -> Dict[str, Any]:
        items = list_circuits_summary()
        return {"count": len(items), "circuits": items}

    @app.get("/api/circuits/{circuit_id}")
    async def api_circuit_detail(circuit_id: str) -> Dict[str, Any]:
        c2d = resolve_circuit_2d(circuit_id=circuit_id, name=circuit_id, grand_prix=circuit_id)
        if not c2d:
            raise HTTPException(status_code=404, detail=f"Circuit '{circuit_id}' not found.")
        return c2d

    @app.get("/api/radio/transcribe")
    async def api_radio_transcribe(url: str = Query(..., min_length=12)) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(rt.transcriber.transcribe_url, url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/predict/post-quali")
    async def api_predict_post_quali(body: Optional[PostQualiRequest] = None) -> Dict[str, Any]:
        req = body or PostQualiRequest()
        try:
            packet = await asyncio.to_thread(rt.generate_post_quali, req.season, req.round)
        except (PredictionError, JolpicaError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await rt.broadcast(packet)
        return packet

    @app.post("/api/mode")
    async def api_mode(body: ModeRequest) -> Dict[str, Any]:
        return await rt.switch_mode(body.mode)

    @app.post("/api/replay/load")
    async def api_replay_load(body: ReplayLoadRequest) -> Dict[str, Any]:
        session_key = body.session_key
        if session_key is None:
            year = int(body.season or 2026)
            try:
                sessions = await asyncio.to_thread(rt.replay.list_available_sessions, year)
            except OpenF1Error as exc:
                raise HTTPException(status_code=502, detail=f"OpenF1 unavailable: {exc}") from exc
            if not sessions:
                raise HTTPException(status_code=404, detail=f"No OpenF1 race sessions found for {year}.")
            chosen = None
            if body.grand_prix:
                needle = body.grand_prix.strip().lower()
                for s in sessions:
                    if needle in str(s.get("grand_prix", "")).lower() or needle in str(s.get("circuit_name", "")).lower():
                        chosen = s
                        break
            elif body.round is not None:
                for s in sessions:
                    if s.get("round") == body.round:
                        chosen = s
                        break
            else:
                # Default to most recent completed session of the requested season
                now_iso = _now_iso()
                completed = [s for s in sessions if s.get("date_end") and str(s["date_end"]) <= now_iso]
                chosen = completed[-1] if completed else sessions[-1]

            if chosen is None or chosen.get("session_key") is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not find OpenF1 race session matching season={year}, round={body.round}, grand_prix={body.grand_prix!r}.",
                )
            session_key = int(chosen["session_key"])

        try:
            status = await rt.replay.load_session(
                int(session_key),
                include_telemetry=body.include_telemetry,
                telemetry_window_s=body.telemetry_window_s,
            )
        except (OpenF1Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.autoplay:
            rt.replay.play()
        else:
            rt.replay.pause()

        await rt.switch_mode("replay")
        frame = rt.replay.build_frame()
        await rt.broadcast(frame)
        return {"status": "loaded", "replay": rt.replay.status(), "session_key": int(session_key)}

    @app.post("/api/replay/control")
    async def api_replay_control(body: ReplayControlRequest) -> Dict[str, Any]:
        if not rt.replay.status()["loaded"]:
            raise HTTPException(status_code=400, detail="No replay session loaded. Call POST /api/replay/load first.")

        if body.action == "play":
            status = rt.replay.play()
        elif body.action == "pause":
            status = rt.replay.pause()
        elif body.action == "set_speed":
            if body.speed is None:
                raise HTTPException(status_code=422, detail="'speed' is required for action='set_speed'.")
            status = rt.replay.set_speed(body.speed)
        elif body.action == "seek_lap":
            if body.lap is None:
                raise HTTPException(status_code=422, detail="'lap' is required for action='seek_lap'.")
            status = rt.replay.seek_lap(body.lap)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported control action: {body.action}")

        # Immediately emit updated frame so UI reflects pause/seek without delay
        frame = rt.replay.build_frame()
        await rt.broadcast(frame)
        return {"status": "ok", "replay": status}

    # ------------------------------------------------------------------ #
    # WebSocket live telemetry / replay stream
    # ------------------------------------------------------------------ #
    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = rt.register_client()

        async def _sender() -> None:
            while True:
                packet = await queue.get()
                await websocket.send_json(packet)

        async def _receiver() -> None:
            while True:
                msg = await websocket.receive_text()
                if msg.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": _now_iso()})

        send_task = asyncio.create_task(_sender())
        recv_task = asyncio.create_task(_receiver())
        try:
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        except WebSocketDisconnect:
            pass
        finally:
            rt.unregister_client(queue)

    # Mount shared PWA frontend at `/` once frontend/ exists (preserving /api/* & /ws/live)
    frontend_dir = config.PROJECT_ROOT / "frontend"
    if mount_frontend and frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    config.configure_logging()
    uvicorn.run(
        "backend.server.main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
    )
