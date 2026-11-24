"""Unified asynchronous stream provider interface for F1 Pit-Wall.

Both :class:`streams.replay_engine.ReplayEngine` (OpenF1 historical replays)
and :class:`streams.signalr_live.SignalRLiveProvider` (unauthenticated live
F1 SignalR feed) implement :class:`BaseStreamProvider` so the FastAPI server
can broadcast contract-valid packets over ``/ws/live`` uniformly regardless of
mode.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, Set


class BaseStreamProvider(ABC):
    """Abstract base class for live and replay telemetry providers."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._running = False
        self._sequence = 0
        self._latest_packet: Optional[Dict[str, Any]] = None
        self._subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def latest_packet(self) -> Optional[Dict[str, Any]]:
        return self._latest_packet

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def subscribe(self, maxsize: int = 8) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        if self._latest_packet is not None:
            try:
                queue.put_nowait(self._latest_packet)
            except asyncio.QueueFull:
                pass
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, packet: Dict[str, Any]) -> None:
        """Store ``packet`` as latest and fan out to all active subscribers without blocking."""
        self._latest_packet = packet
        dead: list[asyncio.Queue[Dict[str, Any]]] = []
        for queue in list(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(packet)
            except Exception:  # noqa: BLE001
                dead.append(queue)
        for q in dead:
            self._subscribers.discard(q)

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Async generator yielding packets as they arrive."""
        queue = self.subscribe()
        try:
            while True:
                packet = await queue.get()
                yield packet
        finally:
            self.unsubscribe(queue)

    @abstractmethod
    async def start(self) -> None:
        """Start streaming frames."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming frames and release resources."""

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Return health/status metadata for ``GET /api/health``."""
