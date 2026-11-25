"""Local Deep Learning Team Radio speech-to-text using faster-whisper ($0, offline cache).

Audio MP3 URLs from OpenF1 (`/v1/team_radio`) and F1 SignalR (`TeamRadio` topic)
are transcribed lazily on demand and persisted atomically to
``backend/cache/radio_transcripts.json`` so 4 Hz packet generation is never
blocked and each clip is only transcribed once.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

import config

LOGGER = logging.getLogger(__name__)

F1_RADIO_PROMPT = (
    "Formula 1 team radio communication: box this lap, stay out, undercut, overcut, "
    "graining, blistering, tyre degradation, soft, medium, hard, intermediate, wet, "
    "safety car, VSC, track limits, brake bias, diff, overtake, Sector 1, Sector 2, Sector 3, "
    "Verstappen, Norris, Leclerc, Piastri, Russell, Hamilton, Antonelli, Sainz, Alonso, Gasly."
)

ALLOWED_RADIO_HOSTS = (
    "livetiming.formula1.com",
    "api.openf1.org",
)

DEFAULT_TRANSCRIPT_CACHE: Path = config.CACHE_DIR / "radio_transcripts.json"

__all__ = [
    "RadioTranscriber",
    "is_allowed_radio_url",
    "F1_RADIO_PROMPT",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def is_allowed_radio_url(url: str) -> bool:
    """Allow only official F1 live timing / OpenF1 audio clip URLs (prevents SSRF)."""
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host in ALLOWED_RADIO_HOSTS


class RadioTranscriber:
    """Lazy faster-whisper speech-to-text engine with atomic JSON disk cache."""

    def __init__(
        self,
        *,
        cache_path: Path = DEFAULT_TRANSCRIPT_CACHE,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._cache = raw
        except (OSError, ValueError) as exc:
            LOGGER.warning("Failed to load radio transcript cache: %s", exc)

    def _save_cache_atomic(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            LOGGER.warning("Failed to persist radio transcript cache: %s", exc)

    def get_cached(self, url: str) -> Optional[str]:
        """Instant non-blocking cache lookup for 4 Hz frame builders."""
        if not url:
            return None
        with self._cache_lock:
            entry = self._cache.get(url)
            if isinstance(entry, dict):
                text = entry.get("transcript")
                return str(text) if text else None
        return None

    def put_cached(self, url: str, transcript: str, *, model: Optional[str] = None) -> Dict[str, Any]:
        entry = {
            "url": url,
            "transcript": transcript.strip(),
            "model": model or f"faster-whisper-{self.model_size}",
            "created_at": _now_iso(),
        }
        with self._cache_lock:
            self._cache[url] = entry
            self._save_cache_atomic()
        return entry

    def _ensure_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            LOGGER.info(
                "Loading faster-whisper model '%s' (%s / %s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            return self._model

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe a local MP3/WAV file using faster-whisper with F1 vocabulary prompt."""
        model = self._ensure_model()
        segments, _info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            initial_prompt=F1_RADIO_PROMPT,
        )
        parts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
        return " ".join(parts).strip()

    def transcribe_url(self, url: str, *, timeout_s: float = 20.0) -> Dict[str, Any]:
        """Download an official F1/OpenF1 team radio MP3 and transcribe it (cached)."""
        clean_url = str(url or "").strip()
        if not is_allowed_radio_url(clean_url):
            raise ValueError("Radio URL must be an https:// F1 Live Timing or OpenF1 recording URL.")

        cached = self.get_cached(clean_url)
        if cached is not None:
            return {
                "url": clean_url,
                "transcript": cached,
                "cached": True,
                "status": "ok",
                "model": f"faster-whisper-{self.model_size}",
            }

        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return {
                "url": clean_url,
                "transcript": None,
                "cached": False,
                "status": "unavailable",
                "detail": "faster-whisper is not installed in the backend environment.",
            }

        tmp_path: Optional[Path] = None
        try:
            with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
                resp = client.get(clean_url, headers={"User-Agent": config.USER_AGENT})
                resp.raise_for_status()
                content = resp.content

            if not content:
                return {
                    "url": clean_url,
                    "transcript": "",
                    "cached": False,
                    "status": "empty",
                }

            with tempfile.NamedTemporaryFile(prefix="f1_radio_", suffix=".mp3", delete=False) as tf:
                tf.write(content)
                tmp_path = Path(tf.name)

            text = self.transcribe_file(tmp_path)
            entry = self.put_cached(clean_url, text)
            return {
                "url": clean_url,
                "transcript": entry["transcript"],
                "cached": False,
                "status": "ok",
                "model": entry["model"],
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Team radio transcription failed for %s: %s", clean_url, exc)
            return {
                "url": clean_url,
                "transcript": None,
                "cached": False,
                "status": "error",
                "detail": str(exc),
            }
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
