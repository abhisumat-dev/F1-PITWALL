"""Vercel FastAPI entrypoint for F1 Pit-Wall."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VERCEL", "1")
os.environ.setdefault("F1_ALLOW_LIVE_NETWORK", "0")

_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from server.main import create_app  # noqa: E402

app = create_app(mount_frontend=True)
