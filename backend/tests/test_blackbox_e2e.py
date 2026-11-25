"""Automated Black-Box End-to-End Test Suite (Phase 3 Verification).

Exercises the FastAPI application, WebSocket stream, PWA static shell, Chrome
Extension Manifest V3 bundle, CSP cleanliness, and late-race ML convergence
strictly from external interfaces (`TestClient` & file-bundle inspection).
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from fastapi.testclient import TestClient
import joblib
from jsonschema import Draft202012Validator
import pytest

import config
from ml.features import (
    build_in_race_features,
    build_pre_race_features,
    make_in_race_x,
    normalize_within_groups,
)
from ml.ingest_history import load_in_race_raw, load_pre_race_raw
from server.main import PitWallRuntime, create_app


@pytest.fixture()
def bb_client() -> TestClient:
    rt = PitWallRuntime(initial_mode="replay")
    app = create_app(runtime=rt, mount_frontend=True)
    with TestClient(app) as client:
        yield client


def test_blackbox_health_circuits_and_negative_boundaries(bb_client: TestClient) -> None:
    # Health & loaded ML artifacts
    r_health = bb_client.get("/api/health")
    assert r_health.status_code == 200
    body = r_health.json()
    assert body["status"] == "ok"
    if config.PRE_RACE_MODEL_PATH.exists():
        assert body["models"]["pre_race"] is not None
    if config.LIVE_RACE_MODEL_PATH.exists():
        assert body["models"]["in_race"] is not None

    # All 26 official circuits + individual lookup + 404 negative test
    r_circs = bb_client.get("/api/circuits")
    assert r_circs.status_code == 200
    assert r_circs.json()["count"] >= 26

    for slug in ("las_vegas", "miami", "monte_carlo", "madring", "sepang", "lusail"):
        assert bb_client.get(f"/api/circuits/{slug}").status_code == 200
    assert bb_client.get("/api/circuits/does_not_exist").status_code == 404

    # Boundary value negative tests (422 / 400)
    assert bb_client.get("/api/sessions?year=1999").status_code == 422
    assert bb_client.get("/api/sessions?year=2099").status_code == 422
    assert bb_client.post("/api/predict/post-quali", json={"season": 2026, "round": 99}).status_code == 422
    assert bb_client.post("/api/mode", json={"mode": "invalid"}).status_code == 422
    assert bb_client.post("/api/replay/control", json={"action": "pause"}).status_code == 400


def test_blackbox_ssrf_and_cors_security(bb_client: TestClient) -> None:
    # SSRF block on radio transcriber endpoint
    for bad in (
        "http://127.0.0.1:8000/api/health",
        "http://169.254.169.254/latest/meta-data/",
        "https://attacker.example.com/audio.mp3",
    ):
        resp = bb_client.get("/api/radio/transcribe", params={"url": bad})
        assert resp.status_code == 400

    # CORS preflight for Chrome Extension & LAN mobile PWA
    for origin in ("chrome-extension://f1pitwallid", "http://192.168.1.42:8000"):
        preflight = bb_client.options(
            "/api/health",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        assert preflight.headers.get("access-control-allow-origin") == origin


def test_blackbox_websocket_ping_pong_and_static_pwa_shell(bb_client: TestClient) -> None:
    with bb_client.websocket_connect("/ws/live") as ws:
        ws.send_text("ping")
        msg = ws.receive_json()
        assert msg["type"] == "pong"

    # Verify all 3 dedicated pages + PWA manifest + Service Worker serve 200 OK
    for path in ("/", "/index.html", "/live.html", "/predictions.html", "/styles.css", "/app.js", "/pwa-manifest.json", "/sw.js"):
        r = bb_client.get(path)
        assert r.status_code == 200, f"Static route {path} failed with {r.status_code}"


def test_blackbox_chrome_extension_mv3_bundle_and_csp_compliance() -> None:
    ext_dir = config.PROJECT_ROOT / "extension"
    manifest = json.loads((ext_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 3
    assert manifest["content_security_policy"]["extension_pages"] == "script-src 'self'; object-src 'self'"
    assert "sidePanel" in manifest["permissions"]
    assert (ext_dir / manifest["background"]["service_worker"]).exists()
    assert (ext_dir / manifest["side_panel"]["default_path"]).exists()

    for icon_rel in manifest["icons"].values():
        assert (ext_dir / icon_rel).exists(), f"Missing extension icon {icon_rel}"

    # Ensure zero external CDN scripts/stylesheets in extension HTML files (strict MV3 CSP)
    for html_name in ("sidepanel.html", "index.html", "live.html", "predictions.html"):
        html_path = ext_dir / html_name
        assert html_path.exists(), f"Missing {html_name} in extension/"
        text = html_path.read_text(encoding="utf-8")
        assert "pwa-manifest.json" not in text
        assert not re.search(r'<(script|link)[^>]+https?://', text, flags=re.IGNORECASE)


def test_phase3_in_race_model_late_race_convergence() -> None:
    """Verify the trained live in-race model becomes sharper & more accurate as laps progress."""
    if not (
        config.LIVE_RACE_MODEL_PATH.exists()
        and config.PRE_RACE_RAW_PATH.exists()
        and config.IN_RACE_RAW_PATH.exists()
    ):
        pytest.skip("Local parquet history or live_race_model.joblib not present")

    art = joblib.load(config.LIVE_RACE_MODEL_PATH)
    pre = build_pre_race_features(load_pre_race_raw())
    inr = build_in_race_features(load_in_race_raw(), pre)
    inr = inr.dropna(subset=["position", "is_podium"]).reset_index(drop=True)

    scores = art["model"].predict_proba(make_in_race_x(inr))[:, 1]
    inr["p"] = normalize_within_groups(inr, ["season", "round", "lap"], scores, 3.0, 1.0)
    inr["rk"] = inr.groupby(["season", "round", "lap"])["p"].rank(ascending=False, method="first")

    early = inr[(inr["lap_fraction"] <= 0.25) & (inr["rk"] <= 3)]["is_podium"].mean()
    mid = inr[(inr["lap_fraction"] > 0.25) & (inr["lap_fraction"] <= 0.75) & (inr["rk"] <= 3)]["is_podium"].mean()
    late = inr[(inr["lap_fraction"] > 0.75) & (inr["rk"] <= 3)]["is_podium"].mean()

    assert late > mid > early > 0.55
