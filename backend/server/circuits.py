"""Real 2D F1 Circuit Geometry Catalog loader and fuzzy matcher.

Loads ``backend/server/circuits_2d.json`` (built from official OpenF1 /
MultiViewer circuit coordinates, rotation angles, corner numbers, and
Sector 1 / 2 / 3 arc-length splits) so every F1 circuit renders accurately in
both Live and Replay modes.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CIRCUITS_JSON_PATH = Path(__file__).resolve().parent / "circuits_2d.json"

# Common Jolpica / Ergast circuit_id aliases -> OpenF1 short_name slugs
CIRCUIT_ALIASES: Dict[str, str] = {
    "vegas": "las_vegas",
    "las_vegas": "las_vegas",
    "losail": "lusail",
    "lusail": "lusail",
    "qatar": "lusail",
    "yas_marina": "yas_marina_circuit",
    "abu_dhabi": "yas_marina_circuit",
    "bahrain": "sakhir",
    "sakhir": "sakhir",
    "jeddah": "jeddah",
    "albert_park": "melbourne",
    "melbourne": "melbourne",
    "suzuka": "suzuka",
    "shanghai": "shanghai",
    "miami": "miami",
    "imola": "imola",
    "monaco": "monte_carlo",
    "monte_carlo": "monte_carlo",
    "villeneuve": "montreal",
    "montreal": "montreal",
    "catalunya": "catalunya",
    "barcelona": "catalunya",
    "red_bull_ring": "spielberg",
    "spielberg": "spielberg",
    "silverstone": "silverstone",
    "hungaroring": "hungaroring",
    "budapest": "hungaroring",
    "spa": "spa_francorchamps",
    "zandvoort": "zandvoort",
    "monza": "monza",
    "baku": "baku",
    "marina_bay": "singapore",
    "singapore": "singapore",
    "americas": "austin",
    "cota": "austin",
    "austin": "austin",
    "rodriguez": "mexico_city",
    "mexico": "mexico_city",
    "interlagos": "interlagos",
    "sao_paulo": "interlagos",
    "madrid": "madrid",
}

__all__ = ["load_circuits_catalog", "list_circuits_summary", "resolve_circuit_2d"]


@lru_cache(maxsize=1)
def load_circuits_catalog() -> Dict[str, Dict[str, Any]]:
    if not CIRCUITS_JSON_PATH.exists():
        return {}
    try:
        payload = json.loads(CIRCUITS_JSON_PATH.read_text(encoding="utf-8"))
        return dict(payload.get("circuits") or {})
    except (OSError, ValueError):
        return {}


def list_circuits_summary() -> List[Dict[str, Any]]:
    catalog = load_circuits_catalog()
    out: List[Dict[str, Any]] = []
    for slug, c in sorted(catalog.items(), key=lambda kv: kv[1].get("short_name", kv[0])):
        out.append(
            {
                "slug": slug,
                "circuit_key": c.get("circuit_key"),
                "short_name": c.get("short_name"),
                "meeting_name": c.get("meeting_name"),
                "country_name": c.get("country_name"),
                "location": c.get("location"),
                "rotation_deg": c.get("rotation_deg"),
                "corners_count": len(c.get("corners") or []),
                "points_count": len(c.get("points") or []),
            }
        )
    return out


def resolve_circuit_2d(
    *,
    circuit_key: Optional[Any] = None,
    circuit_id: Optional[str] = None,
    name: Optional[str] = None,
    grand_prix: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find the matching 2D circuit geometry by OpenF1 key, Jolpica id, or GP/circuit name."""
    catalog = load_circuits_catalog()
    if not catalog:
        return None

    # 1. Exact circuit_key match
    if circuit_key is not None:
        try:
            ck_int = int(circuit_key)
            for item in catalog.values():
                if item.get("circuit_key") == ck_int:
                    return item
        except (TypeError, ValueError):
            pass

    # 2. Direct slug / alias lookup
    for raw in (circuit_id, name, grand_prix):
        if not raw:
            continue
        norm = (
            str(raw)
            .strip()
            .lower()
            .replace("grand prix", "")
            .replace("circuit", "")
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )
        if norm in catalog:
            return catalog[norm]
        mapped = CIRCUIT_ALIASES.get(norm)
        if mapped and mapped in catalog:
            return catalog[mapped]
        for alias_k, alias_v in CIRCUIT_ALIASES.items():
            if alias_k in norm and alias_v in catalog:
                return catalog[alias_v]

    # 3. Substring match on short_name / location / meeting_name
    for raw in (name, grand_prix, circuit_id):
        if not raw:
            continue
        needle = str(raw).strip().lower()
        for slug, item in catalog.items():
            hay = " ".join(
                str(item.get(f) or "").lower()
                for f in ("short_name", "location", "meeting_name", "country_name")
            )
            if needle in hay or str(item.get("short_name", "")).lower() in needle:
                return item

    return None
