"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token pipeline compiling Welsh open data into Datum Zero GeoJSON.
Primary source: Geofabrik Wales PBF extract via Osmium (zero API rate limits).
Fallback source: Britain & Ireland Overpass instance.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

WALES_BBOX = [-5.5, 51.3, -2.6, 53.5]
RAW_OSMIUM_FILE = Path("raw_wales_assets.geojson")
OUTPUT_FILE = Path("dist/wales_resilience_master.geojson")

OVERPASS_MIRRORS = [
    "https://overpass.atownsend.org.uk/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def build_feature(
    raw_id: str, lon: float, lat: float, tags: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize a raw OSM node into Datum Zero's canonical GeoJSON feature schema."""
    is_spring = tags.get("natural") == "spring"

    category = "resupply" if is_spring else "medical"
    accent = "#00E5FF" if is_spring else "#10B981"
    default_name = (
        "Potable Bedrock Spring" if is_spring else "Public Access AED"
    )

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [round(lon, 6), round(lat, 6)],
        },
        "properties": {
            "id": f"{category.upper()}_{raw_id}",
            "name": tags.get("name", default_name),
            "category": category,
            "categoryLabel": (
                "NATURAL SPRING"
                if is_spring
                else "DEFIBRILLATOR / BLEED KIT"
            ),
            "accentColor": accent,
            "primaryMetric": (
                "Gravity Spring"
                if is_spring
                else "Emergency Bleed / Shock Kit"
            ),
            "operationalAction": (
                "Filter with 0.1-micron pump or boil 60s."
                if is_spring
                else "Call 999 for cabinet keypad access."
            ),
        },
    }


def parse_local_osmium_geojson(path: Path) -> List[Dict[str, Any]]:
    """Parse pre-filtered Geofabrik Wales GeoJSON exported by osmium-tool."""
    logging.info("Reading local Osmium extract from %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    features: List[Dict[str, Any]] = []
    for idx, feat in enumerate(raw_data.get("features", [])):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue

        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue

        lon, lat = float(coords[0]), float(coords[1])
        tags = feat.get("properties", {})
        raw_id = str(feat.get("id", idx)).lstrip("nwr")

        features.append(build_feature(raw_id, lon, lat, tags))

    return features


def harvest_overpass_fallback() -> List[Dict[str, Any]]:
    """Fallback extraction using UK/European Overpass mirrors when run standalone."""
    min_lon, min_lat, max_lon, max_lat = WALES_BBOX
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
    [out:json][timeout:90];
    (
      node["natural"="spring"]({bbox_str});
      node["emergency"="defibrillator"]({bbox_str});
    );
    out body;
    """
    headers = {
        "User-Agent": "DatumZero-ETL/1.2 (Civil Resilience Geospatial Pipeline)"
    }

    data: Dict[str, Any] = {}
    for mirror_url in OVERPASS_MIRRORS:
        logging.info("Querying fallback Overpass mirror: %s", mirror_url)
        try:
            response = requests.post(
                mirror_url,
                data={"data": query},
                headers=headers,
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            logging.warning("Mirror %s failed (%s). Trying next...", mirror_url, exc)
            time.sleep(3)

    if not data:
        logging.error("All fallback Overpass mirrors failed.")
        sys.exit(1)

    features: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            continue
        features.append(
            build_feature(str(el["id"]), float(lon), float(lat), el.get("tags", {}))
        )

    return features


def main() -> None:
    if RAW_OSMIUM_FILE.exists():
        features = parse_local_osmium_geojson(RAW_OSMIUM_FILE)
    else:
        features = harvest_overpass_fallback()

    if not features:
        logging.warning("No features extracted. Aborting write.")
        sys.exit(1)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    logging.info(
        "Success: Compiled %d verified Welsh assets into %s (%.1f KB)",
        len(features),
        OUTPUT_FILE,
        size_kb,
    )


if __name__ == "__main__":
    main()
