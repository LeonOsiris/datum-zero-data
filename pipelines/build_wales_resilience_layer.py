"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token pipeline compiling Welsh open data into Datum Zero GeoJSON.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

WALES_BBOX = [-5.5, 51.3, -2.6, 53.5]
OUTPUT_FILE = Path("dist/wales_resilience_master.geojson")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def harvest_overpass_resupply() -> List[Dict[str, Any]]:
    """Query Overpass API for natural springs and public defibrillators across Wales."""
    min_lon, min_lat, max_lon, max_lat = WALES_BBOX
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
    [out:json][timeout:60];
    (
      node["natural"="spring"]({bbox_str});
      node["emergency"="defibrillator"]({bbox_str});
    );
    out body;
    """

    headers = {
        "User-Agent": "DatumZero-ETL/1.0 (Civil Resilience Geospatial Pipeline)"
    }

    logging.info("Querying Overpass API for Wales bounding box [%s]...", bbox_str)
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers=headers,
            timeout=65,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
        logging.error("Overpass API extraction failed: %s", exc)
        sys.exit(1)

    features: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            continue

        tags = el.get("tags", {})
        is_spring = tags.get("natural") == "spring"

        category = "resupply" if is_spring else "medical"
        accent = "#00E5FF" if is_spring else "#10B981"
        default_name = (
            "Potable Bedrock Spring" if is_spring else "Public Access AED"
        )

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 6), round(lat, 6)],
                },
                "properties": {
                    "id": f"{category.upper()}_{el['id']}",
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
        )

    return features


def main() -> None:
    features = harvest_overpass_resupply()

    if not features:
        logging.warning("No features returned. Aborting write.")
        sys.exit(1)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))

    logging.info(
        "Compiled %d real-world assets into %s",
        len(features),
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
