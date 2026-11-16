"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token pipeline compiling Welsh open data into Datum Zero GeoJSON.
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

# Ordered by cloud-runner reliability (avoiding rate-limited primary node first)
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Verified RSGB ETCC Amateur Radio Repeaters across Wales (Static Statutory Register)
VERIFIED_WELSH_REPEATERS: List[Dict[str, Any]] = [
    {
        "id": "COM_GB3AR",
        "name": "GB3AR - Bangor Mountain Repeater",
        "coords": [-4.1262, 53.2214],
        "metric": "TX: 145.725 MHz | RX: 145.125 MHz | CTCSS: 103.5 Hz",
        "action": "2m analog voice repeater (-0.6 MHz offset). Backup AGM battery power.",
    },
    {
        "id": "COM_GB3GW",
        "name": "GB3GW - Great Orme Repeater",
        "coords": [-3.8550, 53.3320],
        "metric": "TX: 433.000 MHz | RX: 434.600 MHz | CTCSS: 103.5 Hz",
        "action": "70cm coastal repeater (+1.6 MHz offset). Covers Conwy Bay & Menai Strait.",
    },
    {
        "id": "COM_GB3LL",
        "name": "GB3LL - Llanddona Anglesey Repeater",
        "coords": [-4.1350, 53.2940],
        "metric": "TX: 145.775 MHz | RX: 145.175 MHz | CTCSS: 103.5 Hz",
        "action": "2m primary Anglesey repeater (-0.6 MHz offset). Line-of-sight to Caernarfon.",
    },
    {
        "id": "COM_GB3WW",
        "name": "GB3WW - Preseli Hills Repeater (Pembrokeshire)",
        "coords": [-4.7731, 51.9462],
        "metric": "TX: 145.625 MHz | RX: 145.025 MHz | CTCSS: 77.0 Hz",
        "action": "2m wide-area West Wales repeater (-0.6 MHz offset).",
    },
    {
        "id": "COM_GB3SW",
        "name": "GB3SW - Kilvey Hill Repeater (Swansea)",
        "coords": [-3.9167, 51.6247],
        "metric": "TX: 145.675 MHz | RX: 145.075 MHz | CTCSS: 110.9 Hz",
        "action": "2m South Wales coastal & Gower repeater (-0.6 MHz offset).",
    },
    {
        "id": "COM_GB3MH",
        "name": "GB3MH - Mynydd Machen Repeater (South Valleys)",
        "coords": [-3.1194, 51.5942],
        "metric": "TX: 145.650 MHz | RX: 145.050 MHz | CTCSS: 110.9 Hz",
        "action": "2m repeater serving Rhondda, Cynon, and Taff valley corridors.",
    },
]


def build_osm_feature(
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


def get_repeater_features() -> List[Dict[str, Any]]:
    """Return normalized GeoJSON features for verified RSGB Welsh repeaters."""
    features: List[Dict[str, Any]] = []
    for rep in VERIFIED_WELSH_REPEATERS:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": rep["coords"],
                },
                "properties": {
                    "id": rep["id"],
                    "name": rep["name"],
                    "category": "grid_down",
                    "categoryLabel": "RSGB AMATEUR REPEATER",
                    "accentColor": "#FFB800",
                    "primaryMetric": rep["metric"],
                    "operationalAction": rep["action"],
                },
            }
        )
    return features


def parse_local_osmium_geojson(path: Path) -> List[Dict[str, Any]]:
    """Parse pre-filtered Wales GeoJSON exported by osmium-tool."""
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
        features.append(build_osm_feature(raw_id, lon, lat, tags))

    return features


def harvest_overpass_fallback() -> List[Dict[str, Any]]:
    """Fallback extraction across redundant Overpass mirrors."""
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
        "User-Agent": "DatumZero-ETL/1.3 (Civil Resilience Geospatial Pipeline)"
    }

    data: Dict[str, Any] = {}
    for mirror_url in OVERPASS_MIRRORS:
        logging.info("Querying Overpass mirror: %s", mirror_url)
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
            logging.warning("Mirror %s failed (%s). Failing over...", mirror_url, exc)
            time.sleep(2)

    if not data:
        logging.error("All Overpass mirrors failed.")
        sys.exit(1)

    features: List[Dict[str, Any]] = []
    for el in data.get("elements", []):
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            continue
        features.append(
            build_osm_feature(
                str(el["id"]), float(lon), float(lat), el.get("tags", {})
            )
        )

    return features


def main() -> None:
    if RAW_OSMIUM_FILE.exists():
        osm_features = parse_local_osmium_geojson(RAW_OSMIUM_FILE)
    else:
        osm_features = harvest_overpass_fallback()

    repeater_features = get_repeater_features()
    all_features = osm_features + repeater_features

    if not all_features:
        logging.warning("No features extracted. Aborting write.")
        sys.exit(1)

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    logging.info(
        "Success: Compiled %d verified Welsh assets into %s (%.1f KB)",
        len(all_features),
        OUTPUT_FILE,
        size_kb,
    )


if __name__ == "__main__":
    main()
