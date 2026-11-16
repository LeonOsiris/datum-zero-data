"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token spatial ETL compiling Welsh open data into Datum Zero GeoJSON.
Generates fixed-scale metric Polygon Threat Clouds (EPSG:27700 -> EPSG:4326) with
concentric feathered dispersion shells (zero pixel-scaling artifacts).
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

WALES_BBOX = [-5.5, 51.3, -2.6, 53.5]
OUTPUT_FILE = Path("dist/wales_resilience_master.geojson")

# Keyless OGC WFS endpoint on DataMapWales (Welsh Government Open Data - OGL v3.0)
DATAMAPWALES_WFS_URL = "https://datamap.gov.wales/geoserver/ows"

OVERPASS_MIRRORS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Verified RSGB ETCC Amateur Radio Repeaters across Wales
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

# Statutory NRW River Telemetry Stations across major Welsh spate basins
VERIFIED_NRW_GAUGES: List[Dict[str, Any]] = [
    {
        "id": "NRW_GAUGE_GLASLYN",
        "name": "NRW Gauge: River Glaslyn at Beddgelert",
        "coords": [-4.1025, 53.0110],
        "metric": "Alert Stage: 1.65m | Severe Flood: 2.10m | Basin: Eryri South",
        "action": "Monitors Yr Wyddfa southern runoff. Above 1.65m, A498 Aberglaslyn Pass floods; divert via A4085.",
    },
    {
        "id": "NRW_GAUGE_CONWY",
        "name": "NRW Gauge: River Conwy at Llanrwst (Pont Fawr)",
        "coords": [-3.7960, 53.1385],
        "metric": "Alert Stage: 2.40m | Bridge Overtop: 3.10m | Basin: Conwy",
        "action": "Fast-responding Carneddau spate river. Pont Fawr closes early; use eastern A470 high corridor.",
    },
    {
        "id": "NRW_GAUGE_DEE",
        "name": "NRW Gauge: River Dee at Bala / Llyn Tegid Outflow",
        "coords": [-3.5885, 52.9092],
        "metric": "Alert Stage: 1.95m | Floodplain Spill: 2.50m | Basin: Upper Dee",
        "action": "Controls Dee valley inundation downstream toward Corwen and Llangollen.",
    },
    {
        "id": "NRW_GAUGE_DYFI",
        "name": "NRW Gauge: Afon Dyfi at Machynlleth Bridge",
        "coords": [-3.8548, 52.5964],
        "metric": "Alert Stage: 2.80m | A487 Cut-Off: 3.40m | Basin: Dyfi",
        "action": "Single-corridor A487 north-south artery severs when tidal surge meets spate discharge.",
    },
    {
        "id": "NRW_GAUGE_TEIFI",
        "name": "NRW Gauge: Afon Teifi at Lampeter",
        "coords": [-4.0792, 52.1118],
        "metric": "Alert Stage: 2.15m | Severe Flood: 2.85m | Basin: Teifi",
        "action": "Low-lying valley floor inundation risk across A482/A475 river crossings.",
    },
    {
        "id": "NRW_GAUGE_TYWI",
        "name": "NRW Gauge: River Tywi at Carmarthen Quay",
        "coords": [-4.3055, 51.8542],
        "metric": "Alert Stage: 3.20m | Tidal Lock: 4.10m | Basin: Tywi",
        "action": "Wide floodplain inundation across A40/A484 valley approaches during spring tides.",
    },
    {
        "id": "NRW_GAUGE_TAFF",
        "name": "NRW Gauge: River Taff at Pontypridd",
        "coords": [-3.3388, 51.6012],
        "metric": "Alert Stage: 2.60m | Severe Flood: 3.50m | Basin: Taff / Rhondda",
        "action": "Confluence bottleneck for Rhondda and Taff valleys; severe urban flash-flood velocity.",
    },
    {
        "id": "NRW_GAUGE_USK",
        "name": "NRW Gauge: River Usk at Brecon Bridge",
        "coords": [-3.3935, 51.9478],
        "metric": "Alert Stage: 2.30m | Severe Flood: 3.10m | Basin: Usk",
        "action": "Drains northern Bannau Brycheiniog; low town bridges and riverside roads flood rapidly.",
    },
]

# Statutory HSE Public COMAH Upper/Lower-Tier Industrial Sites in Wales (with metric clearance radius)
VERIFIED_COMAH_SITES: List[Dict[str, Any]] = [
    {
        "id": "COMAH_PEMBROKE",
        "name": "HSE COMAH Zone: Milford Haven / Pembroke Refinery",
        "coords": [-5.0195, 51.6895],
        "radius_m": 1200,
        "metric": "Upper-Tier COMAH | Exclusion Buffer: 1,200m | W_comah +8.0",
        "action": "Major hydrocarbon storage & refining facility. Maintain 1.2km upwind clearance during industrial alarms.",
    },
    {
        "id": "COMAH_BARRY",
        "name": "HSE COMAH Zone: Barry Docks Chemical Complex",
        "coords": [-3.2535, 51.3995],
        "radius_m": 900,
        "metric": "Upper-Tier COMAH | Exclusion Buffer: 900m | W_comah +8.0",
        "action": "Silicone & specialty chemical installations. Shelter indoors or evacuate cross-wind if vapour release occurs.",
    },
    {
        "id": "COMAH_BAGLAN",
        "name": "HSE COMAH Zone: Baglan Bay / Port Talbot Industrial Corridor",
        "coords": [-3.8165, 51.6085],
        "radius_m": 1000,
        "metric": "Upper-Tier COMAH | Exclusion Buffer: 1,000m | W_comah +8.0",
        "action": "Heavy industrial & energy storage corridor adjacent to M4. Divert inland during hazmat incidents.",
    },
    {
        "id": "COMAH_DEESIDE",
        "name": "HSE COMAH Zone: Deeside / Shotton Industrial & Gas Terminal",
        "coords": [-3.0465, 53.2185],
        "radius_m": 850,
        "metric": "Upper-Tier COMAH | Exclusion Buffer: 850m | W_comah +8.0",
        "action": "Combined cycle power & chemical manufacturing zone on Dee estuary.",
    },
    {
        "id": "COMAH_WREXHAM",
        "name": "HSE COMAH Zone: Wrexham Industrial Estate Hazmat Sector",
        "coords": [-2.9195, 53.0355],
        "radius_m": 750,
        "metric": "Lower-Tier COMAH | Exclusion Buffer: 750m | W_comah +8.0",
        "action": "Industrial chemical & manufacturing storage buffer. Maintain perimeter clearance.",
    },
]

# Major Welsh River Flood Plain & Hydrological Disruption Corridors
VERIFIED_FLOOD_CORRIDORS: List[Dict[str, Any]] = [
    {
        "id": "HAZ_FLOOD_LLYN_PERIS",
        "name": "Llyn Peris / A4086 Hydrological & Cordon Buffer",
        "line": [[-4.1440, 53.1350], [-4.1280, 53.1230], [-4.1160, 53.1110]],
        "buffer_m": 320,
        "metric": "W_flood +10.0 | Valley Inundation & Infrastructure Cordon",
        "action": "High-voltage surge & reservoir tailrace overflow corridor. Divert above 150m contour via Waunfawr.",
    },
    {
        "id": "HAZ_FLOOD_CONWY_VALLEY",
        "name": "Conwy Valley Floodplain (Llanrwst to Trefriw)",
        "line": [[-3.8150, 53.1750], [-3.8020, 53.1520], [-3.7950, 53.1340], [-3.7850, 53.1120]],
        "buffer_m": 450,
        "metric": "1-in-30-Yr Flood Zone 3 | W_flood +10.0",
        "action": "B5106 & valley floor submerge rapidly during Carneddau storms. Use elevated eastern A470.",
    },
    {
        "id": "HAZ_FLOOD_ABERGLASLYN",
        "name": "Afon Glaslyn / Beddgelert Flood Corridor",
        "line": [[-4.1060, 53.0160], [-4.1010, 53.0050], [-4.1080, 52.9880]],
        "buffer_m": 280,
        "metric": "1-in-30-Yr Flood Zone 3 | W_flood +10.0",
        "action": "Narrow gorge bottleneck on A498. Prone to flash river overtopping and rockfall.",
    },
    {
        "id": "HAZ_FLOOD_DYFI_ESTUARY",
        "name": "Afon Dyfi Floodplain (Machynlleth Crossing)",
        "line": [[-3.8850, 52.5920], [-3.8540, 52.5970], [-3.8220, 52.6040]],
        "buffer_m": 500,
        "metric": "1-in-30-Yr Tidal/Fluvial Zone | W_flood +10.0",
        "action": "A487 bridge approaches flood during combined high tide and mountain runoff.",
    },
    {
        "id": "HAZ_FLOOD_TAFF_RHONDDA",
        "name": "Taff & Rhondda Confluence Flood Corridor (Pontypridd)",
        "line": [[-3.3680, 51.6150], [-3.3390, 51.6010], [-3.3210, 51.5820]],
        "buffer_m": 380,
        "metric": "1-in-30-Yr Spate Flood Zone | W_flood +10.0",
        "action": "Steep valley walls funnel high-velocity water onto A470/A4058 valley floor.",
    },
    {
        "id": "HAZ_FLOOD_TYWI_VALLEY",
        "name": "Lower Tywi Floodplain (Llandeilo to Carmarthen)",
        "line": [[-4.3100, 51.8520], [-4.1850, 51.8640], [-4.0120, 51.8810]],
        "buffer_m": 550,
        "metric": "1-in-30-Yr Flood Zone 3 | W_flood +10.0",
        "action": "Extensive agricultural & road plain inundation. Maintain route along elevated A40 ridge.",
    },
]


def round_coordinates(coords: Any, precision: int = 5) -> Any:
    """Recursively round GeoJSON coordinates to keep mobile WebGL payloads compact."""
    if isinstance(coords, (list, tuple)):
        if len(coords) > 0 and isinstance(coords[0], (int, float)):
            return [round(float(c), precision) for c in coords[:2]]
        return [round_coordinates(c, precision) for c in coords]
    return coords


def make_feathered_metric_polygons(
    geom_wgs84: Any,
    core_buffer_m: float,
    halo_extra_m: float,
    base_props: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Create two true geographic polygons in British National Grid (EPSG:27700):
    1. Core threat zone (`cloudRing: 'core'`) at exact physical radius.
    2. Outer atmospheric dispersion ring (`cloudRing: 'halo'`) slightly wider in true meters.
    This guarantees soft-edged Threat Clouds locked to real-world scale at all zoom levels.
    """
    gdf = gpd.GeoDataFrame([{"geometry": geom_wgs84}], crs="EPSG:4326").to_crs(epsg=27700)

    core_geom = gdf.geometry.iloc[0].buffer(core_buffer_m, cap_style="round").simplify(
        tolerance=10.0, preserve_topology=True
    )
    halo_geom = gdf.geometry.iloc[0].buffer(
        core_buffer_m + halo_extra_m, cap_style="round"
    ).simplify(tolerance=14.0, preserve_topology=True)

    core_wgs = gpd.GeoSeries([core_geom], crs="EPSG:27700").to_crs(epsg=4326).iloc[0]
    halo_wgs = gpd.GeoSeries([halo_geom], crs="EPSG:27700").to_crs(epsg=4326).iloc[0]

    wales_clip = box(*WALES_BBOX)
    core_wgs = core_wgs.intersection(wales_clip)
    halo_wgs = halo_wgs.intersection(wales_clip)

    out: List[Dict[str, Any]] = []
    if not halo_wgs.is_empty:
        halo_dict = mapping(halo_wgs)
        halo_dict["coordinates"] = round_coordinates(halo_dict["coordinates"], 5)
        out.append(
            {
                "type": "Feature",
                "geometry": halo_dict,
                "properties": {
                    **base_props,
                    "id": f"{base_props['id']}_HALO",
                    "cloudRing": "halo",
                },
            }
        )
    if not core_wgs.is_empty:
        core_dict = mapping(core_wgs)
        core_dict["coordinates"] = round_coordinates(core_dict["coordinates"], 5)
        out.append(
            {
                "type": "Feature",
                "geometry": core_dict,
                "properties": {
                    **base_props,
                    "cloudRing": "core",
                },
            }
        )
    return out


def parse_max_voltage(voltage_tag: Any) -> Optional[int]:
    """Extract highest transmission voltage in Volts from OSM voltage tag."""
    if not voltage_tag:
        return None
    matches = re.findall(r"\d+", str(voltage_tag))
    if not matches:
        return None
    max_v = max(int(v) for v in matches)
    if max_v in (132, 275, 400):
        max_v *= 1000
    return max_v


def classify_emf_buffer(voltage_v: Optional[int]) -> Optional[Tuple[int, str]]:
    """Return (buffer_meters, tier_label) per Biot-Savart specification."""
    if not voltage_v:
        return None
    if voltage_v >= 400000:
        return (150, "400kV Supergrid")
    if voltage_v >= 275000:
        return (100, "275kV Transmission")
    if voltage_v >= 132000:
        return (50, "132kV Regional Grid")
    return None


def build_point_feature(
    raw_id: str, lon: float, lat: float, tags: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Normalize a raw OSM Point node into Datum Zero's canonical schema."""
    if tags.get("natural") == "spring":
        category = "resupply"
        accent = "#00E5FF"
        label = "NATURAL SPRING"
        default_name = "Potable Bedrock Spring"
        metric = "Gravity Spring"
        action = "Filter with 0.1-micron pump or boil 60s."
        weight = 0.0
    elif tags.get("emergency") == "defibrillator":
        category = "medical"
        accent = "#10B981"
        label = "DEFIBRILLATOR / BLEED KIT"
        default_name = "Public Access AED"
        metric = "Emergency Bleed / Shock Kit"
        action = "Call 999 for cabinet keypad access."
        weight = 0.0
    elif tags.get("man_made") == "surveillance":
        category = "baseline"
        accent = "#94A3B8"
        label = "PUBLIC SURVEILLANCE (CCTV)"
        default_name = "Municipal / Traffic Camera"
        metric = f"Type: {tags.get('surveillance:type', 'Traffic / Public')}"
        action = "Public right-of-way camera installation."
        weight = 0.0
    elif tags.get("ford") == "yes":
        category = "hazards"
        accent = "#FF3B30"
        label = "UNMONITORED RIVER FORD"
        default_name = "Low-Water River Ford Crossing"
        metric = "Flood Cut-Off Risk | W_flood +10.0"
        action = "Impassable during heavy rain/spate discharge. Verify water depth before crossing."
        weight = 10.0
    else:
        return None

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [round(lon, 6), round(lat, 6)],
        },
        "properties":
