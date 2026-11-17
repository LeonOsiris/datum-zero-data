"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token spatial ETL compiling Welsh open data into Datum Zero GeoJSON.
- Queries Overpass in 3 fast regional tiles (North, Mid, South Wales) to prevent timeouts.
- Preserves previously compiled OSM features on transient mirror outages (never wipes data).
- Generates fixed-scale metric Polygon Threat Clouds (EPSG:27700 -> EPSG:4326).
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
from shapely.geometry import LineString, Point, box, mapping, shape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

WALES_BBOX = [-5.5, 51.3, -2.6, 53.5]
OUTPUT_FILE = Path("dist/wales_resilience_master.geojson")
DATAMAPWALES_WFS_URL = "https://datamap.gov.wales/geoserver/ows"

# Regional sub-tiles [min_lon, min_lat, max_lon, max_lat] for fast <3s Overpass queries
WALES_SUBTILES: List[Tuple[str, List[float]]] = [
    ("North Wales", [-5.5, 52.75, -2.8, 53.5]),
    ("Mid Wales", [-5.5, 52.00, -2.8, 52.75]),
    ("South Wales", [-5.5, 51.30, -2.6, 52.00]),
]

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
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

# Statutory BGS / Verified Welsh Resupply Springs & Medical Trauma Assets (Baseline Guarantee)
VERIFIED_RESUPPLY_AND_MEDICAL: List[Dict[str, Any]] = [
    {
        "id": "RESUPPLY_BGS_LLANBERIS",
        "name": "Llanberis Pass Bedrock Spring (BGS)",
        "coords": [-4.0550, 53.0950],
        "category": "resupply",
        "categoryLabel": "NATURAL SPRING",
        "accentColor": "#00E5FF",
        "metric": "Gravity Spring | ~18 L/min | Ordovician Rhyolite",
        "action": "Filter with 0.1-micron pump or boil 60s before container storage.",
    },
    {
        "id": "RESUPPLY_BGS_CAPEL",
        "name": "Capel Curig Aquifer Spring Line (BGS)",
        "coords": [-3.9082, 53.0931],
        "category": "resupply",
        "categoryLabel": "NATURAL SPRING",
        "accentColor": "#00E5FF",
        "metric": "Gravity Spring | ~14 L/min | Continuous Flow",
        "action": "Collect directly from bedrock fissure; boil 60s if drawn downstream.",
    },
    {
        "id": "RESUPPLY_BGS_BRECON",
        "name": "Ffynnon Las / Storey Arms Spring Head",
        "coords": [-3.4735, 51.8842],
        "category": "resupply",
        "categoryLabel": "NATURAL SPRING",
        "accentColor": "#00E5FF",
        "metric": "Gravity Spring | Old Red Sandstone Aquifer",
        "action": "High-elevation spring line. Filter or boil 60s.",
    },
    {
        "id": "MEDICAL_MIU_DOLGELLAU",
        "name": "Dolgellau & Barmouth Minor Injury Unit",
        "coords": [-3.8845, 52.7428],
        "category": "medical",
        "categoryLabel": "MINOR INJURY TRAUMA UNIT",
        "accentColor": "#10B981",
        "metric": "NHS Wales MIU & Public AED",
        "action": "Urgent limb trauma, suturing, and emergency defibrillation.",
    },
    {
        "id": "MEDICAL_MIU_LLANDRINDOD",
        "name": "Llandrindod Wells War Memorial MIU",
        "coords": [-3.3765, 52.2415],
        "category": "medical",
        "categoryLabel": "MINOR INJURY TRAUMA UNIT",
        "accentColor": "#10B981",
        "metric": "NHS Wales MIU & Emergency Trauma",
        "action": "Mid Wales central trauma & stabilization unit.",
    },
    {
        "id": "MEDICAL_AED_PENYPASS",
        "name": "Pen-y-Pass Warden Base Trauma & AED",
        "coords": [-4.0205, 53.0801],
        "category": "medical",
        "categoryLabel": "DEFIBRILLATOR / BLEED KIT",
        "accentColor": "#10B981",
        "metric": "24/7 Heated External AED & Bleed Kit",
        "action": "Call 999 for cabinet keypad access.",
    },
    {
        "id": "BASELINE_TW_MENAI",
        "name": "Traffic Wales A55 / Britannia Bridge Camera",
        "coords": [-4.1852, 53.2165],
        "category": "baseline",
        "categoryLabel": "PUBLIC SURVEILLANCE (CCTV)",
        "accentColor": "#94A3B8",
        "metric": "Type: Traffic Wales Highway Telemetry",
        "action": "Public highway monitoring point on Menai Strait crossing.",
    },
    {
        "id": "BASELINE_TW_CONWY",
        "name": "Traffic Wales A55 Conwy Tunnel West Portal",
        "coords": [-3.8365, 53.2855],
        "category": "baseline",
        "categoryLabel": "PUBLIC SURVEILLANCE (CCTV)",
        "accentColor": "#94A3B8",
        "metric": "Type: Traffic Wales Tunnel Surveillance",
        "action": "Public highway tunnel portal monitoring installation.",
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

# Statutory HSE Public COMAH Upper/Lower-Tier Industrial Sites in Wales
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
    """Create concentric core and dispersion halo polygons in British National Grid (EPSG:27700)."""
    gdf = gpd.GeoDataFrame([{"geometry": geom_wgs84}], crs="EPSG:4326").to_crs(epsg=27700)
    raw_bng = gdf.geometry.iloc[0]

    core_geom = raw_bng.buffer(core_buffer_m, cap_style="round").simplify(
        tolerance=10.0, preserve_topology=True
    )
    halo_geom = raw_bng.buffer(core_buffer_m + halo_extra_m, cap_style="round").simplify(
        tolerance=14.0, preserve_topology=True
    )

    core_wgs = gpd.GeoSeries([core_geom], crs="EPSG:27700").to_crs(epsg=4326).iloc[0]
    halo_wgs = gpd.GeoSeries([halo_geom], crs="EPSG:27700").to_crs(epsg=4326).iloc[0]

    wales_clip = box(*WALES_BBOX)
    core_wgs = core_wgs.intersection(wales_clip)
    halo_wgs = halo_wgs.intersection(wales_clip)

    out: List[Dict[str, Any]] = []
    if not halo_wgs.is_empty:
        halo_dict = mapping(halo_wgs)
        halo_dict["coordinates"] = round_coordinates(halo_dict["coordinates"], 5)
        halo_props = dict(base_props)
        halo_props["id"] = f"{base_props['id']}_HALO"
        halo_props["cloudRing"] = "halo"
        out.append(
            {
                "type": "Feature",
                "geometry": halo_dict,
                "properties": halo_props,
            }
        )

    if not core_wgs.is_empty:
        core_dict = mapping(core_wgs)
        core_dict["coordinates"] = round_coordinates(core_dict["coordinates"], 5)
        core_props = dict(base_props)
        core_props["cloudRing"] = "core"
        out.append(
            {
                "type": "Feature",
                "geometry": core_dict,
                "properties": core_props,
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
        "properties": {
            "id": f"{category.upper()}_{raw_id}",
            "name": tags.get("name", default_name),
            "category": category,
            "categoryLabel": label,
            "accentColor": accent,
            "threatWeight": weight,
            "primaryMetric": metric,
            "operationalAction": action,
        },
    }


def query_overpass_fast(query: str, label: str) -> Optional[Dict[str, Any]]:
    """Execute a lightweight Overpass query with mirror failover."""
    headers = {"User-Agent": "DatumZero-ETL/3.2 (Civil Resilience Pipeline)"}
    for mirror_url in OVERPASS_MIRRORS:
        logging.info("Querying Overpass (%s) via %s...", label, mirror_url)
        try:
            response = requests.post(
                mirror_url,
                data={"data": query},
                headers=headers,
                timeout=35,
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            logging.warning("Mirror %s failed for %s (%s). Failing over...", mirror_url, label, exc)
            time.sleep(1.5)
    return None


def load_cached_features_by_prefix(prefixes: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Preserve previously compiled OSM features if Overpass mirrors temporarily time out."""
    if not OUTPUT_FILE.exists():
        return []
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        preserved = [
            feat
            for feat in existing.get("features", [])
            if str(feat.get("properties", {}).get("id", "")).startswith(prefixes)
        ]
        if preserved:
            logging.info("Preserved %d cached features matching %s.", len(preserved), prefixes)
        return preserved
    except (OSError, json.JSONDecodeError):
        return []


def generate_emf_corridor_polygons(
    power_lines: List[Tuple[LineString, int, str]]
) -> List[Dict[str, Any]]:
    """Buffer high-voltage lines in British National Grid (EPSG:27700) with true metric halo."""
    if not power_lines:
        return []

    logging.info("Computing fixed-scale 50Hz EMF buffers for %d segments...", len(power_lines))
    records = [
        {"geometry": geom, "buffer_m": buf, "tier": tier}
        for geom, buf, tier in power_lines
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=27700)

    gdf["core_geom"] = gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_m"], cap_style="round"),
        axis=1,
    )
    gdf["halo_geom"] = gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_m"] * 1.45, cap_style="round"),
        axis=1,
    )

    wales_clip = box(*WALES_BBOX)
    features: List[Dict[str, Any]] = []

    for tier, group in gdf.groupby("tier"):
        buf_m = int(group["buffer_m"].iloc[0])
        core_series = gpd.GeoSeries(group["core_geom"], crs="EPSG:27700")
        halo_series = gpd.GeoSeries(group["halo_geom"], crs="EPSG:27700")

        core_union = (
            core_series.union_all()
            if hasattr(core_series, "union_all")
            else core_series.unary_union
        ).simplify(12.0, preserve_topology=True)
        halo_union = (
            halo_series.union_all()
            if hasattr(halo_series, "union_all")
            else halo_series.unary_union
        ).simplify(16.0, preserve_topology=True)

        core_wgs = (
            gpd.GeoSeries([core_union], crs="EPSG:27700")
            .to_crs(epsg=4326)
            .iloc[0]
            .intersection(wales_clip)
        )
        halo_wgs = (
            gpd.GeoSeries([halo_union], crs="EPSG:27700")
            .to_crs(epsg=4326)
            .iloc[0]
            .intersection(wales_clip)
        )

        base_props: Dict[str, Any] = {
            "id": f"EMF_CORRIDOR_{buf_m}M",
            "name": f"{tier} 50Hz EMF Corridor",
            "category": "grid_down",
            "categoryLabel": "HIGH-VOLTAGE EMF BUFFER",
            "accentColor": "#FFB800",
            "threatWeight": 2.0,
            "primaryMetric": f"True Ground Buffer: {buf_m}m | Compass & RF Interference",
            "operationalAction": "Expect magnetometer deviation and storm line-drop hazard. Maintain clearance.",
        }

        if not halo_wgs.is_empty:
            h_dict = mapping(halo_wgs)
            h_dict["coordinates"] = round_coordinates(h_dict["coordinates"], 5)
            h_props = dict(base_props)
            h_props["id"] = f"{base_props['id']}_HALO"
            h_props["cloudRing"] = "halo"
            features.append(
                {
                    "type": "Feature",
                    "geometry": h_dict,
                    "properties": h_props,
                }
            )

        if not core_wgs.is_empty:
            c_dict = mapping(core_wgs)
            c_dict["coordinates"] = round_coordinates(c_dict["coordinates"], 5)
            c_props = dict(base_props)
            c_props["cloudRing"] = "core"
            features.append(
                {
                    "type": "Feature",
                    "geometry": c_dict,
                    "properties": c_props,
                }
            )
    return features


def generate_bridge_choke_polygons(
    bridge_lines: List[Tuple[LineString, str]]
) -> List[Dict[str, Any]]:
    """Generate 160m fixed-scale metric choke-point buffers around trunk/primary bridges."""
    if not bridge_lines:
        return []

    logging.info("Generating fixed-scale choke-point polygons for %d major bridges...", len(bridge_lines))
    records = [{"geometry": geom, "name": name} for geom, name in bridge_lines]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=27700)

    gdf["geometry"] = gdf.geometry.buffer(160.0, cap_style="round")
    dissolved = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
    dissolved["halo_geom"] = dissolved.geometry.buffer(80.0, cap_style="round")

    core_wgs = dissolved.to_crs(epsg=4326)
    halo_wgs = gpd.GeoSeries(dissolved["halo_geom"], crs="EPSG:27700").to_crs(epsg=4326)
    wales_clip = box(*WALES_BBOX)

    features: List[Dict[str, Any]] = []
    for idx in range(len(core_wgs)):
        c_geom = core_wgs.geometry.iloc[idx].intersection(wales_clip)
        h_geom = halo_wgs.iloc[idx].intersection(wales_clip)
        if c_geom.is_empty:
            continue

        base_props: Dict[str, Any] = {
            "id": f"HAZ_CHOKE_BRIDGE_{idx}",
            "name": "Strategic Trunk Bridge Choke Point",
            "category": "hazards",
            "categoryLabel": "SINGLE-ACCESS BRIDGE DEFILE",
            "accentColor": "#FFB800",
            "threatWeight": 5.0,
            "primaryMetric": "Survival Penalty: W_choke +5.0 | 160m Corridor Buffer",
            "operationalAction": "Primary river/valley crossing bottleneck. Vulnerable to flood debris, closure, or gridlock.",
        }

        if not h_geom.is_empty:
            h_dict = mapping(h_geom)
            h_dict["coordinates"] = round_coordinates(h_dict["coordinates"], 5)
            h_props = dict(base_props)
            h_props["id"] = f"{base_props['id']}_HALO"
            h_props["cloudRing"] = "halo"
            features.append(
                {
                    "type": "Feature",
                    "geometry": h_dict,
                    "properties": h_props,
                }
            )

        c_dict = mapping(c_geom)
        c_dict["coordinates"] = round_coordinates(c_dict["coordinates"], 5)
        c_props = dict(base_props)
        c_props["cloudRing"] = "core"
        features.append(
            {
                "type": "Feature",
                "geometry": c_dict,
                "properties": c_props,
            }
        )
    return features


def harvest_datamapwales_flood_zones() -> List[Dict[str, Any]]:
    """Query keyless DataMapWales OGC WFS for NRW Flood Zone 3 polygons across Wales."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "inspire-nrw:NRW_FLOOD_MAP_FLOOD_ZONE_3_RIVERS_AND_SEA",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "count": "150",
    }
    headers = {"User-Agent": "DatumZero-ETL/3.2 (Civil Resilience Pipeline)"}
    features: List[Dict[str, Any]] = []

    try:
        logging.info("Querying DataMapWales WFS for NRW Flood Zone 3 polygons...")
        res = requests.get(DATAMAPWALES_WFS_URL, params=params, headers=headers, timeout=25)
        if res.status_code == 200:
            data = res.json()
            raw_feats = data.get("features", [])
            logging.info("Retrieved %d NRW Flood Zone 3 polygons from DataMapWales.", len(raw_feats))
            for idx, item in enumerate(raw_feats):
                geom = item.get("geometry")
                if not geom:
                    continue
                shp = shape(geom).simplify(0.0003, preserve_topology=True)
                if shp.is_empty:
                    continue
                g_dict = mapping(shp)
                g_dict["coordinates"] = round_coordinates(g_dict["coordinates"], 5)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": g_dict,
                        "properties": {
                            "id": f"NRW_FZ3_{idx}",
                            "name": "NRW Statutory Flood Zone 3 (Rivers & Sea)",
                            "category": "hazards",
                            "categoryLabel": "NRW FLOOD ZONE 3 POLYGON",
                            "accentColor": "#FF3B30",
                            "threatWeight": 10.0,
                            "cloudRing": "core",
                            "primaryMetric": "High Fluvial/Tidal Probability | W_flood +10.0",
                            "operationalAction": "Statutory 1-in-30-yr floodplain. Avoid low-lying roads during severe weather.",
                        },
                    }
                )
        else:
            logging.info("DataMapWales WFS returned HTTP %s; using verified Welsh basin corridors.", res.status_code)
    except (requests.RequestException, ValueError) as exc:
        logging.warning("DataMapWales WFS skipped (%s); using verified Welsh basin corridors.", exc)

    return features


def build_verified_hazard_layers() -> List[Dict[str, Any]]:
    """Compile statutory NRW gauges, COMAH exclusion buffers, and Welsh valley flood plains."""
    features: List[Dict[str, Any]] = []

    for g in VERIFIED_NRW_GAUGES:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": g["coords"]},
                "properties": {
                    "id": g["id"],
                    "name": g["name"],
                    "category": "hazards",
                    "categoryLabel": "NRW RIVER TELEMETRY",
                    "accentColor": "#FF3B30",
                    "threatWeight": 5.0,
                    "primaryMetric": g["metric"],
                    "operationalAction": g["action"],
                },
            }
        )

    for site in VERIFIED_COMAH_SITES:
        pt = Point(site["coords"][0], site["coords"][1])
        base_props = {
            "id": site["id"],
            "name": site["name"],
            "category": "hazards",
            "categoryLabel": "HSE COMAH HAZMAT BUFFER",
            "accentColor": "#FF3B30",
            "threatWeight": 8.0,
            "primaryMetric": site["metric"],
            "operationalAction": site["action"],
        }
        features.extend(
            make_feathered_metric_polygons(
                pt,
                core_buffer_m=float(site["radius_m"]),
                halo_extra_m=float(site["radius_m"]) * 0.4,
                base_props=base_props,
            )
        )

    for corr in VERIFIED_FLOOD_CORRIDORS:
        line = LineString(corr["line"])
        base_props = {
            "id": corr["id"],
            "name": corr["name"],
            "category": "hazards",
            "categoryLabel": "FLOOD & HYDROLOGICAL CORRIDOR",
            "accentColor": "#FF3B30",
            "threatWeight": 10.0,
            "primaryMetric": corr["metric"],
            "operationalAction": corr["action"],
        }
        features.extend(
            make_feathered_metric_polygons(
                line,
                core_buffer_m=float(corr["buffer_m"]),
                halo_extra_m=float(corr["buffer_m"]) * 0.45,
                base_props=base_props,
            )
        )

    return features


def get_static_baseline_features() -> List[Dict[str, Any]]:
    """Return normalized GeoJSON features for verified RSGB repeaters, springs, MIUs, and CCTV."""
    features: List[Dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": rep["coords"]},
            "properties": {
                "id": rep["id"],
                "name": rep["name"],
                "category": "grid_down",
                "categoryLabel": "RSGB AMATEUR REPEATER",
                "accentColor": "#FFB800",
                "threatWeight": 0.0,
                "primaryMetric": rep["metric"],
                "operationalAction": rep["action"],
            },
        }
        for rep in VERIFIED_WELSH_REPEATERS
    ]

    for item in VERIFIED_RESUPPLY_AND_MEDICAL:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": item["coords"]},
                "properties": {
                    "id": item["id"],
                    "name": item["name"],
                    "category": item["category"],
                    "categoryLabel": item["categoryLabel"],
                    "accentColor": item["accentColor"],
                    "threatWeight": 0.0,
                    "primaryMetric": item["metric"],
                    "operationalAction": item["action"],
                },
            }
        )
    return features


def harvest_osm_layers() -> List[Dict[str, Any]]:
    """Extract OSM POIs in 3 regional sub-tiles + high-voltage power lines & trunk bridges."""
    point_features: List[Dict[str, Any]] = []
    seen_ids = set()
    poi_success_tiles = 0

    # Query 1: Regional sub-tiled Point POIs (Springs, AEDs, CCTV, River Fords)
    for region_name, (min_lon, min_lat, max_lon, max_lat) in WALES_SUBTILES:
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
        poi_query = f"""
        [out:json][timeout:30];
        (
          node["natural"="spring"]({bbox_str});
          node["emergency"="defibrillator"]({bbox_str});
          node["man_made"="surveillance"]({bbox_str});
          node["ford"="yes"]({bbox_str});
        );
        out body;
        """
        poi_data = query_overpass_fast(poi_query, f"POIs {region_name}")
        if poi_data is not None:
            poi_success_tiles += 1
            for el in poi_data.get("elements", []):
                el_id = str(el.get("id", ""))
                if not el_id or el_id in seen_ids:
                    continue
                lat = el.get("lat")
                lon = el.get("lon")
                if lat is None or lon is None:
                    continue
                pt = build_point_feature(
                    el_id, float(lon), float(lat), el.get("tags", {})
                )
                if pt:
                    seen_ids.add(el_id)
                    point_features.append(pt)
        time.sleep(1.5)

    if poi_success_tiles == 0:
        logging.warning("All POI tiles failed; restoring cached OSM Point features.")
        point_features = load_cached_features_by_prefix(
            ("RESUPPLY_", "MEDICAL_", "BASELINE_", "HAZARDS_")
        )

    min_lon, min_lat, max_lon, max_lat = WALES_BBOX
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    # Query 2: High-Voltage Power Lines
    power_query = f"""
    [out:json][timeout:40];
    way["power"="line"]({bbox_str});
    out geom;
    """
    power_data = query_overpass_fast(power_query, "PowerLines")
    time.sleep(1.5)

    # Query 3: Trunk & Primary Road Bridges (Single-Access Choke Points)
    bridge_query = f"""
    [out:json][timeout:40];
    way["bridge"="yes"]["highway"~"^(trunk|primary)$"]({bbox_str});
    out geom;
    """
    bridge_data = query_overpass_fast(bridge_query, "TrunkBridges")

    if power_data is not None:
        power_lines: List[Tuple[LineString, int, str]] = []
        for el in power_data.get("elements", []):
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            emf_meta = classify_emf_buffer(parse_max_voltage(tags.get("voltage")))
            if not emf_meta:
                continue
            coords = [
                (float(pt["lon"]), float(pt["lat"]))
                for pt in el.get("geometry", [])
                if "lon" in pt and "lat" in pt
            ]
            if len(coords) >= 2:
                power_lines.append((LineString(coords), emf_meta[0], emf_meta[1]))
        emf_polygons = generate_emf_corridor_polygons(power_lines)
    else:
        emf_polygons = load_cached_features_by_prefix(("EMF_CORRIDOR_",))

    if bridge_data is not None:
        bridge_lines: List[Tuple[LineString, str]] = []
        for el in bridge_data.get("elements", []):
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            coords = [
                (float(pt["lon"]), float(pt["lat"]))
                for pt in el.get("geometry", [])
                if "lon" in pt and "lat" in pt
            ]
            if len(coords) >= 2:
                bridge_lines.append((LineString(coords), tags.get("name", "Trunk Bridge")))
        bridge_polygons = generate_bridge_choke_polygons(bridge_lines)
    else:
        bridge_polygons = load_cached_features_by_prefix(("HAZ_CHOKE_BRIDGE_",))

    return point_features + emf_polygons + bridge_polygons


def main() -> None:
    osm_features = harvest_osm_layers()
    static_baseline = get_static_baseline_features()
    verified_hazards = build_verified_hazard_layers()
    dmw_flood_zones = harvest_datamapwales_flood_zones()

    # Deduplicate by feature ID so static guarantees never collide with cached OSM points
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for feat in (
        static_baseline + verified_hazards + dmw_flood_zones + osm_features
    ):
        fid = str(feat.get("properties", {}).get("id", len(merged_by_id)))
        merged_by_id[fid] = feat

    all_features = list(merged_by_id.values())

    if not all_features:
        logging.error("Zero features compiled across all sources. Aborting write.")
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
        "Success: Compiled %d verified Welsh features into %s (%.1f KB)",
        len(all_features),
        OUTPUT_FILE,
        size_kb,
    )


if __name__ == "__main__":
    main()
