"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token spatial ETL compiling Welsh open data into Datum Zero GeoJSON.

Dual-Layer Readability Architecture:
Every feature emits two tiers of text in `properties`:
1. Immediate Peek Tier (`plainBadge`, `plainSummary`):
   - Plain, everyday English with zero GIS, RF, or geological jargon.
   - Immediately readable in 2 seconds when tapping a point or cloud on mobile.
2. Expanded Technical Tier (`categoryLabel`, `primaryMetric`, `operationalAction`, `threatWeight`):
   - Full specialist telemetry (VHF frequencies/CTCSS, BGS geology, NRW stage thresholds,
     Biot-Savart EMF radii, UKHSA radon bands, statutory WCA 1981 / Theft Act 1968 rules)
     revealed when the user expands the bottom sheet.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import sys
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
DATAMAPWALES_WFS_URL = "https://datamap.gov.wales/geoserver/ows"
DEFRA_UKAIR_STATIONS_URL = "https://uk-air.defra.gov.uk/sos-ukair/api/v1/stations"

# 7 balanced regional tiles [min_lon, min_lat, max_lon, max_lat]
WALES_SUBTILES: List[Tuple[str, List[float]]] = [
    ("North-West Wales (Anglesey & Llyn)", [-5.50, 52.75, -4.18, 53.50]),
    ("North-Central Wales (Eryri & Conwy)", [-4.18, 52.75, -3.60, 53.50]),
    ("North-East Wales (Clwyd & Dee)", [-3.60, 52.75, -2.80, 53.50]),
    ("Mid-West Wales (Ceredigion)", [-5.50, 52.00, -3.70, 52.75]),
    ("Mid-East Wales (Powys)", [-3.70, 52.00, -2.90, 52.75]),
    ("South-West Wales (Pembroke & Carmarthen)", [-5.50, 51.30, -4.15, 52.00]),
    ("South-Central Wales (Swansea & Neath)", [-4.15, 51.30, -3.45, 52.00]),
    ("South-East Wales (Cardiff, Newport & Valleys)", [-3.45, 51.30, -2.60, 52.00]),
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
        "name": "GB3AR - Bangor Mountain Radio Repeater",
        "coords": [-4.1262, 53.2214],
        "plainSummary": "Backup emergency two-way radio relay with battery power for when mobile networks fail.",
        "metric": "TX: 145.725 MHz | RX: 145.125 MHz | Offset: -0.6 MHz | CTCSS: 103.5 Hz",
        "action": "2m analog voice repeater (-0.6 MHz shift). Solar + deep-cycle AGM battery backup monitored by North Wales RAYNET.",
    },
    {
        "id": "COM_GB3GW",
        "name": "GB3GW - Great Orme Radio Repeater",
        "coords": [-3.8550, 53.3320],
        "plainSummary": "Coastal emergency two-way radio relay covering Conwy Bay, Llandudno, and Anglesey.",
        "metric": "TX: 433.000 MHz | RX: 434.600 MHz | Offset: +1.6 MHz | CTCSS: 103.5 Hz",
        "action": "70cm coastal analog voice repeater (+1.6 MHz shift). Long-range line-of-sight across Conwy Bay & Menai Strait.",
    },
    {
        "id": "COM_GB3LL",
        "name": "GB3LL - Llanddona Anglesey Radio Repeater",
        "coords": [-4.1350, 53.2940],
        "plainSummary": "Primary Anglesey emergency two-way radio relay reaching across to Caernarfon.",
        "metric": "TX: 145.775 MHz | RX: 145.175 MHz | Offset: -0.6 MHz | CTCSS: 103.5 Hz",
        "action": "2m primary Anglesey repeater (-0.6 MHz shift). Line-of-sight to Caernarfon and north Eryri coast.",
    },
    {
        "id": "COM_GB3WW",
        "name": "GB3WW - Preseli Hills Radio Repeater",
        "coords": [-4.7731, 51.9462],
        "plainSummary": "Wide-area West Wales emergency two-way radio relay atop the Preseli ridge.",
        "metric": "TX: 145.625 MHz | RX: 145.025 MHz | Offset: -0.6 MHz | CTCSS: 77.0 Hz",
        "action": "2m wide-area West Wales repeater (-0.6 MHz shift) covering Pembrokeshire and Carmarthenshire.",
    },
    {
        "id": "COM_GB3SW",
        "name": "GB3SW - Kilvey Hill Radio Repeater (Swansea)",
        "coords": [-3.9167, 51.6247],
        "plainSummary": "Emergency two-way radio relay covering Swansea Bay and the Gower peninsula.",
        "metric": "TX: 145.675 MHz | RX: 145.075 MHz | Offset: -0.6 MHz | CTCSS: 110.9 Hz",
        "action": "2m South Wales coastal & Gower repeater (-0.6 MHz shift).",
    },
    {
        "id": "COM_GB3MH",
        "name": "GB3MH - Mynydd Machen Radio Repeater",
        "coords": [-3.1194, 51.5942],
        "plainSummary": "Emergency two-way radio relay serving the Rhondda, Cynon, and Taff valleys.",
        "metric": "TX: 145.650 MHz | RX: 145.050 MHz | Offset: -0.6 MHz | CTCSS: 110.9 Hz",
        "action": "2m repeater serving South Wales valley corridors and Newport approaches.",
    },
    {
        "id": "COM_GB3MW",
        "name": "GB3MW - Aberystwyth / Plynlimon Radio Repeater",
        "coords": [-3.9825, 52.4142],
        "plainSummary": "Mid-Wales emergency two-way radio relay covering Cardigan Bay and the Cambrian hills.",
        "metric": "TX: 145.600 MHz | RX: 145.000 MHz | Offset: -0.6 MHz | CTCSS: 77.0 Hz",
        "action": "2m Mid-Wales coastal & Cambrian Mountains repeater (-0.6 MHz shift).",
    },
]

# Topological Cellular Dead Zones
VERIFIED_CELLULAR_DEAD_ZONES: List[Dict[str, Any]] = [
    {
        "id": "SIG_DEAD_OGWEN",
        "name": "Ogwen Valley — No Mobile Signal",
        "line": [[-4.0560, 53.1680], [-4.0340, 53.1440], [-4.0160, 53.1240], [-3.9880, 53.1120], [-3.9580, 53.1030]],
        "buffer_m": 950,
        "plainSummary": "No mobile phone signal or mobile data in this steep valley. Download maps before entering.",
        "metric": "4-MNO Blackout (EE/Three/O2/Vodafone) | Routing Penalty: W_signal +2.5",
        "action": "Deep glaciated trough blocks mast line-of-sight. Use 2m VHF 145.500 MHz or GB3AR (145.725 MHz).",
    },
    {
        "id": "SIG_DEAD_LLANBERIS_PASS",
        "name": "Llanberis Pass — No Mobile Signal",
        "line": [[-4.1080, 53.1110], [-4.0750, 53.0990], [-4.0420, 53.0890], [-4.0120, 53.0800], [-3.9820, 53.0830]],
        "buffer_m": 900,
        "plainSummary": "No mobile phone signal or 999 voice reception inside this steep mountain pass.",
        "metric": "4-MNO Blackout | Glaciated Slate Gorge | Routing Penalty: W_signal +2.5",
        "action": "Zero mobile data or 999 voice inside pass. Climb above 450m ridge or relay via Capel Curig.",
    },
    {
        "id": "SIG_DEAD_RHINOGYDD",
        "name": "Rhinogydd & Coed y Brenin — No Mobile Signal",
        "line": [[-4.0220, 52.8740], [-3.9850, 52.8480], [-3.9420, 52.8220], [-3.9120, 52.7960], [-3.8880, 52.7750]],
        "buffer_m": 1500,
        "plainSummary": "Wide mountain area with no mobile phone reception across any UK network.",
        "metric": "4-MNO Blackout | Cambrian Shielding | Routing Penalty: W_signal +2.5",
        "action": "Persistent upland shadow across Ardudwy range. Pre-cache offline route before entering.",
    },
    {
        "id": "SIG_DEAD_BERWYN",
        "name": "Berwyn Mountains — No Mobile Signal",
        "line": [[-3.6050, 52.8910], [-3.5680, 52.8710], [-3.5320, 52.8520], [-3.4920, 52.8360], [-3.4550, 52.8210]],
        "buffer_m": 1400,
        "plainSummary": "Remote high moorland road with zero phone signal and long stretches without shelter.",
        "metric": "4-MNO Blackout | High Moorland Defile | Routing Penalty: W_signal +2.5",
        "action": "Zero cellular coverage along B4391 pass. Maintain BLE stand-off mesh with convoy.",
    },
    {
        "id": "SIG_DEAD_ELAN_CAMBRIAN",
        "name": "Elan Valley Wilderness — No Mobile Signal",
        "line": [[-3.7550, 52.3520], [-3.7020, 52.3180], [-3.6480, 52.2820], [-3.6040, 52.2540], [-3.5680, 52.2280]],
        "buffer_m": 1900,
        "plainSummary": "Large Mid-Wales upland zone with no mobile signal or internet access.",
        "metric": "4-MNO Blackout | Cambrian Plateau | Routing Penalty: W_signal +2.5",
        "action": "Largest continuous cellular shadow in Mid Wales. Rely on offline .pmtiles & VHF simplex.",
    },
    {
        "id": "SIG_DEAD_BLACK_MOUNTAIN",
        "name": "Black Mountain Pass — No Mobile Signal",
        "line": [[-3.8540, 51.8720], [-3.8160, 51.8440], [-3.7780, 51.8210], [-3.7420, 51.8040], [-3.7080, 51.7910]],
        "buffer_m": 1350,
        "plainSummary": "Steep hills block mobile phone towers here. Climb to higher ridge ground for signal.",
        "metric": "4-MNO Blackout | A4069 & Usk Headwaters | Routing Penalty: W_signal +2.5",
        "action": "Steep escarpment blocks South Wales masts. Gain northern ridge crest for line-of-sight.",
    },
]

# High-Power Welsh Broadcast & RF Transmitter Sites
VERIFIED_RF_TRANSMITTERS: List[Dict[str, Any]] = [
    {
        "id": "RF_TX_WENVOE",
        "name": "Wenvoe Major TV & Radio Mast",
        "coords": [-3.2818, 51.4595],
        "radius_m": 850,
        "plainSummary": "High-power broadcast mast. Strong radio signals nearby can interfere with handheld radios.",
        "metric": "ERP: 500 kW UHF / VHF | 850m High RF Density Contour",
        "action": "Strong local RF field; potential front-end overload on unshielded handheld receivers within 850m.",
    },
    {
        "id": "RF_TX_MOELYPARC",
        "name": "Moel-y-Parc TV & Radio Mast",
        "coords": [-3.3145, 53.2255],
        "radius_m": 750,
        "plainSummary": "High hilltop broadcast mast with strong line-of-sight across North-East Wales.",
        "metric": "ERP: 100 kW | Clwydian Range Line-of-Sight Hub",
        "action": "Dominant North-East Wales RF beacon. Elevated ground provides optimal 2m/70cm simplex range.",
    },
    {
        "id": "RF_TX_BLAENPLWYF",
        "name": "Blaenplwyf Coastal Radio & TV Mast",
        "coords": [-4.1145, 52.3715],
        "radius_m": 750,
        "plainSummary": "Major coastal broadcast mast near Aberystwyth; expect strong radio interference close by.",
        "metric": "ERP: 100 kW | Cardigan Bay Maritime & Broadcast Hub",
        "action": "Primary Mid-Wales coastal transmitter. Expect elevated RF noise floor near perimeter.",
    },
    {
        "id": "RF_TX_LLANDDONA",
        "name": "Llanddona Anglesey Radio & TV Mast",
        "coords": [-4.1325, 53.2932],
        "radius_m": 700,
        "plainSummary": "Major hilltop broadcast mast on Anglesey co-located with emergency radio relay GB3LL.",
        "metric": "ERP: 100 kW | Co-located with GB3LL 2m Repeater",
        "action": "Covers Menai Strait, Eryri north coast, and Irish Sea approaches.",
    },
]

# Organic Multi-Vertex UKHSA / BGS Radon Gas Action Belts
VERIFIED_RADON_ZONES: List[Dict[str, Any]] = [
    {
        "id": "RADON_HALKYN_CLWYD",
        "name": "Halkyn & Clwydian Natural Radon Gas Area",
        "poly": [
            [-3.335, 53.268], [-3.265, 53.275], [-3.195, 53.258], [-3.135, 53.225],
            [-3.112, 53.175], [-3.148, 53.135], [-3.215, 53.142], [-3.268, 53.168],
            [-3.308, 53.212], [-3.335, 53.268],
        ],
        "plainSummary": "Natural underground radon gas is high here. Keep basements, caves, and low shelters well ventilated.",
        "metric": "UKHSA Class: >30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Carboniferous limestone karst releases elevated radon gas. Ventilate basements, caves, or subterranean shelters.",
    },
    {
        "id": "RADON_GWYNEDD_SLATE",
        "name": "Gwynedd Slate Belt Natural Radon Gas Area",
        "poly": [
            [-4.215, 53.178], [-4.125, 53.192], [-4.035, 53.165], [-3.975, 53.115],
            [-3.962, 53.055], [-4.028, 53.035], [-4.115, 53.048], [-4.178, 53.082],
            [-4.212, 53.132], [-4.215, 53.178],
        ],
        "plainSummary": "Rock here can release natural radon gas. Avoid sleeping in unventilated cellars or old mine tunnels.",
        "metric": "UKHSA Class: 10–30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Fractured metamorphic slate & granite bedrock. Avoid prolonged occupancy in unventilated mine adits or cellars.",
    },
    {
        "id": "RADON_PEMBROKE_GOWER",
        "name": "South Pembrokeshire & Gower Radon Gas Area",
        "poly": [
            [-5.045, 51.712], [-4.935, 51.725], [-4.815, 51.708], [-4.695, 51.685],
            [-4.662, 51.635], [-4.745, 51.612], [-4.865, 51.608], [-4.975, 51.628],
            [-5.035, 51.668], [-5.045, 51.712],
        ],
        "plainSummary": "Elevated natural radon gas in coastal limestone. Keep ground-level shelters well aired.",
        "metric": "UKHSA Class: 10–30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Coastal limestone fault zone. Ensure active cross-ventilation in ground-level or below-grade shelters.",
    },
    {
        "id": "RADON_VALLEYS_TAFF",
        "name": "South Wales Valleys Radon Gas Area",
        "poly": [
            [-3.515, 51.735], [-3.415, 51.748], [-3.305, 51.732], [-3.218, 51.695],
            [-3.205, 51.625], [-3.258, 51.582], [-3.355, 51.578], [-3.445, 51.608],
            [-3.498, 51.668], [-3.515, 51.735],
        ],
        "plainSummary": "Underground rock strata can trap natural radon gas in basements and enclosed ground floors.",
        "metric": "UKHSA Class: 10–30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Deep-mined strata & valley fault lines concentrate soil gas in enclosed lower storeys.",
    },
]

# Statutory SSSIs (Wildlife & Countryside Act 1981 — STRICT NO-FORAGING ZONES)
VERIFIED_SSSI_ZONES: List[Dict[str, Any]] = [
    {
        "id": "SSSI_ERYRI_CORE",
        "name": "Yr Wyddfa & Cwm Idwal Protected Nature Reserve",
        "poly": [
            [-4.105, 53.128], [-4.045, 53.142], [-3.992, 53.124], [-3.978, 53.082],
            [-4.005, 53.048], [-4.058, 53.044], [-4.095, 53.068], [-4.112, 53.098],
            [-4.105, 53.128],
        ],
        "plainSummary": "Strictly protected wildlife area. Do NOT pick plants, uproot anything, or disturb wildlife here.",
        "metric": "Statutory SSSI | Wildlife & Countryside Act 1981 (Section 28)",
        "action": "FORAGING PROHIBITED: High-Conservation SSSI Zone. No harvesting, uprooting, or disturbance permitted.",
    },
    {
        "id": "SSSI_CORS_CARON",
        "name": "Cors Caron Protected Bog Reserve (Tregaron)",
        "poly": [
            [-3.948, 52.266], [-3.918, 52.275], [-3.892, 52.262], [-3.888, 52.238],
            [-3.908, 52.218], [-3.938, 52.222], [-3.954, 52.242], [-3.948, 52.266],
        ],
        "plainSummary": "Protected deep peat bog. No plant picking allowed — stay strictly on the wooden boardwalks.",
        "metric": "Statutory SSSI & Ramsar Wetland | Wildlife & Countryside Act 1981",
        "action": "FORAGING PROHIBITED: Protected peatland habitat. Keep strictly to designated raised boardwalks.",
    },
    {
        "id": "SSSI_GOWER_COAST",
        "name": "Gower Cliffs Protected Coastal Reserve",
        "poly": [
            [-4.324, 51.578], [-4.278, 51.582], [-4.225, 51.568], [-4.198, 51.548],
            [-4.218, 51.532], [-4.268, 51.535], [-4.308, 51.548], [-4.328, 51.564],
            [-4.324, 51.578],
        ],
        "plainSummary": "Protected coastal cliff reserve. No plant harvesting or disturbing nesting seabirds.",
        "metric": "Statutory Coastal SSSI | Wildlife & Countryside Act 1981",
        "action": "FORAGING PROHIBITED: Protected maritime cliff flora and seabird nesting habitat.",
    },
]

# Ancient Woodland Foraging Biomes
VERIFIED_FORAGING_BIOMES: List[Dict[str, Any]] = [
    {
        "id": "FORAGE_COED_Y_BRENIN",
        "name": "Coed y Brenin Woodland (Shelter & Wild Edibles)",
        "poly": [
            [-3.918, 52.838], [-3.878, 52.846], [-3.842, 52.832], [-3.836, 52.798],
            [-3.862, 52.776], [-3.898, 52.782], [-3.922, 52.808], [-3.918, 52.838],
        ],
        "plainSummary": "Dense forest shelter with seasonal berries, nuts, and edible leaves. Never eat wild mushrooms.",
        "metric": "NRW Ancient Semi-Natural Woodland | Theft Act 1968 Compliant",
        "action": "Legal personal-use above-ground flora only (wild garlic, nettles, bilberries, hazelnuts). No uprooting. DO NOT harvest wild fungi/mushrooms.",
    },
    {
        "id": "FORAGE_GWYDYR_CONWY",
        "name": "Gwydyr Forest Woodland (Shelter & Wild Edibles)",
        "poly": [
            [-3.858, 53.118], [-3.818, 53.126], [-3.782, 53.112], [-3.776, 53.082],
            [-3.802, 53.066], [-3.838, 53.072], [-3.862, 53.095], [-3.858, 53.118],
        ],
        "plainSummary": "Good wind shelter, fallen firewood, and seasonal berries/leaves. Never pick wild mushrooms.",
        "metric": "Mixed Broadleaf/Conifer Canopy | Thermal Shelter & Flora",
        "action": "Dense canopy windbreak & fallen deadwood fuel. Personal above-ground foliage/berries only; strictly exclude wild mushrooms.",
    },
    {
        "id": "FORAGE_ELAN_OAKWOODS",
        "name": "Elan & Wye Oakwoods (Shelter & Wild Edibles)",
        "poly": [
            [-3.578, 52.288], [-3.535, 52.296], [-3.492, 52.282], [-3.486, 52.248],
            [-3.515, 52.226], [-3.555, 52.232], [-3.582, 52.258], [-3.578, 52.288],
        ],
        "plainSummary": "Oak and hazel woodland with seasonal hazelnuts, blackberries, and sorrel. Never eat wild mushrooms.",
        "metric": "Sessile Oak & Hazel Coppice | Theft Act 1968 Compliant",
        "action": "High seasonal mast (hazelnuts, blackberries, sorrel). Above-ground personal harvest only; zero wild fungi.",
    },
    {
        "id": "FORAGE_WENTWOOD",
        "name": "Wentwood Forest (Shelter & Wild Edibles)",
        "poly": [
            [-2.838, 51.656], [-2.775, 51.668], [-2.705, 51.652], [-2.696, 51.618],
            [-2.735, 51.596], [-2.795, 51.602], [-2.842, 51.628], [-2.838, 51.656],
        ],
        "plainSummary": "South Wales ancient broadleaf forest with wild garlic, elderberries, and hazel. Never eat wild mushrooms.",
        "metric": "Largest Ancient Woodland in South Wales | Broadleaf Canopy",
        "action": "Rich spring wild garlic, elderflower/elderberry, and hazel. Above-ground personal use only; exclude wild fungi.",
    },
]

# Designated Open Access Land (CRoW Act 2000)
VERIFIED_CROW_ACCESS_LAND: List[Dict[str, Any]] = [
    {
        "id": "CROW_CARNEDDAU_GLYDERAU",
        "name": "Carneddau & Glyderau Open Mountain Land",
        "poly": [
            [-4.055, 53.205], [-3.965, 53.218], [-3.872, 53.195], [-3.862, 53.145],
            [-3.895, 53.108], [-3.975, 53.112], [-4.038, 53.135], [-4.062, 53.172],
            [-4.055, 53.205],
        ],
        "plainSummary": "Public right to walk freely off-path across open mountain ground to bypass flooded valleys.",
        "metric": "CRoW Act 2000 Statutory Access | Mountain & Moorland",
        "action": "Legal right of foot access across uncultivated upland terrain for high-contour ridge bypasses.",
    },
    {
        "id": "CROW_BANNAU_CENTRAL",
        "name": "Central Brecon Beacons Open Mountain Land",
        "poly": [
            [-3.538, 51.912], [-3.445, 51.924], [-3.348, 51.905], [-3.338, 51.858],
            [-3.385, 51.828], [-3.465, 51.832], [-3.525, 51.855], [-3.545, 51.885],
            [-3.538, 51.912],
        ],
        "plainSummary": "Public right to walk freely off-path across open mountain ridges above valley flood zones.",
        "metric": "CRoW Act 2000 Statutory Access | Old Red Sandstone Escarpment",
        "action": "Legal off-trail upland movement zone above valley flood corridors.",
    },
    {
        "id": "CROW_PRESELI_HILLS",
        "name": "Preseli Hills Open Moorland",
        "poly": [
            [-4.858, 51.972], [-4.775, 51.984], [-4.685, 51.968], [-4.678, 51.935],
            [-4.725, 51.918], [-4.805, 51.922], [-4.855, 51.942], [-4.858, 51.972],
        ],
        "plainSummary": "Open moorland with public walking access and clear hilltop range for emergency radios.",
        "metric": "CRoW Act 2000 Statutory Access | West Wales Upland Corridor",
        "action": "Open moorland ridge corridor with long-range VHF line-of-sight.",
    },
]

# DEFRA UK-AIR / AURN Welsh Air Quality & Toxic Particulate Monitoring Stations
VERIFIED_DEFRA_AURN_STATIONS: List[Dict[str, Any]] = [
    {
        "id": "AURN_PORT_TALBOT",
        "name": "Port Talbot Air Quality & Smoke Sensor",
        "coords": [-3.7708, 51.5839],
        "plainSummary": "Government air monitor tracking industrial fumes and wildfire smoke levels.",
        "metric": "Live UK-AIR AURN Telemetry | PM2.5 & PM10 Particulates",
        "action": "Monitors steelworks, M4 corridor, and Afan valley wildfire smoke particulates.",
    },
    {
        "id": "AURN_CARDIFF",
        "name": "Cardiff Centre Air Quality & Smoke Sensor",
        "coords": [-3.1763, 51.4818],
        "plainSummary": "Government air monitor tracking city pollution and drifting wildfire smoke.",
        "metric": "Live UK-AIR AURN Telemetry | PM2.5 / NO2 / O3",
        "action": "Tracks South-East Wales urban air quality and Bristol Channel smoke drift.",
    },
    {
        "id": "AURN_SWANSEA",
        "name": "Swansea Valley Air Quality & Smoke Sensor",
        "coords": [-3.9474, 51.6326],
        "plainSummary": "Government air monitor tracking trapped valley smoke and winter smog.",
        "metric": "Live UK-AIR AURN Telemetry | PM2.5 & PM10 Particulates",
        "action": "Monitors particulate trapping during winter anticyclonic temperature inversions.",
    },
    {
        "id": "AURN_WREXHAM",
        "name": "Wrexham Air Quality & Smoke Sensor",
        "coords": [-2.9912, 53.0422],
        "plainSummary": "Government air monitor tracking North-East Wales industrial fumes and moorland wildfire smoke.",
        "metric": "Live UK-AIR AURN Telemetry | PM2.5 & PM10 Particulates",
        "action": "Detects Dee Valley inversions and Ruabon/Llantysilio moorland wildfire smoke.",
    },
    {
        "id": "AURN_ASTON_HILL",
        "name": "Aston Hill Rural Air Quality Sensor (Powys)",
        "coords": [-3.0341, 52.5038],
        "plainSummary": "Hilltop air monitor tracking regional smoke clouds drifting across Mid Wales.",
        "metric": "Live UK-AIR AURN Telemetry | Rural Background PM2.5 / O3",
        "action": "High-elevation rural particulate and ozone sensor tracking regional plume transport.",
    },
    {
        "id": "AURN_NARBERTH",
        "name": "Narberth Rural Air Quality Sensor (Pembrokeshire)",
        "coords": [-4.6915, 51.7818],
        "plainSummary": "West Wales air monitor tracking coastal wind and Milford Haven industrial air.",
        "metric": "Live UK-AIR AURN Telemetry | Maritime Background PM2.5",
        "action": "Monitors Atlantic inflow and Milford Haven industrial corridor downwind particulates.",
    },
]

# Statutory BGS / Verified Welsh Resupply Springs & Medical Trauma Assets
VERIFIED_RESUPPLY_AND_MEDICAL: List[Dict[str, Any]] = [
    {
        "id": "RESUPPLY_BGS_LLANBERIS",
        "name": "Llanberis Pass Natural Mountain Spring",
        "coords": [-4.0550, 53.0950],
        "category": "resupply",
        "plainBadge": "NATURAL WATER SPRING",
        "categoryLabel": "BGS BEDROCK WATER SPRING",
        "accentColor": "#00E5FF",
        "plainSummary": "Clean mountain spring water flowing from rock. Filter or boil for 1 minute before drinking.",
        "metric": "Flow: ~18 L/min | Gravity Head | Ordovician Rhyolite Bedrock",
        "action": "Filter with 0.1-micron pump or boil 60s before container storage.",
    },
    {
        "id": "RESUPPLY_BGS_CAPEL",
        "name": "Capel Curig Natural Spring Line",
        "coords": [-3.9082, 53.0931],
        "category": "resupply",
        "plainBadge": "NATURAL WATER SPRING",
        "categoryLabel": "BGS BEDROCK WATER SPRING",
        "accentColor": "#00E5FF",
        "plainSummary": "Year-round natural spring water. Collect straight from the rock and boil or filter.",
        "metric": "Flow: ~14 L/min | Continuous Groundwater Discharge",
        "action": "Collect directly from bedrock fissure; boil 60s if drawn downstream.",
    },
    {
        "id": "RESUPPLY_BGS_BRECON",
        "name": "Storey Arms / Ffynnon Las Mountain Spring",
        "coords": [-3.4735, 51.8842],
        "category": "resupply",
        "plainBadge": "NATURAL WATER SPRING",
        "categoryLabel": "BGS BEDROCK WATER SPRING",
        "accentColor": "#00E5FF",
        "plainSummary": "High-ground natural spring near Pen y Fan. Filter or boil for 1 minute before drinking.",
        "metric": "High-Elevation Spring | Old Red Sandstone Aquifer",
        "action": "High-elevation spring line. Filter or boil 60s.",
    },
    {
        "id": "RESUPPLY_BGS_PRESELI",
        "name": "Preseli Ridge Natural Spring",
        "coords": [-4.7420, 51.9585],
        "category": "resupply",
        "plainBadge": "NATURAL WATER SPRING",
        "categoryLabel": "BGS BEDROCK WATER SPRING",
        "accentColor": "#00E5FF",
        "plainSummary": "Natural upland spring water in West Wales. Boil or filter before storing.",
        "metric": "Flow: ~12 L/min | Upland Igneous Outcrop",
        "action": "Filtered upland discharge. Boil 60s or filter prior to storage.",
    },
    {
        "id": "MEDICAL_MIU_DOLGELLAU",
        "name": "Dolgellau Minor Injury & Trauma Clinic",
        "coords": [-3.8845, 52.7428],
        "category": "medical",
        "plainBadge": "URGENT INJURY CLINIC",
        "categoryLabel": "NHS WALES MINOR INJURY UNIT",
        "accentColor": "#10B981",
        "plainSummary": "NHS clinic for cuts, broken bones, burns, and emergency heart defibrillation.",
        "metric": "NHS Wales MIU & Public Access AED",
        "action": "Urgent limb trauma, suturing, and emergency defibrillation.",
    },
    {
        "id": "MEDICAL_MIU_LLANDRINDOD",
        "name": "Llandrindod Wells Minor Injury Clinic",
        "coords": [-3.3765, 52.2415],
        "category": "medical",
        "plainBadge": "URGENT INJURY CLINIC",
        "categoryLabel": "NHS WALES MINOR INJURY UNIT",
        "accentColor": "#10B981",
        "plainSummary": "Central Mid-Wales NHS urgent care clinic for injuries and emergency stabilization.",
        "metric": "NHS Wales MIU & Emergency Trauma",
        "action": "Mid Wales central trauma & stabilization unit.",
    },
    {
        "id": "MEDICAL_MIU_TENBY",
        "name": "Tenby Walk-In Injury & Trauma Clinic",
        "coords": [-4.7062, 51.6725],
        "category": "medical",
        "plainBadge": "URGENT INJURY CLINIC",
        "categoryLabel": "NHS WALES MINOR INJURY UNIT",
        "accentColor": "#10B981",
        "plainSummary": "South Pembrokeshire NHS walk-in injury clinic and emergency defibrillator.",
        "metric": "NHS Wales Community Trauma & AED",
        "action": "South Pembrokeshire urgent care and defibrillation point.",
    },
    {
        "id": "MEDICAL_AED_PENYPASS",
        "name": "Pen-y-Pass Defibrillator & Bleed Kit",
        "coords": [-4.0205, 53.0801],
        "category": "medical",
        "plainBadge": "DEFIBRILLATOR & BLEED KIT",
        "categoryLabel": "24/7 PUBLIC AED & TRAUMA KIT",
        "accentColor": "#10B981",
        "plainSummary": "24/7 emergency heart shock box and heavy bleeding kit on the warden building wall.",
        "metric": "24/7 Heated External AED & Bleed Kit",
        "action": "Call 999 for cabinet keypad access.",
    },
    {
        "id": "BASELINE_TW_MENAI",
        "name": "Britannia Bridge Highway Camera (A55)",
        "coords": [-4.1852, 53.2165],
        "category": "baseline",
        "plainBadge": "PUBLIC TRAFFIC CAMERA",
        "categoryLabel": "TRAFFIC WALES SURVEILLANCE",
        "accentColor": "#94A3B8",
        "plainSummary": "Public Traffic Wales road camera monitoring the main bridge onto Anglesey.",
        "metric": "Type: Traffic Wales Highway Telemetry",
        "action": "Public highway monitoring point on Menai Strait crossing.",
    },
    {
        "id": "BASELINE_TW_SEVERN",
        "name": "Prince of Wales Bridge Camera (M4)",
        "coords": [-2.7085, 51.5772],
        "category": "baseline",
        "plainBadge": "PUBLIC TRAFFIC CAMERA",
        "categoryLabel": "TRAFFIC WALES SURVEILLANCE",
        "accentColor": "#94A3B8",
        "plainSummary": "Public Traffic Wales road camera monitoring the main M4 Severn crossing.",
        "metric": "Type: Traffic Wales Strategic Crossing Camera",
        "action": "Monitors primary South Wales M4 arterial river crossing.",
    },
]

# Statutory NRW River Telemetry Stations
VERIFIED_NRW_GAUGES: List[Dict[str, Any]] = [
    {
        "id": "NRW_GAUGE_GLASLYN",
        "name": "River Glaslyn Flood Gauge (Beddgelert)",
        "coords": [-4.1025, 53.0110],
        "plainSummary": "Tracks river rise from Snowdon. When high, the A498 valley road floods quickly.",
        "metric": "Alert Stage: 1.65m | Severe Flood: 2.10m | Basin: Eryri South",
        "action": "Monitors Yr Wyddfa southern runoff. Above 1.65m, A498 Aberglaslyn Pass floods; divert via A4085.",
    },
    {
        "id": "NRW_GAUGE_OGWEN",
        "name": "River Ogwen Flood Gauge (Bethesda)",
        "coords": [-4.0615, 53.1802],
        "plainSummary": "Tracks fast mountain runoff that can flood the A5 valley road and bridge.",
        "metric": "Alert Stage: 1.80m | Severe Flood: 2.40m | Basin: Ogwen",
        "action": "Steep Glyderau/Carneddau catchment; rapid spate surge threatens A5 valley bridge.",
    },
    {
        "id": "NRW_GAUGE_CONWY",
        "name": "River Conwy Flood Gauge (Llanrwst)",
        "coords": [-3.7960, 53.1385],
        "plainSummary": "Tracks fast river flooding at Pont Fawr bridge. Use the higher A470 road in heavy rain.",
        "metric": "Alert Stage: 2.40m | Bridge Overtop: 3.10m | Basin: Conwy",
        "action": "Fast-responding Carneddau spate river. Pont Fawr closes early; use eastern A470 high corridor.",
    },
    {
        "id": "NRW_GAUGE_CLWYD",
        "name": "River Clwyd Flood Gauge (St Asaph)",
        "coords": [-3.4435, 53.2578],
        "plainSummary": "Tracks river surge at St Asaph bridge that can flood A525 and A55 slip roads.",
        "metric": "Alert Stage: 2.20m | Severe Flood: 2.95m | Basin: Clwyd",
        "action": "Flash-flood constriction at Spring Gardens bridge; impacts A525 and A55 slip approaches.",
    },
    {
        "id": "NRW_GAUGE_DEE",
        "name": "River Dee Flood Gauge (Bala / Llyn Tegid)",
        "coords": [-3.5885, 52.9092],
        "plainSummary": "Monitors lake outflow flooding the Dee valley toward Corwen and Llangollen.",
        "metric": "Alert Stage: 1.95m | Floodplain Spill: 2.50m | Basin: Upper Dee",
        "action": "Controls Dee valley inundation downstream toward Corwen and Llangollen.",
    },
    {
        "id": "NRW_GAUGE_MAWDDACH",
        "name": "River Mawddach Flood Gauge (Ganllwyd)",
        "coords": [-3.8892, 52.8015],
        "plainSummary": "Tracks mountain river surge that can spill onto the A470 valley road.",
        "metric": "Alert Stage: 1.90m | A470 Encroachment: 2.45m | Basin: Mawddach",
        "action": "Drains Rhinogydd & Coed y Brenin; A470 valley gorge floods during prolonged orographic rain.",
    },
    {
        "id": "NRW_GAUGE_DYFI",
        "name": "River Dyfi Flood Gauge (Machynlleth Bridge)",
        "coords": [-3.8548, 52.5964],
        "plainSummary": "High river water and high tide here can cut off the main A487 north-south bridge.",
        "metric": "Alert Stage: 2.80m | A487 Cut-Off: 3.40m | Basin: Dyfi",
        "action": "Single-corridor A487 north-south artery severs when tidal surge meets spate discharge.",
    },
    {
        "id": "NRW_GAUGE_YSTWYTH",
        "name": "River Ystwyth Flood Gauge (Llanilar)",
        "coords": [-4.0165, 52.3562],
        "plainSummary": "Tracks fast hill runoff that can flood low B4575 river bridges near Aberystwyth.",
        "metric": "Alert Stage: 2.05m | Severe Flood: 2.70m | Basin: Ystwyth",
        "action": "Plynlimon runoff corridor; B4575 valley crossings submerge rapidly.",
    },
    {
        "id": "NRW_GAUGE_WYE_RHAYADER",
        "name": "River Wye Flood Gauge (Rhayader)",
        "coords": [-3.5132, 52.3018],
        "plainSummary": "Tracks river surge from the Elan hills along the central A470 corridor.",
        "metric": "Alert Stage: 2.10m | Severe Flood: 2.80m | Basin: Upper Wye",
        "action": "Monitors Elan Valley & Cambrian runoff along the central A470 trunk corridor.",
    },
    {
        "id": "NRW_GAUGE_TEIFI",
        "name": "River Teifi Flood Gauge (Lampeter)",
        "coords": [-4.0792, 52.1118],
        "plainSummary": "Tracks valley flooding across low A482 and A475 river crossings.",
        "metric": "Alert Stage: 2.15m | Severe Flood: 2.85m | Basin: Teifi",
        "action": "Low-lying valley floor inundation risk across A482/A475 river crossings.",
    },
    {
        "id": "NRW_GAUGE_CLEDDAU",
        "name": "Western Cleddau Flood Gauge (Haverfordwest)",
        "coords": [-4.9685, 51.8022],
        "plainSummary": "High river water meeting spring tides can flood town quay roads and bridges.",
        "metric": "Alert Stage: 2.35m | Tidal Lock: 3.10m | Basin: Cleddau",
        "action": "Preseli runoff meets Milford Haven spring tide; town quay and A4076 bridges flood.",
    },
    {
        "id": "NRW_GAUGE_TYWI",
        "name": "River Tywi Flood Gauge (Carmarthen Quay)",
        "coords": [-4.3055, 51.8542],
        "plainSummary": "Tracks deep river and tidal flooding across A40 and A484 valley approaches.",
        "metric": "Alert Stage: 3.20m | Tidal Lock: 4.10m | Basin: Tywi",
        "action": "Wide floodplain inundation across A40/A484 valley approaches during spring tides.",
    },
    {
        "id": "NRW_GAUGE_TAWE",
        "name": "River Tawe Flood Gauge (Pontardawe)",
        "coords": [-3.8482, 51.7165],
        "plainSummary": "Tracks fast river flooding along the A4067 Swansea Valley floor.",
        "metric": "Alert Stage: 2.25m | Severe Flood: 3.00m | Basin: Swansea Valley",
        "action": "Drains western Bannau Brycheiniog; A4067 Swansea Valley floor prone to flash inundation.",
    },
    {
        "id": "NRW_GAUGE_NEATH",
        "name": "River Neath Flood Gauge (Resolven)",
        "coords": [-3.7025, 51.7082],
        "plainSummary": "Tracks waterfall valley runoff that can flood low A465 slip roads.",
        "metric": "Alert Stage: 2.45m | Severe Flood: 3.20m | Basin: Vale of Neath",
        "action": "Waterfall country spate discharge threatens A465 low-lying slip roads.",
    },
    {
        "id": "NRW_GAUGE_TAFF",
        "name": "River Taff Flood Gauge (Pontypridd)",
        "coords": [-3.3388, 51.6012],
        "plainSummary": "Tracks fast, dangerous flash-flood surges where the Rhondda and Taff rivers meet.",
        "metric": "Alert Stage: 2.60m | Severe Flood: 3.50m | Basin: Taff / Rhondda",
        "action": "Confluence bottleneck for Rhondda and Taff valleys; severe urban flash-flood velocity.",
    },
    {
        "id": "NRW_GAUGE_ELY",
        "name": "River Ely Flood Gauge (St Fagans / Cardiff)",
        "coords": [-3.2685, 51.4842],
        "plainSummary": "Tracks river flooding across low west Cardiff roads and A4232 underpasses.",
        "metric": "Alert Stage: 2.15m | Severe Flood: 2.85m | Basin: Ely",
        "action": "Submerges western Cardiff low-lying crossings and A4232 approach culverts.",
    },
    {
        "id": "NRW_GAUGE_USK_BRECON",
        "name": "River Usk Flood Gauge (Brecon Bridge)",
        "coords": [-3.3935, 51.9478],
        "plainSummary": "Tracks Brecon Beacons runoff that quickly floods low town bridges and riverside paths.",
        "metric": "Alert Stage: 2.30m | Severe Flood: 3.10m | Basin: Usk",
        "action": "Drains northern Bannau Brycheiniog; low town bridges and riverside roads flood rapidly.",
    },
    {
        "id": "NRW_GAUGE_USK_NEWPORT",
        "name": "River Usk Tidal Barrier (Newport)",
        "coords": [-2.9925, 51.5865],
        "plainSummary": "Tracks extreme Bristol Channel high tides meeting River Usk floodwater.",
        "metric": "Alert Stage: 6.80m AOD | Severe Surge: 7.50m AOD | Basin: Lower Usk",
        "action": "Extreme Bristol Channel tidal range coincides with Usk spate flow.",
    },
]

# Objective Physical Terrain, Tidal Stranding & Road Defile Hazards
VERIFIED_TERRAIN_AND_TIDAL_HAZARDS: List[Dict[str, Any]] = [
    {
        "id": "HAZ_TIDAL_TRAETH_LAFAN",
        "name": "Traeth Lafan Tidal Flats (Llanfairfechan)",
        "coords": [-4.0185, 53.2582],
        "plainBadge": "FAST TIDE HAZARD",
        "label": "COASTAL TIDAL STRANDING HAZARD",
        "plainSummary": "Incoming tide fills deep channels behind you faster than walking pace. Check tide times.",
        "metric": "Incoming Tide Velocity: 4+ knots | Routing Penalty: W_flood +10.0",
        "action": "Extensive mudflats flood rapidly from behind via deep channels. Never cross without tide tables.",
    },
    {
        "id": "HAZ_TERRAIN_A5_OGWEN",
        "name": "A5 Ogwen Valley Rockfall & Ice Gorge",
        "coords": [-4.0150, 53.1250],
        "plainBadge": "ROCKFALL & ICE ROAD",
        "label": "DFT STATS19 & TERRAIN DEFILE",
        "plainSummary": "Narrow mountain gorge prone to falling rocks, fierce crosswinds, and winter black ice.",
        "metric": "DfT STATS19 Cluster | Gale Funnel & Verglas | Penalty: W_incident +4.0",
        "action": "Sheer glaciated slabs above single-carriageway; severe crosswinds and winter verglas.",
    },
    {
        "id": "HAZ_TERRAIN_PENYPASS",
        "name": "A4086 Pen-y-Pass Steep Mountain Road",
        "coords": [-4.0205, 53.0801],
        "plainBadge": "DANGEROUS MOUNTAIN PASS",
        "label": "DFT STATS19 COLLISION DEFILE",
        "plainSummary": "Steep 1-in-8 mountain pass with blind crests, freezing fog, and frequent winter ice.",
        "metric": "1:8 Gradient | Blind Crests & Freezing Fog | Penalty: W_incident +4.0",
        "action": "Impassable in freezing fog/snow; deploy high-visibility markers if evacuating on foot.",
    },
    {
        "id": "HAZ_TERRAIN_CRIMEA",
        "name": "A470 Crimea Pass (Blaenau Ffestiniog)",
        "coords": [-3.9145, 53.0122],
        "plainBadge": "SNOW & WHITEOUT PASS",
        "label": "HIGH-ALTITUDE PASS EXPOSURE",
        "plainSummary": "High 385m pass that is often the first North Wales main road closed by snowdrifts.",
        "metric": "Elevation: 385m | Snowdrift & Whiteout Corridor | Penalty: W_incident +3.5",
        "action": "Frequently first North Wales trunk pass closed by snowdrifts and freezing cloud.",
    },
    {
        "id": "HAZ_TERRAIN_BERWYN",
        "name": "B4391 Milltir Cerrig Pass (Berwyn Hills)",
        "coords": [-3.5415, 52.8625],
        "plainBadge": "EXPOSED HIGH PASS",
        "label": "EXPOSED UPLAND DEFILE",
        "plainSummary": "Isolated 486m moorland road with zero phone signal, no shelter, and deep snowdrifts.",
        "metric": "Elevation: 486m | Zero Shelter & Deep Drifts | Penalty: W_incident +3.5",
        "action": "Remote high-moorland pass with zero cellular signal and rapid whiteout exposure.",
    },
    {
        "id": "HAZ_TIDAL_NEWGALE",
        "name": "A487 Newgale Beach Storm Surge Road",
        "coords": [-5.1265, 51.8522],
        "plainBadge": "WAVE OVERTOPPING ROAD",
        "label": "COASTAL OVERTOPPING DEFILE",
        "plainSummary": "Storm waves throw heavy stones across this coastal road at high tide. Divert inland.",
        "metric": "Spring Tide + SW Gale Overtopping | Penalty: W_flood +10.0",
        "action": "Storm waves hurl cobbles across A487 coastal causeway; divert inland via Roch/Camrose.",
    },
    {
        "id": "HAZ_TIDAL_CEFN_SIDAN",
        "name": "Cefn Sidan Sandbar & Quicksand Hazard",
        "coords": [-4.3685, 51.7245],
        "plainBadge": "QUICKSAND & FAST TIDE",
        "label": "COASTAL TIDAL STRANDING HAZARD",
        "plainSummary": "Shifting quicksand channels and fast incoming tides cut off outer sandbanks. Stay near dunes.",
        "metric": "8-Mile Sandbar | Shifting Quicksand Channels | Penalty: W_flood +10.0",
        "action": "Rapid tidal cut-off across outer sandbanks; keep strictly above high-water dune line.",
    },
    {
        "id": "HAZ_TIDAL_WORMS_HEAD",
        "name": "Worm's Head Tidal Causeway (Rhossili)",
        "coords": [-4.3165, 51.5648],
        "plainBadge": "TIDE CUT-OFF CAUSEWAY",
        "label": "COASTAL TIDAL CUT-OFF",
        "plainSummary": "Rocky crossing floods 2.5 hours before high tide and stays underwater for 5 hours.",
        "metric": "Submerged 5 Hours Per Tide Cycle | Penalty: W_flood +10.0",
        "action": "Rocky causeway floods 2.5h before high water; stranding risk with severe Atlantic swell.",
    },
    {
        "id": "HAZ_TERRAIN_BLACK_MOUNTAIN",
        "name": "A4069 Black Mountain Hairpin Pass",
        "coords": [-3.8165, 51.8325],
        "plainBadge": "STEEP HAIRPIN PASS",
        "label": "DFT STATS19 & EXPOSURE DEFILE",
        "plainSummary": "Exposed 493m mountain road with sharp hairpins, sheer drops, and severe ice/mist.",
        "metric": "Elevation: 493m | Sheer Drop-Offs & Ice | Penalty: W_incident +4.0",
        "action": "Exposed hairpin mountain road linking Brynaman and Llangadog; hazardous in mist/ice.",
    },
    {
        "id": "HAZ_TIDAL_SULLY",
        "name": "Sully Island Tidal Causeway",
        "coords": [-3.1955, 51.3965],
        "plainBadge": "FAST TIDE CAUSEWAY",
        "label": "COASTAL TIDAL STRANDING HAZARD",
        "plainSummary": "Massive 12m Bristol Channel tides flood this causeway very rapidly. Check tide times.",
        "metric": "Bristol Channel 12m Tidal Surge | Penalty: W_flood +10.0",
        "action": "Second-highest tidal range in the world; causeway floods violently on incoming tide.",
    },
]

# Statutory HSE Public COMAH Upper/Lower-Tier Industrial Sites in Wales
VERIFIED_COMAH_SITES: List[Dict[str, Any]] = [
    {
        "id": "COMAH_PEMBROKE",
        "name": "Milford Haven / Pembroke Refinery Safety Zone",
        "coords": [-5.0195, 51.6895],
        "radius_m": 1200,
        "plainSummary": "Major fuel & chemical storage site. Keep 1.2 km upwind if an industrial alarm sounds.",
        "metric": "Upper-Tier HSE COMAH | Exclusion Buffer: 1,200m | Penalty: W_comah +8.0",
        "action": "Major hydrocarbon storage & refining facility. Maintain 1.2km upwind clearance during industrial alarms.",
    },
    {
        "id": "COMAH_BARRY",
        "name": "Barry Docks Chemical Plant Safety Zone",
        "coords": [-3.2535, 51.3995],
        "radius_m": 900,
        "plainSummary": "Major chemical site. Stay indoors with windows shut or move cross-wind during alerts.",
        "metric": "Upper-Tier HSE COMAH | Exclusion Buffer: 900m | Penalty: W_comah +8.0",
        "action": "Silicone & specialty chemical installations. Shelter indoors or evacuate cross-wind if vapour release occurs.",
    },
    {
        "id": "COMAH_BAGLAN",
        "name": "Baglan Bay & Port Talbot Industrial Safety Zone",
        "coords": [-3.8165, 51.6085],
        "radius_m": 1000,
        "plainSummary": "Heavy industrial and gas storage corridor by the M4. Divert inland during chemical alerts.",
        "metric": "Upper-Tier HSE COMAH | Exclusion Buffer: 1,000m | Penalty: W_comah +8.0",
        "action": "Heavy industrial & energy storage corridor adjacent to M4. Divert inland during hazmat incidents.",
    },
    {
        "id": "COMAH_DEESIDE",
        "name": "Deeside & Shotton Power / Gas Safety Zone",
        "coords": [-3.0465, 53.2185],
        "radius_m": 850,
        "plainSummary": "Power station and industrial chemical zone on the Dee estuary. Maintain perimeter clearance.",
        "metric": "Upper-Tier HSE COMAH | Exclusion Buffer: 850m | Penalty: W_comah +8.0",
        "action": "Combined cycle power & chemical manufacturing zone on Dee estuary.",
    },
    {
        "id": "COMAH_WREXHAM",
        "name": "Wrexham Industrial Estate Hazmat Buffer",
        "coords": [-2.9195, 53.0355],
        "radius_m": 750,
        "plainSummary": "Industrial chemical storage zone. Avoid area if fire or vapour sirens sound.",
        "metric": "Lower-Tier HSE COMAH | Exclusion Buffer: 750m | Penalty: W_comah +8.0",
        "action": "Industrial chemical & manufacturing storage buffer. Maintain perimeter clearance.",
    },
]

# Major Welsh River Flood Plain & Hydrological Disruption Corridors
VERIFIED_FLOOD_CORRIDORS: List[Dict[str, Any]] = [
    {
        "id": "HAZ_FLOOD_LLYN_PERIS",
        "name": "Llyn Peris & A4086 Valley Flood Zone",
        "line": [[-4.1440, 53.1350], [-4.1280, 53.1230], [-4.1160, 53.1110]],
        "buffer_m": 320,
        "plainSummary": "Low valley flood and power-surge zone. Climb onto higher ground above Llanberis via Waunfawr.",
        "metric": "Civil Protection Corridor | Routing Penalty: W_flood +10.0",
        "action": "High-voltage surge & reservoir tailrace overflow corridor. Divert above 150m contour via Waunfawr.",
    },
    {
        "id": "HAZ_FLOOD_CONWY_VALLEY",
        "name": "Conwy Valley Floodplain (Llanrwst to Trefriw)",
        "line": [[-3.8150, 53.1750], [-3.8020, 53.1520], [-3.7950, 53.1340], [-3.7850, 53.1120]],
        "buffer_m": 450,
        "plainSummary": "Valley floor and B5106 flood fast during heavy rain. Use the higher A470 on the east side.",
        "metric": "NRW 1-in-30-Yr Flood Zone 3 | Routing Penalty: W_flood +10.0",
        "action": "B5106 & valley floor submerge rapidly during Carneddau storms. Use elevated eastern A470.",
    },
    {
        "id": "HAZ_FLOOD_ABERGLASLYN",
        "name": "Afon Glaslyn Gorge Flood Zone (Beddgelert)",
        "line": [[-4.1060, 53.0160], [-4.1010, 53.0050], [-4.1080, 52.9880]],
        "buffer_m": 280,
        "plainSummary": "Narrow river gorge on the A498 prone to flash flooding and falling rocks in storms.",
        "metric": "NRW 1-in-30-Yr Flood Zone 3 | Routing Penalty: W_flood +10.0",
        "action": "Narrow gorge bottleneck on A498. Prone to flash river overtopping and rockfall.",
    },
    {
        "id": "HAZ_FLOOD_DYFI_ESTUARY",
        "name": "Afon Dyfi Floodplain (Machynlleth)",
        "line": [[-3.8850, 52.5920], [-3.8540, 52.5970], [-3.8220, 52.6040]],
        "buffer_m": 500,
        "plainSummary": "A487 bridge approaches flood when high tides meet heavy mountain rain.",
        "metric": "NRW 1-in-30-Yr Tidal/Fluvial Zone | Routing Penalty: W_flood +10.0",
        "action": "A487 bridge approaches flood during combined high tide and mountain runoff.",
    },
    {
        "id": "HAZ_FLOOD_TAFF_RHONDDA",
        "name": "Pontypridd Taff & Rhondda Flood Zone",
        "line": [[-3.3680, 51.6150], [-3.3390, 51.6010], [-3.3210, 51.5820]],
        "buffer_m": 380,
        "plainSummary": "Steep valleys funnel fast, deep floodwater onto town roads and the A470 floor.",
        "metric": "NRW 1-in-30-Yr Spate Flood Zone | Routing Penalty: W_flood +10.0",
        "action": "Steep valley walls funnel high-velocity water onto A470/A4058 valley floor.",
    },
    {
        "id": "HAZ_FLOOD_TYWI_VALLEY",
        "name": "Lower Tywi Floodplain (Llandeilo to Carmarthen)",
        "line": [[-4.3100, 51.8520], [-4.1850, 51.8640], [-4.0120, 51.8810]],
        "buffer_m": 550,
        "plainSummary": "Wide river valley prone to deep flooding. Keep to the higher A40 ridge road.",
        "metric": "NRW 1-in-30-Yr Flood Zone 3 | Routing Penalty: W_flood +10.0",
        "action": "Extensive agricultural & road plain inundation. Maintain route along elevated A40 ridge.",
    },
]



# Mountain Crag, Knife-Edge Ridge & MREW Physical Rescue Hazard Clouds
# Derived from OSM natural=cliff / natural=arete vectors & MREW terrain incident reports
# (Strictly objective physical terrain hazards; self-harm/suicide data purged per Suicide Act 1961)
VERIFIED_MOUNTAIN_CRAG_CLOUDS: List[Dict[str, Any]] = [
    {
        "id": "CRAG_CRIB_GOCH_SNOWDON",
        "name": "Crib Goch & Garnedd Ugain Knife-Edge Ridge",
        "line": [[-4.0635, 53.0768], [-4.0725, 53.0752], [-4.0785, 53.0738], [-4.0763, 53.0685]],
        "buffer_m": 220,
        "plainSummary": "Exposed Grade 1 knife-edge ridge with 300m sheer drops on both sides. Lethal in high winds or ice.",
        "metric": "MREW High-Frequency Terrain Cluster | 923m–1,085m | Penalty: W_incident +6.0",
        "action": "Unroped scrambling with severe fall exposure. Avoid in gusts >30 mph, cloud, or winter verglas; use Llanberis or Ranger path.",
    },
    {
        "id": "CRAG_CLOGWYN_GARNEDD",
        "name": "Yr Wyddfa East Face / Clogwyn y Garnedd Sheer Cliffs",
        "line": [[-4.0755, 53.0682], [-4.0705, 53.0655], [-4.0665, 53.0625]],
        "buffer_m": 240,
        "plainSummary": "Vertical 250m cliff face directly below Snowdon summit & Pyg/Miners upper zig-zags.",
        "metric": "OSM natural=cliff | 250m Vertical Drop & Winter Cornice | Penalty: W_incident +6.0",
        "action": "Stay strictly on marked paths in mist or snow; winter snow cornices overhang the eastern cliff edge.",
    },
    {
        "id": "CRAG_TRYFAN_BRISTLY",
        "name": "Tryfan North Ridge & Bristly Ridge Crags (Ogwen)",
        "line": [[-3.9985, 53.1205], [-3.9978, 53.1155], [-3.9995, 53.1105], [-4.0065, 53.1065]],
        "buffer_m": 260,
        "plainSummary": "Steep rocky peak requiring hands-on climbing; high rate of navigation errors and steep crag falls.",
        "metric": "OVMRO Primary Terrain Cluster | 917m | Penalty: W_incident +6.0",
        "action": "Easy to stray onto vertical western gullies in low cloud. Descend only via verified Heather Terrace / Bwlch Tryfan.",
    },
    {
        "id": "CRAG_DEVILS_KITCHEN",
        "name": "Twll Du (Devil's Kitchen) & Y Garn Steep Chute",
        "line": [[-4.0345, 53.1125], [-4.0395, 53.1145], [-4.0455, 53.1165]],
        "buffer_m": 230,
        "plainSummary": "Sheer black cliff gully above Llyn Idwal with loose rock, icefalls, and steep drop-offs.",
        "metric": "OSM natural=cliff | 800m–947m | Loose Scree & Ice | Penalty: W_incident +5.5",
        "action": "Never attempt to climb or descend inside the Devil's Kitchen cleft; follow the stepped stone path to the west.",
    },
    {
        "id": "CRAG_CARNEDD_LLEWELYN_BLACK_LADDERS",
        "name": "Ysgolion Duon (Black Ladders) Cliff Escarpment",
        "line": [[-3.9765, 53.1645], [-3.9685, 53.1615], [-3.9625, 53.1575]],
        "buffer_m": 250,
        "plainSummary": "300m vertical mountain cliff wall in the Carneddau prone to rockfall and winter avalanche.",
        "metric": "OSM natural=cliff | 1,064m Plateau Edge | Penalty: W_incident +5.5",
        "action": "Keep well back from the northern plateau rim in whiteout or high winds.",
    },
    {
        "id": "CRAG_CADAIR_IDRIS_CRAIG_CAU",
        "name": "Cadair Idris — Craig Cau & Penygadair North Cliffs",
        "line": [[-3.9165, 52.7015], [-3.9085, 52.6995], [-3.9015, 52.6975]],
        "buffer_m": 260,
        "plainSummary": "Towering glacial amphitheatre cliffs above Llyn Cau with sudden coastal fog and steep drops.",
        "metric": "Aberdyfi SAR Terrain Cluster | 893m | Penalty: W_incident +5.5",
        "action": "Rapid sea-fog engulfment causes hikers to walk off northern crags; navigate by compass bearing along ridge.",
    },
    {
        "id": "CRAG_PEN_Y_FAN_NORTH",
        "name": "Pen y Fan & Corn Du North Escarpment Cliffs",
        "line": [[-3.4425, 51.8845], [-3.4365, 51.8835], [-3.4285, 51.8815]],
        "buffer_m": 240,
        "plainSummary": "Steep sandstone cliff drop-offs along the northern summit edge of the Brecon Beacons.",
        "metric": "SWMRA Terrain Cluster | 886m | Cornice & Drop Exposure | Penalty: W_incident +5.0",
        "action": "Follow the wide southern ridge path; northern face drops sharply into Cwm Llwch.",
    },
]

# BGS Clean Water Groundwater Aquifers (Resupply Polygons)
VERIFIED_AQUIFER_ZONES: List[Dict[str, Any]] = [
    {
        "id": "AQUIFER_VALE_CLWYD",
        "name": "Vale of Clwyd Sandstone Groundwater Aquifer",
        "poly": [
            [-3.48, 53.31], [-3.36, 53.30], [-3.28, 53.21], [-3.25, 53.11],
            [-3.31, 53.08], [-3.39, 53.12], [-3.45, 53.22], [-3.48, 53.31],
        ],
        "plainSummary": "Major underground sandstone water store feeding reliable springs and farm boreholes.",
        "metric": "BGS Principal Permo-Triassic Sandstone Aquifer | High Yield",
        "action": "High-capacity groundwater basin; prioritize spring lines along eastern limestone fault margin.",
    },
    {
        "id": "AQUIFER_BRECON_ORS",
        "name": "Bannau Brycheiniog Old Red Sandstone Aquifer",
        "poly": [
            [-3.68, 51.94], [-3.46, 51.96], [-3.22, 51.93], [-3.18, 51.84],
            [-3.34, 51.81], [-3.56, 51.82], [-3.68, 51.88], [-3.68, 51.94],
        ],
        "plainSummary": "Upland sandstone rock layer feeding dozens of clean, cold mountain springs year-round.",
        "metric": "BGS Secondary A / Principal Devonian Sandstone Aquifer",
        "action": "Abundant upland spring discharge along northern escarpment; filter or boil 60s before storage.",
    },
    {
        "id": "AQUIFER_SOUTH_PEMBROKE",
        "name": "South Pembrokeshire Carboniferous Limestone Aquifer",
        "poly": [
            [-5.02, 51.70], [-4.82, 51.71], [-4.68, 51.68], [-4.69, 51.62],
            [-4.88, 51.61], [-5.01, 51.64], [-5.02, 51.70],
        ],
        "plainSummary": "Coastal limestone groundwater basin feeding natural springs and wells across South Pembroke.",
        "metric": "BGS Principal Carboniferous Limestone Karst Aquifer",
        "action": "Fast-draining karst aquifer; always filter or boil spring water after heavy agricultural runoff.",
    },
]

# BGS GeoSure Landslide & Steep Valley Slope Instability Corridors (Hazards)
VERIFIED_LANDSLIDE_ZONES: List[Dict[str, Any]] = [
    {
        "id": "SLIP_RHONDDA_VALLEYS",
        "name": "Rhondda & Taff Valley Steep Slope / Landslide Zone",
        "line": [[-3.52, 51.68], [-3.44, 51.65], [-3.37, 51.62], [-3.32, 51.59]],
        "buffer_m": 420,
        "plainSummary": "Steep valley sides prone to mudslides, fallen trees, and road blockages during heavy rain.",
        "metric": "BGS GeoSure High Landslide Susceptibility | Penalty: W_incident +3.0",
        "action": "Glacial till & coal-measure slopes saturate rapidly; watch for debris flows blocking A4058/A4061.",
    },
    {
        "id": "SLIP_NANT_FFRANCON",
        "name": "Nant Ffrancon & Llyn Ogwen Scree / Rockslide Corridor",
        "line": [[-4.04, 53.15], [-4.02, 53.13], [-3.99, 53.11]],
        "buffer_m": 340,
        "plainSummary": "Steep mountain cliffs above the A5 prone to rockfalls and scree slides in freeze-thaw weather.",
        "metric": "BGS GeoSure Steep Bedrock / Scree Instability | Penalty: W_incident +3.0",
        "action": "Avoid parking or sheltering directly beneath steep eastern crags during heavy rain or thaw.",
    },
    {
        "id": "SLIP_EBBW_SIRHOWY",
        "name": "Ebbw & Sirhowy Valley Steep Escarpment Slip Zone",
        "line": [[-3.22, 51.76], [-3.19, 51.71], [-3.16, 51.66]],
        "buffer_m": 380,
        "plainSummary": "Steep South Wales valley slopes prone to earthslips and blocked roads after prolonged storms.",
        "metric": "BGS GeoSure High Mass-Movement Zone | Penalty: W_incident +3.0",
        "action": "Monitor valley road embankments for slumping and culvert washouts during named storms.",
    },
]

# Ministry of Defence (MoD) Live-Fire & Restricted Training Ranges (National Security Act 2023 Compliant)
VERIFIED_MOD_RESTRICTED_ZONES: List[Dict[str, Any]] = [
    {
        "id": "MOD_SENNYBRIDGE_SENTA",
        "name": "Sennybridge Military Training Area (Restricted Access)",
        "poly": [
            [-3.72, 52.06], [-3.54, 52.07], [-3.48, 52.01], [-3.55, 51.96],
            [-3.69, 51.97], [-3.74, 52.02], [-3.72, 52.06],
        ],
        "plainSummary": "Active military live-firing range. Public entry is strictly prohibited when red flags/lamps are active.",
        "metric": "MoD Statutory Danger Area (SENTA) | OS OpenData Boundary",
        "action": "Do not enter perimeter or touch unexploded ordnance; keep strictly to public A40/B4519 trunk roads.",
    },
    {
        "id": "MOD_CASTLEMARTIN",
        "name": "Castlemartin Coastal Firing Range (Restricted Access)",
        "poly": [
            [-5.06, 51.64], [-4.94, 51.64], [-4.93, 51.60], [-5.05, 51.60], [-5.06, 51.64],
        ],
        "plainSummary": "Coastal military live-firing range in South Pembrokeshire. Coastal path closes during firing.",
        "metric": "MoD Statutory Danger Area (Castlemartin) | OS OpenData Boundary",
        "action": "Check range firing notices; divert inland via Warren/St Twynnells when coastal perimeter is closed.",
    },
    {
        "id": "MOD_PENDINE_ABERPORTH",
        "name": "Pendine Sands & Test Range (Restricted Access)",
        "poly": [
            [-4.56, 51.76], [-4.46, 51.76], [-4.46, 51.73], [-4.56, 51.73], [-4.56, 51.76],
        ],
        "plainSummary": "Restricted coastal test range and tidal beach danger zone. Obey perimeter warning signs.",
        "metric": "MoD Statutory Danger Area (Pendine) | OS OpenData Boundary",
        "action": "Public access prohibited inside range markers; high-speed tidal surge hazard on adjacent flats.",
    },
]

# Verified Welsh Mountain Rescue Bases, Major A&E Hospitals & Strategic Telecom Hills
VERIFIED_EMERGENCY_AND_COMMS_POINTS: List[Dict[str, Any]] = [
    {
        "id": "BASE_MRT_LLANBERIS",
        "name": "Llanberis Mountain Rescue Team Base",
        "coords": [-4.1295, 53.1188],
        "category": "baseline",
        "plainBadge": "MOUNTAIN RESCUE BASE",
        "categoryLabel": "MREW MOUNTAIN RESCUE HQ",
        "accentColor": "#94A3B8",
        "plainSummary": "Busiest mountain rescue base in the UK, serving Yr Wyddfa (Snowdon) and surrounding peaks.",
        "metric": "MREW / North Wales Mountain Rescue Association",
        "action": "Call 999 and ask for Police -> Mountain Rescue in emergencies.",
    },
    {
        "id": "BASE_MRT_OGWEN",
        "name": "Ogwen Valley Mountain Rescue Base (Bryn Poeth)",
        "coords": [-3.9985, 53.1235],
        "category": "baseline",
        "plainBadge": "MOUNTAIN RESCUE BASE",
        "categoryLabel": "MREW MOUNTAIN RESCUE HQ",
        "accentColor": "#94A3B8",
        "plainSummary": "Mountain rescue station serving Tryfan, the Glyderau, and Carneddau ridges.",
        "metric": "OVMRO | A5 Capel Curig / Ogwen Corridor",
        "action": "Call 999 and ask for Police -> Mountain Rescue in emergencies.",
    },
    {
        "id": "BASE_MRT_ABERDYFI",
        "name": "Aberdyfi Search & Rescue Team Base",
        "coords": [-4.0455, 52.5435],
        "category": "baseline",
        "plainBadge": "MOUNTAIN RESCUE BASE",
        "categoryLabel": "MREW SEARCH & RESCUE HQ",
        "accentColor": "#94A3B8",
        "plainSummary": "Mid-Wales mountain and estuary search & rescue base covering Cadair Idris and the Dyfi.",
        "metric": "MREW | Southern Eryri & Dyfi Estuary",
        "action": "Call 999 and ask for Police / Coastguard -> Search & Rescue.",
    },
    {
        "id": "BASE_MRT_BRECON",
        "name": "Brecon Mountain Rescue Team Base",
        "coords": [-3.3855, 51.9425],
        "category": "baseline",
        "plainBadge": "MOUNTAIN RESCUE BASE",
        "categoryLabel": "MREW MOUNTAIN RESCUE HQ",
        "accentColor": "#94A3B8",
        "plainSummary": "Mountain rescue base serving Pen y Fan and the central Brecon Beacons.",
        "metric": "SWMRA | Bannau Brycheiniog Central Sector",
        "action": "Call 999 and ask for Police -> Mountain Rescue in emergencies.",
    },
    {
        "id": "BASE_RNLI_HOLYHEAD",
        "name": "Holyhead RNLI Lifeboat Station",
        "coords": [-4.6285, 53.3145],
        "category": "baseline",
        "plainBadge": "RNLI LIFEBOAT STATION",
        "categoryLabel": "RNLI ALL-WEATHER LIFEBOAT",
        "accentColor": "#94A3B8",
        "plainSummary": "24/7 all-weather coastal lifeboat station with backup generator power and marine VHF.",
        "metric": "RNLI Severn-Class All-Weather & D-Class Inshore",
        "action": "Call 999 and ask for Coastguard for coastal or tidal emergencies.",
    },
    {
        "id": "BASE_RNLI_MUMBLES",
        "name": "The Mumbles RNLI Lifeboat Station (Swansea)",
        "coords": [-3.9745, 51.5695],
        "category": "baseline",
        "plainBadge": "RNLI LIFEBOAT STATION",
        "categoryLabel": "RNLI ALL-WEATHER LIFEBOAT",
        "accentColor": "#94A3B8",
        "plainSummary": "24/7 coastal lifeboat station guarding Swansea Bay and Bristol Channel approaches.",
        "metric": "RNLI Tamar-Class All-Weather & D-Class Inshore",
        "action": "Call 999 and ask for Coastguard for coastal or tidal emergencies.",
    },
    {
        "id": "MED_HOSP_BANGOR",
        "name": "Ysbyty Gwynedd Major A&E Hospital (Bangor)",
        "coords": [-4.1605, 53.2095],
        "category": "medical",
        "plainBadge": "MAJOR A&E HOSPITAL",
        "categoryLabel": "NHS WALES ACUTE HOSPITAL",
        "accentColor": "#10B981",
        "plainSummary": "24/7 major Accident & Emergency hospital with full trauma care and backup power.",
        "metric": "Betsi Cadwaladr UHB | 24/7 Emergency Department",
        "action": "Primary North-West Wales emergency hospital.",
    },
    {
        "id": "MED_HOSP_WREXHAM",
        "name": "Wrexham Maelor Major A&E Hospital",
        "coords": [-3.0125, 53.0465],
        "category": "medical",
        "plainBadge": "MAJOR A&E HOSPITAL",
        "categoryLabel": "NHS WALES ACUTE HOSPITAL",
        "accentColor": "#10B981",
        "plainSummary": "24/7 major Accident & Emergency hospital serving North-East Wales.",
        "metric": "Betsi Cadwaladr UHB | 24/7 Emergency Department",
        "action": "Primary North-East Wales emergency hospital.",
    },
    {
        "id": "MED_HOSP_ABERYSTWYTH",
        "name": "Bronglais General A&E Hospital (Aberystwyth)",
        "coords": [-4.0725, 52.4155],
        "category": "medical",
        "plainBadge": "MAJOR A&E HOSPITAL",
        "categoryLabel": "NHS WALES ACUTE HOSPITAL",
        "accentColor": "#10B981",
        "plainSummary": "Main Mid-Wales 24/7 Accident & Emergency hospital serving Ceredigion and Powys.",
        "metric": "Hywel Dda UHB | 24/7 Emergency Department",
        "action": "Primary Mid-Wales coastal emergency hospital.",
    },
    {
        "id": "MED_HOSP_CARMARTHEN",
        "name": "Glangwili General A&E Hospital (Carmarthen)",
        "coords": [-4.2865, 51.8665],
        "category": "medical",
        "plainBadge": "MAJOR A&E HOSPITAL",
        "categoryLabel": "NHS WALES ACUTE HOSPITAL",
        "accentColor": "#10B981",
        "plainSummary": "24/7 Accident & Emergency hospital serving Carmarthenshire and West Wales.",
        "metric": "Hywel Dda UHB | 24/7 Emergency Department",
        "action": "Primary South-West Wales emergency hospital.",
    },
    {
        "id": "MED_HOSP_SWANSEA",
        "name": "Morriston Major Trauma & A&E Hospital (Swansea)",
        "coords": [-3.9355, 51.6845],
        "category": "medical",
        "plainBadge": "MAJOR A&E HOSPITAL",
        "categoryLabel": "NHS WALES ACUTE HOSPITAL",
        "accentColor": "#10B981",
        "plainSummary": "Major South-West Wales 24/7 trauma, burns, and Accident & Emergency hospital.",
        "metric": "Swansea Bay UHB | Regional Trauma & Burns Centre",
        "action": "Primary Swansea Bay emergency hospital.",
    },
    {
        "id": "MED_HOSP_CARDIFF",
        "name": "University Hospital of Wales Major Trauma Centre",
        "coords": [-3.1885, 51.5075],
        "category": "medical",
        "plainBadge": "MAJOR TRAUMA HOSPITAL",
        "categoryLabel": "NHS WALES TERTIARY TRAUMA CENTRE",
        "accentColor": "#10B981",
        "plainSummary": "Wales's primary major trauma hospital with 24/7 emergency care and helipad.",
        "metric": "Cardiff & Vale UHB | Major Trauma Centre",
        "action": "Tertiary emergency trauma hub for South-East Wales.",
    },
    {
        "id": "GRID_MAST_WENVOE",
        "name": "Wenvoe Strategic Telecom & Broadcast Tower",
        "coords": [-3.2818, 51.4595],
        "category": "grid_down",
        "plainBadge": "MAJOR TELECOM TOWER",
        "categoryLabel": "STRATEGIC BROADCAST & TELECOM HUB",
        "accentColor": "#FFB800",
        "plainSummary": "Main South Wales TV, radio, and mobile telecom hub with backup diesel generators.",
        "metric": "300m Guyed Mast | 500 kW ERP Broadcast & Cellular",
        "action": "Key regional broadcast & emergency radio relay site.",
    },
    {
        "id": "GRID_MAST_MOELYPARC",
        "name": "Moel-y-Parc Telecom & Radio Tower",
        "coords": [-3.3145, 53.2255],
        "category": "grid_down",
        "plainBadge": "MAJOR TELECOM TOWER",
        "categoryLabel": "STRATEGIC BROADCAST & TELECOM HUB",
        "accentColor": "#FFB800",
        "plainSummary": "Main North-East Wales hilltop TV, radio, and mobile telecom relay tower.",
        "metric": "235m Mast | Clwydian Range Line-of-Sight",
        "action": "High-ground line-of-sight landmark for emergency VHF radio.",
    },
    {
        "id": "GRID_MAST_BLAENPLWYF",
        "name": "Blaenplwyf Coastal Telecom & Radio Tower",
        "coords": [-4.1145, 52.3715],
        "category": "grid_down",
        "plainBadge": "MAJOR TELECOM TOWER",
        "categoryLabel": "STRATEGIC BROADCAST & TELECOM HUB",
        "accentColor": "#FFB800",
        "plainSummary": "Main Mid-Wales coastal broadcast and telecom relay mast near Aberystwyth.",
        "metric": "235m Mast | Cardigan Bay Coverage",
        "action": "Key Mid-Wales broadcast and cellular backhaul site.",
    },
]

# Wales Air Ambulance & Major Hospital Helipads (Baseline Emergency Aviation)
VERIFIED_HELIPADS: List[Dict[str, Any]] = [
    {
        "id": "HELI_CAERNARFON_WAA",
        "name": "Caernarfon Airport / Wales Air Ambulance North Base",
        "coords": [-4.3375, 53.1025],
        "plainSummary": "North Wales Air Ambulance helicopter base and coastal airfield.",
        "metric": "ICAO: EGCK | Wales Air Ambulance Charity HEMS Base",
        "action": "Active emergency helicopter operations; maintain strict perimeter clearance.",
    },
    {
        "id": "HELI_YSBYTY_GWYNEDD",
        "name": "Ysbyty Gwynedd Hospital Helipad (Bangor)",
        "coords": [-4.1592, 53.2088],
        "plainSummary": "Main North-West Wales hospital emergency helicopter landing pad.",
        "metric": "Hospital Trauma Helipad | Bangor Major A&E",
        "action": "Emergency air-ambulance landing zone; keep approach path clear.",
    },
    {
        "id": "HELI_WELSHPOOL_WAA",
        "name": "Welshpool Airport / Wales Air Ambulance Mid Base",
        "coords": [-3.1535, 52.6285],
        "plainSummary": "Mid-Wales Air Ambulance helicopter base and regional airfield.",
        "metric": "ICAO: EGCW | Wales Air Ambulance Mid-Wales Hub",
        "action": "Active emergency helicopter operations; keep clear of landing strip.",
    },
    {
        "id": "HELI_DREFACH_WAA",
        "name": "Drefach / Llanelli Wales Air Ambulance South-West Base",
        "coords": [-4.1125, 51.8165],
        "plainSummary": "South-West Wales Air Ambulance critical care helicopter base.",
        "metric": "Wales Air Ambulance Charity HEMS Hub",
        "action": "Active 24/7 emergency medical helicopter base.",
    },
    {
        "id": "HELI_CARDIFF_UHW",
        "name": "University Hospital of Wales Helipad (Cardiff)",
        "coords": [-3.1895, 51.5068],
        "plainSummary": "Major South Wales trauma hospital helicopter landing pad.",
        "metric": "Major Trauma Centre Helipad | Cardiff UHW",
        "action": "Primary South Wales tertiary trauma air-ambulance pad.",
    },
]

MINOR_PATH_TYPES = {"footway", "path", "track", "bridleway", "cycleway", "steps"}


def round_coordinates(coords: Any, precision: int = 5) -> Any:
    """Recursively round GeoJSON coordinates to keep mobile WebGL payloads compact."""
    if isinstance(coords, (list, tuple)):
        if len(coords) > 0 and isinstance(coords[0], (int, float)):
            return [round(float(c), precision) for c in coords[:2]]
        return [round_coordinates(c, precision) for c in coords]
    return coords


def make_three_ring_cloud_polygons(
    geom_wgs84: Any,
    core_buffer_m: float,
    mid_extra_m: float,
    halo_extra_m: float,
    base_props: Dict[str, Any],
    morph_smooth_m: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Generate 3 concentric, rounded geographic shells in British National Grid (EPSG:27700):
      1. `core` (exact physical boundary)
      2. `mid`  (soft intermediate plume)
      3. `halo` (outer vaporous fade)
    """
    gdf = gpd.GeoDataFrame([{"geometry": geom_wgs84}], crs="EPSG:4326").to_crs(epsg=27700)
    raw_bng = gdf.geometry.iloc[0]

    if morph_smooth_m > 0:
        raw_bng = raw_bng.buffer(morph_smooth_m, cap_style="round", join_style="round").buffer(
            -morph_smooth_m * 0.85, cap_style="round", join_style="round"
        )

    core_geom = (
        raw_bng.buffer(core_buffer_m, cap_style="round", join_style="round")
        if core_buffer_m != 0
        else raw_bng
    ).simplify(tolerance=14.0, preserve_topology=True)

    mid_geom = raw_bng.buffer(
        core_buffer_m + mid_extra_m, cap_style="round", join_style="round"
    ).simplify(tolerance=18.0, preserve_topology=True)

    halo_geom = raw_bng.buffer(
        core_buffer_m + halo_extra_m, cap_style="round", join_style="round"
    ).simplify(tolerance=22.0, preserve_topology=True)

    wales_clip = box(*WALES_BBOX)
    rings = [
        ("halo", f"{base_props['id']}_HALO", halo_geom),
        ("mid", f"{base_props['id']}_MID", mid_geom),
        ("core", base_props["id"], core_geom),
    ]

    out: List[Dict[str, Any]] = []
    for ring_name, fid, bng_geom in rings:
        if bng_geom.is_empty:
            continue
        wgs_geom = (
            gpd.GeoSeries([bng_geom], crs="EPSG:27700")
            .to_crs(epsg=4326)
            .iloc[0]
            .intersection(wales_clip)
        )
        if wgs_geom.is_empty:
            continue
        g_dict = mapping(wgs_geom)
        g_dict["coordinates"] = round_coordinates(g_dict["coordinates"], 4)
        props = dict(base_props)
        props["id"] = fid
        props["cloudRing"] = ring_name
        out.append(
            {
                "type": "Feature",
                "geometry": g_dict,
                "properties": props,
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
    """Normalize a raw OSM Point/Way-Center element with both plain-English and technical tiers."""
    if tags.get("natural") == "spring":
        category = "resupply"
        accent = "#00E5FF"
        plain_badge = "NATURAL WATER SPRING"
        label = "OSM HYDROLOGY / NATURAL SPRING"
        default_name = "Natural Mountain Spring"
        plain_summary = "Natural groundwater spring. Filter or boil for 1 minute before drinking."
        metric = "Gravity Spring Discharge | OSM natural=spring"
        action = "Filter with 0.1-micron pump or boil at a rolling boil for 60s."
        weight = 0.0
    elif (
        tags.get("amenity") == "drinking_water"
        or tags.get("man_made") in ("water_well", "water_tower")
    ):
        is_tower = tags.get("man_made") == "water_tower"
        category = "resupply"
        accent = "#00E5FF"
        plain_badge = "WATER RESERVOIR TOWER" if is_tower else "DRINKING WATER TAP / WELL"
        label = "GRAVITY WATER STORAGE" if is_tower else "POTABLE WATER POINT / BOREHOLE"
        default_name = "Elevated Water Storage Tower" if is_tower else "Public Drinking Water Tap / Well"
        plain_summary = (
            "Elevated gravity water tank. Holds pressurized water for a short time after power fails."
            if is_tower
            else "Public water tap or well. Fill containers early during grid warnings."
        )
        metric = "Gravity Head Storage | OSM water_tower" if is_tower else "Potable Tap / Shallow Well | OSM drinking_water"
        action = "Fill containers early during grid warnings; filter or boil if mains pressure drops."
        weight = 0.0
    elif tags.get("amenity") in ("hospital", "clinic"):
        is_hosp = tags.get("amenity") == "hospital"
        category = "medical"
        accent = "#10B981"
        plain_badge = "HOSPITAL / A&E" if is_hosp else "MEDICAL CLINIC"
        label = "NHS WALES HOSPITAL" if is_hosp else "NHS / COMMUNITY MEDICAL CLINIC"
        default_name = "NHS Hospital" if is_hosp else "Medical & Injury Clinic"
        plain_summary = (
            "Major hospital with emergency medical care and backup generator power."
            if is_hosp
            else "Local medical clinic for urgent injuries, wounds, and first-aid stabilization."
        )
        metric = "NHS Wales Acute Facility" if is_hosp else "NHS / Primary Care Trauma & Triage"
        action = "Verify A&E vs. Minor Injury Unit opening hours before traveling."
        weight = 0.0
    elif tags.get("man_made") == "mast":
        category = "grid_down"
        accent = "#FFB800"
        plain_badge = "TELECOM & RADIO MAST"
        label = "COMMUNICATIONS TOWER / MAST"
        default_name = "Mobile & Radio Telecom Mast"
        plain_summary = "Mobile phone and radio relay tower. High ground nearby gives the strongest signal."
        metric = "Telecom / Broadcast Mast | High Ground Line-of-Sight"
        action = "Most mobile masts have 4–12 hr battery backup before failing in prolonged blackouts."
        weight = 0.0
    elif tags.get("aeroway") == "helipad":
        category = "baseline"
        accent = "#10B981"
        plain_badge = "EMERGENCY HELIPAD"
        label = "CAA / HOSPITAL / SAR HELIPAD"
        default_name = "Designated Helicopter Landing Pad"
        plain_summary = "Clear helicopter landing pad used by Air Ambulance and Search & Rescue. Keep well clear."
        metric = "Designated Helipad | CAA / OSM aeroway=helipad"
        action = "Keep 50m perimeter clear of loose gear, drones, and vehicles during rotor operations."
        weight = 0.0
    elif (
        tags.get("amenity") == "fire_station"
        or tags.get("emergency") in ("lifeboat_station", "mountain_rescue")
    ):
        if tags.get("emergency") == "lifeboat_station":
            plain_badge = "RNLI LIFEBOAT STATION"
            default_name = "RNLI Coastal Lifeboat Station"
            plain_summary = "Coastal sea rescue lifeboat station with backup generator power and VHF marine radio."
            metric = "RNLI Maritime SAR Station"
        elif tags.get("emergency") == "mountain_rescue":
            plain_badge = "MOUNTAIN RESCUE BASE"
            default_name = "Mountain Rescue Team Base"
            plain_summary = "Volunteer mountain search & rescue base with VHF radio and cold-weather trauma gear."
            metric = "MREW / North & South Wales Mountain Rescue"
        else:
            plain_badge = "FIRE & RESCUE STATION"
            default_name = "Fire & Rescue Station"
            plain_summary = "Local Fire & Rescue station — statutory community emergency hub with backup power."
            metric = "Welsh Fire & Rescue Service Station"
        category = "baseline"
        accent = "#94A3B8"
        label = "STATUTORY EMERGENCY HUB"
        action = "Statutory emergency service base; note stations may be uncrewed while teams are deployed."
        weight = 0.0
    elif tags.get("emergency") == "defibrillator":
        category = "medical"
        accent = "#10B981"
        plain_badge = "DEFIBRILLATOR (AED)"
        label = "PUBLIC ACCESS DEFIBRILLATOR"
        default_name = "Public Heart Defibrillator (AED)"
        plain_summary = "Emergency heart shock kit. Call 999 for the cabinet keypad code."
        metric = "Public Access AED | OSM emergency=defibrillator"
        action = "Call 999 for cabinet keypad access; follow audio voice prompts on unit."
        weight = 0.0
    elif tags.get("man_made") == "surveillance":
        category = "baseline"
        accent = "#94A3B8"
        plain_badge = "PUBLIC CCTV CAMERA"
        label = "PUBLIC SURVEILLANCE (CCTV)"
        default_name = "Public / Traffic Camera"
        plain_summary = "Public highway or town security camera."
        metric = f"Surveillance Type: {tags.get('surveillance:type', 'Traffic / Public')}"
        action = "Public right-of-way camera installation."
        weight = 0.0
    elif tags.get("ford") == "yes":
        hw = str(tags.get("highway", "")).lower()
        if hw in MINOR_PATH_TYPES and not tags.get("name"):
            return None
        category = "hazards"
        accent = "#FF3B30"
        plain_badge = "FLOODING RIVER CROSSING"
        label = "VEHICULAR RIVER FORD"
        default_name = "Low-Water River Ford"
        plain_summary = "Road crosses directly through a river. Dangerous and impassable after heavy rain."
        metric = "Unmonitored Ford Crossing | Routing Penalty: W_flood +10.0"
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
            "plainBadge": plain_badge,
            "plainSummary": plain_summary,
            "categoryLabel": label,
            "accentColor": accent,
            "threatWeight": weight,
            "primaryMetric": metric,
            "operationalAction": action,
        },
    }


def query_overpass_fast(
    query: str, label: str, mirror_offset: int = 0
) -> Optional[Dict[str, Any]]:
    """Execute a fast Overpass query across 2 distinct mirrors with a tight 20s timeout."""
    headers = {"User-Agent": "DatumZero-ETL/4.3 (Civil Resilience Pipeline)"}
    n = len(OVERPASS_MIRRORS)
    ordered_mirrors = [OVERPASS_MIRRORS[(mirror_offset + i) % n] for i in range(min(3, n))]

    for mirror_url in ordered_mirrors:
        logging.info("Querying Overpass (%s) via %s...", label, mirror_url)
        try:
            response = requests.post(
                mirror_url,
                data={"data": query},
                headers=headers,
                timeout=25,
            )
            if response.status_code == 429:
                logging.warning("Mirror %s rate-limited (429) on %s. Rotating...", mirror_url, label)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            logging.warning("Mirror %s failed for %s (%s). Failing over...", mirror_url, label, exc)
    return None


def load_existing_geojson_features() -> List[Dict[str, Any]]:
    """Load previously compiled features from dist/wales_resilience_master.geojson if present."""
    if not OUTPUT_FILE.exists():
        return []
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return existing.get("features", [])
    except (OSError, json.JSONDecodeError):
        return []


def load_cached_features_by_prefix(prefixes: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Preserve previously compiled OSM features if Overpass mirrors temporarily time out."""
    preserved = [
        feat
        for feat in load_existing_geojson_features()
        if str(feat.get("properties", {}).get("id", "")).startswith(prefixes)
    ]
    if preserved:
        logging.info("Preserved %d cached features matching %s.", len(preserved), prefixes)
    return preserved


def restore_cached_points_in_bbox(bbox: List[float]) -> List[Dict[str, Any]]:
    """Restore previously compiled OSM Point features inside a specific sub-tile bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    restored: List[Dict[str, Any]] = []
    for feat in load_existing_geojson_features():
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        fid = str(feat.get("properties", {}).get("id", ""))
        if not fid.upper().startswith(("RESUPPLY_", "MEDICAL_", "BASELINE_", "HAZARDS_", "GRID_DOWN_")):
            continue
        coords = geom.get("coordinates", [])
        if len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                restored.append(feat)
    if restored:
        logging.info("Restored %d cached OSM points inside bbox %s.", len(restored), bbox)
    return restored


def generate_emf_corridor_polygons(
    power_lines: List[Tuple[LineString, int, str]]
) -> List[Dict[str, Any]]:
    """Buffer high-voltage lines in British National Grid (EPSG:27700) with 3-ring feathered edges."""
    if not power_lines:
        return []

    logging.info("Computing 3-ring feathered 50Hz EMF corridors for %d segments...", len(power_lines))
    records = [
        {"geometry": geom, "buffer_m": buf, "tier": tier}
        for geom, buf, tier in power_lines
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=27700)

    gdf["core_geom"] = gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_m"], cap_style="round"),
        axis=1,
    )
    gdf["mid_geom"] = gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_m"] * 1.30, cap_style="round"),
        axis=1,
    )
    gdf["halo_geom"] = gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_m"] * 1.65, cap_style="round"),
        axis=1,
    )

    wales_clip = box(*WALES_BBOX)
    features: List[Dict[str, Any]] = []

    for tier, group in gdf.groupby("tier"):
        buf_m = int(group["buffer_m"].iloc[0])
        for ring_name, col_name, id_suffix, simp_tol in [
            ("halo", "halo_geom", "_HALO", 18.0),
            ("mid", "mid_geom", "_MID", 15.0),
            ("core", "core_geom", "", 12.0),
        ]:
            series = gpd.GeoSeries(group[col_name], crs="EPSG:27700")
            unioned = (
                series.union_all()
                if hasattr(series, "union_all")
                else series.unary_union
            ).simplify(simp_tol, preserve_topology=True)

            wgs_geom = (
                gpd.GeoSeries([unioned], crs="EPSG:27700")
                .to_crs(epsg=4326)
                .iloc[0]
                .intersection(wales_clip)
            )
            if wgs_geom.is_empty:
                continue

            g_dict = mapping(wgs_geom)
            g_dict["coordinates"] = round_coordinates(g_dict["coordinates"], 5)
            features.append(
                {
                    "type": "Feature",
                    "geometry": g_dict,
                    "properties": {
                        "id": f"EMF_CORRIDOR_{buf_m}M{id_suffix}",
                        "name": f"{tier} Overhead Power Line Zone",
                        "category": "grid_down",
                        "plainBadge": "HIGH-VOLTAGE POWER LINE",
                        "plainSummary": f"High-voltage pylons ({buf_m}m safety zone). Can throw off compasses and drop wires in gales.",
                        "categoryLabel": "HIGH-VOLTAGE EMF BUFFER",
                        "accentColor": "#FFB800",
                        "threatWeight": 2.0,
                        "cloudRing": ring_name,
                        "primaryMetric": f"Biot-Savart 50Hz Exclusion: {buf_m}m | Magnetometer & RF Interference",
                        "operationalAction": "Expect compass deviation and storm line-drop hazard. Maintain clearance.",
                    },
                }
            )
    return features


def generate_bridge_choke_polygons(
    bridge_lines: List[Tuple[LineString, str]]
) -> List[Dict[str, Any]]:
    """Generate 3-ring feathered 160m metric choke-point clouds around trunk/primary bridges."""
    if not bridge_lines:
        return []

    logging.info("Generating 3-ring feathered choke clouds for %d major bridges...", len(bridge_lines))
    records = [{"geometry": geom, "name": name} for geom, name in bridge_lines]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(epsg=27700)

    gdf["geometry"] = gdf.geometry.buffer(160.0, cap_style="round")
    dissolved = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
    dissolved["mid_geom"] = dissolved.geometry.buffer(55.0, cap_style="round")
    dissolved["halo_geom"] = dissolved.geometry.buffer(115.0, cap_style="round")

    core_wgs = dissolved.to_crs(epsg=4326)
    mid_wgs = gpd.GeoSeries(dissolved["mid_geom"], crs="EPSG:27700").to_crs(epsg=4326)
    halo_wgs = gpd.GeoSeries(dissolved["halo_geom"], crs="EPSG:27700").to_crs(epsg=4326)
    wales_clip = box(*WALES_BBOX)

    features: List[Dict[str, Any]] = []
    for idx in range(len(core_wgs)):
        base_props: Dict[str, Any] = {
            "id": f"HAZ_CHOKE_BRIDGE_{idx}",
            "name": "Main Road Bridge Bottleneck",
            "category": "hazards",
            "plainBadge": "BRIDGE BOTTLENECK",
            "plainSummary": "Key single-route bridge crossing that easily jams or closes during floods and emergencies.",
            "categoryLabel": "SINGLE-ACCESS BRIDGE DEFILE",
            "accentColor": "#FFB800",
            "threatWeight": 5.0,
            "primaryMetric": "160m Corridor Defile | Routing Penalty: W_choke +5.0",
            "operationalAction": "Primary river/valley crossing bottleneck. Vulnerable to flood debris, closure, or gridlock.",
        }

        for ring_name, geom_obj, id_suffix in [
            ("halo", halo_wgs.iloc[idx].intersection(wales_clip), "_HALO"),
            ("mid", mid_wgs.iloc[idx].intersection(wales_clip), "_MID"),
            ("core", core_wgs.geometry.iloc[idx].intersection(wales_clip), ""),
        ]:
            if geom_obj.is_empty:
                continue
            g_dict = mapping(geom_obj)
            g_dict["coordinates"] = round_coordinates(g_dict["coordinates"], 5)
            props = dict(base_props)
            props["id"] = f"{base_props['id']}{id_suffix}"
            props["cloudRing"] = ring_name
            features.append(
                {
                    "type": "Feature",
                    "geometry": g_dict,
                    "properties": props,
                }
            )
    return features


def harvest_datamapwales_wfs_layer(
    type_name: str,
    id_prefix: str,
    name_default: str,
    category: str,
    plain_badge: str,
    plain_summary: str,
    category_label: str,
    accent_color: str,
    threat_weight: float,
    metric_text: str,
    action_text: str,
    max_count: int = 80,
) -> List[Dict[str, Any]]:
    """Generic keyless OGC WFS poller for DataMapWales statutory layers (OGL v3.0)."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "count": str(max_count),
    }
    headers = {"User-Agent": "DatumZero-ETL/4.2 (Civil Resilience Pipeline)"}
    features: List[Dict[str, Any]] = []

    try:
        logging.info("Querying DataMapWales WFS (%s)...", type_name)
        res = requests.get(DATAMAPWALES_WFS_URL, params=params, headers=headers, timeout=12)
        if res.status_code == 200:
            data = res.json()
            raw_feats = data.get("features", [])
            logging.info("Retrieved %d features from %s.", len(raw_feats), type_name)
            for idx, item in enumerate(raw_feats):
                geom = item.get("geometry")
                if not geom:
                    continue
                shp = shape(geom).simplify(0.0006, preserve_topology=True)
                if shp.is_empty:
                    continue
                props = item.get("properties", {})
                raw_name = (
                    props.get("sssi_name")
                    or props.get("site_name")
                    or props.get("name")
                    or name_default
                )
                base_props = {
                    "id": f"{id_prefix}_{idx}",
                    "name": str(raw_name),
                    "category": category,
                    "plainBadge": plain_badge,
                    "plainSummary": plain_summary,
                    "categoryLabel": category_label,
                    "accentColor": accent_color,
                    "threatWeight": threat_weight,
                    "primaryMetric": metric_text,
                    "operationalAction": action_text,
                }
                features.extend(
                    make_three_ring_cloud_polygons(
                        shp,
                        core_buffer_m=0.0,
                        mid_extra_m=120.0,
                        halo_extra_m=260.0,
                        base_props=base_props,
                        morph_smooth_m=80.0,
                    )
                )
        else:
            logging.info("DataMapWales WFS %s returned HTTP %s (using built-in baseline).", type_name, res.status_code)
    except (requests.RequestException, ValueError) as exc:
        logging.warning("DataMapWales WFS %s skipped (%s).", type_name, exc)

    return features


def harvest_defra_aurn_telemetry() -> List[Dict[str, Any]]:
    """Poll DEFRA UK-AIR SOS API for live Welsh particulate station status."""
    live_labels: Dict[str, str] = {}
    headers = {"User-Agent": "DatumZero-ETL/4.2 (Civil Resilience Pipeline)"}
    try:
        res = requests.get(DEFRA_UKAIR_STATIONS_URL, headers=headers, timeout=10)
        if res.status_code == 200:
            stations = res.json()
            for st in stations:
                props = st.get("properties", {})
                label = str(props.get("label", ""))
                if any(
                    k in label.lower()
                    for k in ("port talbot", "cardiff", "swansea", "wrexham", "aston hill", "narberth")
                ):
                    live_labels[label.lower()] = label
            logging.info("Verified %d active Welsh stations on DEFRA UK-AIR SOS.", len(live_labels))
    except (requests.RequestException, ValueError) as exc:
        logging.info("DEFRA UK-AIR SOS endpoint skipped (%s); using baseline AURN stations.", exc)

    features: List[Dict[str, Any]] = []
    for st in VERIFIED_DEFRA_AURN_STATIONS:
        is_online = any(
            key in st["name"].lower() for key in ("port talbot", "cardiff", "swansea", "wrexham", "aston hill", "narberth")
        ) and bool(live_labels)
        status_prefix = "ONLINE (UK-AIR AURN)" if is_online else "AURN BASELINE STATION"
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": st["coords"]},
                "properties": {
                    "id": st["id"],
                    "name": st["name"],
                    "category": "hazards",
                    "plainBadge": "AIR & SMOKE SENSOR",
                    "plainSummary": st["plainSummary"],
                    "categoryLabel": "DEFRA AIR QUALITY SENSOR",
                    "accentColor": "#FF3B30",
                    "threatWeight": 1.5,
                    "primaryMetric": f"{status_prefix} | {st['metric']}",
                    "operationalAction": st["action"],
                },
            }
        )
    return features


def build_priority_1_and_2_layers() -> List[Dict[str, Any]]:
    """
    Compile Priority 1 & 2 Geospatial Intelligence as organic, 3-ring feathered clouds
    with dual plain-English and technical metadata tiers.
    """
    features: List[Dict[str, Any]] = []

    for dz in VERIFIED_CELLULAR_DEAD_ZONES:
        line = LineString(dz["line"])
        buf = float(dz["buffer_m"])
        base_props = {
            "id": dz["id"],
            "name": dz["name"],
            "category": "grid_down",
            "plainBadge": "NO MOBILE SIGNAL",
            "plainSummary": dz["plainSummary"],
            "categoryLabel": "CELLULAR DEAD ZONE",
            "accentColor": "#FFB800",
            "threatWeight": 2.5,
            "primaryMetric": dz["metric"],
            "operationalAction": dz["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                line,
                core_buffer_m=buf,
                mid_extra_m=buf * 0.25,
                halo_extra_m=buf * 0.55,
                base_props=base_props,
            )
        )

    for tx in VERIFIED_RF_TRANSMITTERS:
        pt = Point(tx["coords"][0], tx["coords"][1])
        rad = float(tx["radius_m"])
        base_props = {
            "id": tx["id"],
            "name": tx["name"],
            "category": "grid_down",
            "plainBadge": "HIGH-POWER RADIO MAST",
            "plainSummary": tx["plainSummary"],
            "categoryLabel": "RF TRANSMITTER FOOTPRINT",
            "accentColor": "#FFB800",
            "threatWeight": 0.5,
            "primaryMetric": tx["metric"],
            "operationalAction": tx["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                pt,
                core_buffer_m=rad,
                mid_extra_m=rad * 0.28,
                halo_extra_m=rad * 0.60,
                base_props=base_props,
            )
        )

    for rz in VERIFIED_RADON_ZONES:
        poly = Polygon(rz["poly"])
        base_props = {
            "id": rz["id"],
            "name": rz["name"],
            "category": "hazards",
            "plainBadge": "RADON GAS AREA",
            "plainSummary": rz["plainSummary"],
            "categoryLabel": "RADON GAS ACTION ZONE",
            "accentColor": "#FFB800",
            "threatWeight": 1.0,
            "primaryMetric": rz["metric"],
            "operationalAction": rz["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                poly,
                core_buffer_m=0.0,
                mid_extra_m=650.0,
                halo_extra_m=1450.0,
                base_props=base_props,
                morph_smooth_m=1200.0,
            )
        )

    for sssi in VERIFIED_SSSI_ZONES:
        poly = Polygon(sssi["poly"])
        base_props = {
            "id": sssi["id"],
            "name": sssi["name"],
            "category": "resupply",
            "plainBadge": "NO FORAGING (PROTECTED)",
            "plainSummary": sssi["plainSummary"],
            "categoryLabel": "SSSI PROTECTED ZONE (NO FORAGING)",
            "accentColor": "#FF3B30",
            "threatWeight": 0.0,
            "primaryMetric": sssi["metric"],
            "operationalAction": sssi["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                poly,
                core_buffer_m=0.0,
                mid_extra_m=450.0,
                halo_extra_m=1000.0,
                base_props=base_props,
                morph_smooth_m=900.0,
            )
        )

    for fb in VERIFIED_FORAGING_BIOMES:
        poly = Polygon(fb["poly"])
        base_props = {
            "id": fb["id"],
            "name": fb["name"],
            "category": "resupply",
            "plainBadge": "WILD FOOD WOODLAND",
            "plainSummary": fb["plainSummary"],
            "categoryLabel": "SUBSISTENCE FORAGING BIOME",
            "accentColor": "#00E5FF",
            "threatWeight": 0.0,
            "primaryMetric": fb["metric"],
            "operationalAction": fb["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                poly,
                core_buffer_m=0.0,
                mid_extra_m=450.0,
                halo_extra_m=1000.0,
                base_props=base_props,
                morph_smooth_m=850.0,
            )
        )

    for crow in VERIFIED_CROW_ACCESS_LAND:
        poly = Polygon(crow["poly"])
        base_props = {
            "id": crow["id"],
            "name": crow["name"],
            "category": "baseline",
            "plainBadge": "OPEN WALKING LAND",
            "plainSummary": crow["plainSummary"],
            "categoryLabel": "CROW OPEN ACCESS LAND",
            "accentColor": "#94A3B8",
            "threatWeight": 0.0,
            "primaryMetric": crow["metric"],
            "operationalAction": crow["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                poly,
                core_buffer_m=0.0,
                mid_extra_m=550.0,
                halo_extra_m=1250.0,
                base_props=base_props,
                morph_smooth_m=1100.0,
            )
        )

    # 7. BGS Clean Water Groundwater Aquifers (Resupply Polygons)
    for aq in VERIFIED_AQUIFER_ZONES:
        poly = Polygon(aq["poly"])
        base_props = {
            "id": aq["id"],
            "name": aq["name"],
            "category": "resupply",
            "plainBadge": "GROUNDWATER AQUIFER",
            "plainSummary": aq["plainSummary"],
            "categoryLabel": "BGS CLEAN WATER AQUIFER",
            "accentColor": "#00E5FF",
            "threatWeight": 0.0,
            "primaryMetric": aq["metric"],
            "operationalAction": aq["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                poly,
                core_buffer_m=0.0,
                mid_extra_m=600.0,
                halo_extra_m=1350.0,
                base_props=base_props,
                morph_smooth_m=1100.0,
            )
        )

    # 8. BGS Landslide & Steep Valley Slope Instability Corridors (Hazards)
    for sl in VERIFIED_LANDSLIDE_ZONES:
        line = LineString(sl["line"])
        buf = float(sl["buffer_m"])
        base_props = {
            "id": sl["id"],
            "name": sl["name"],
            "category": "hazards",
            "plainBadge": "LANDSLIDE & SCREE ZONE",
            "plainSummary": sl["plainSummary"],
            "categoryLabel": "SLOPE INSTABILITY CORRIDOR",
            "accentColor": "#FF3B30",
            "threatWeight": 3.0,
            "primaryMetric": sl["metric"],
            "operationalAction": sl["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                line,
                core_buffer_m=buf,
                mid_extra_m=buf * 0.28,
                halo_extra_m=buf * 0.60,
                base_props=base_props,
            )
        )

    # 9. MoD Live-Fire & Restricted Training Areas (National Security Act 2023 Compliant)
    for mod in VERIFIED_MOD_RESTRICTED_ZONES:
        poly = Polygon(mod["poly"])
        base_props = {
            "id": mod["id"],
            "name": mod["name"],
            "category": "hazards",
            "plainBadge": "MOD FIRING RANGE (NO ENTRY)",
            "plainSummary": mod["plainSummary"],
            "categoryLabel": "MOD RESTRICTED DANGER AREA",
            "accentColor": "#FF3B30",
            "threatWeight": 10.0,
            "primaryMetric": mod["metric"],
            "operationalAction": mod["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                poly,
                core_buffer_m=0.0,
                mid_extra_m=450.0,
                halo_extra_m=1000.0,
                base_props=base_props,
                morph_smooth_m=800.0,
            )
        )

    return features


def build_verified_hazard_layers() -> List[Dict[str, Any]]:
    """Compile statutory NRW gauges, terrain/tidal defiles, COMAH buffers, and flood plains."""
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
                    "plainBadge": "RIVER FLOOD GAUGE",
                    "plainSummary": g["plainSummary"],
                    "categoryLabel": "NRW RIVER TELEMETRY",
                    "accentColor": "#FF3B30",
                    "threatWeight": 5.0,
                    "primaryMetric": g["metric"],
                    "operationalAction": g["action"],
                },
            }
        )

    for th in VERIFIED_TERRAIN_AND_TIDAL_HAZARDS:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": th["coords"]},
                "properties": {
                    "id": th["id"],
                    "name": th["name"],
                    "category": "hazards",
                    "plainBadge": th["plainBadge"],
                    "plainSummary": th["plainSummary"],
                    "categoryLabel": th["label"],
                    "accentColor": "#FF3B30",
                    "threatWeight": 8.0,
                    "primaryMetric": th["metric"],
                    "operationalAction": th["action"],
                },
            }
        )

    for site in VERIFIED_COMAH_SITES:
        pt = Point(site["coords"][0], site["coords"][1])
        rad = float(site["radius_m"])
        base_props = {
            "id": site["id"],
            "name": site["name"],
            "category": "hazards",
            "plainBadge": "CHEMICAL HAZARD ZONE",
            "plainSummary": site["plainSummary"],
            "categoryLabel": "HSE COMAH HAZMAT BUFFER",
            "accentColor": "#FF3B30",
            "threatWeight": 8.0,
            "primaryMetric": site["metric"],
            "operationalAction": site["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                pt,
                core_buffer_m=rad,
                mid_extra_m=rad * 0.25,
                halo_extra_m=rad * 0.55,
                base_props=base_props,
            )
        )

    for crag in VERIFIED_MOUNTAIN_CRAG_CLOUDS:
        line = LineString(crag["line"])
        buf = float(crag["buffer_m"])
        base_props = {
            "id": crag["id"],
            "name": crag["name"],
            "category": "hazards",
            "plainBadge": "STEEP CLIFF & RIDGE HAZARD",
            "plainSummary": crag["plainSummary"],
            "categoryLabel": "MOUNTAIN CRAG & RESCUE CLUSTER",
            "accentColor": "#FF3B30",
            "threatWeight": 6.0,
            "primaryMetric": crag["metric"],
            "operationalAction": crag["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                line,
                core_buffer_m=buf,
                mid_extra_m=buf * 0.30,
                halo_extra_m=buf * 0.65,
                base_props=base_props,
            )
        )

    for corr in VERIFIED_FLOOD_CORRIDORS:
        line = LineString(corr["line"])
        buf = float(corr["buffer_m"])
        base_props = {
            "id": corr["id"],
            "name": corr["name"],
            "category": "hazards",
            "plainBadge": "FLOOD DANGER ZONE",
            "plainSummary": corr["plainSummary"],
            "categoryLabel": "FLOOD & HYDROLOGICAL CORRIDOR",
            "accentColor": "#FF3B30",
            "threatWeight": 10.0,
            "primaryMetric": corr["metric"],
            "operationalAction": corr["action"],
        }
        features.extend(
            make_three_ring_cloud_polygons(
                line,
                core_buffer_m=buf,
                mid_extra_m=buf * 0.28,
                halo_extra_m=buf * 0.60,
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
                "plainBadge": "EMERGENCY RADIO RELAY",
                "plainSummary": rep["plainSummary"],
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
                    "plainBadge": item["plainBadge"],
                    "plainSummary": item["plainSummary"],
                    "categoryLabel": item["categoryLabel"],
                    "accentColor": item["accentColor"],
                    "threatWeight": 0.0,
                    "primaryMetric": item["metric"],
                    "operationalAction": item["action"],
                },
            }
        )
    for pt in VERIFIED_EMERGENCY_AND_COMMS_POINTS:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": pt["coords"]},
                "properties": {
                    "id": pt["id"],
                    "name": pt["name"],
                    "category": pt["category"],
                    "plainBadge": pt["plainBadge"],
                    "plainSummary": pt["plainSummary"],
                    "categoryLabel": pt["categoryLabel"],
                    "accentColor": pt["accentColor"],
                    "threatWeight": 0.0,
                    "primaryMetric": pt["metric"],
                    "operationalAction": pt["action"],
                },
            }
        )

    for heli in VERIFIED_HELIPADS:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": heli["coords"]},
                "properties": {
                    "id": heli["id"],
                    "name": heli["name"],
                    "category": "baseline",
                    "plainBadge": "EMERGENCY HELIPAD",
                    "plainSummary": heli["plainSummary"],
                    "categoryLabel": "CAA / HEMS HELIPAD",
                    "accentColor": "#10B981",
                    "threatWeight": 0.0,
                    "primaryMetric": heli["metric"],
                    "operationalAction": heli["action"],
                },
            }
        )

    return features


def _fetch_single_poi_tile(
    tile_idx: int, region_name: str, tile_bbox: List[float]
) -> Tuple[str, List[float], Optional[Dict[str, Any]]]:
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    poi_query = f"""
    [out:json][timeout:20];
    (
      node["natural"="spring"]({bbox_str});
      node["amenity"="drinking_water"]({bbox_str});
      node["man_made"="water_well"]({bbox_str});
      node["man_made"="water_tower"]({bbox_str});
      node["emergency"="defibrillator"]({bbox_str});
      node["amenity"="hospital"]({bbox_str});
      way["amenity"="hospital"]({bbox_str});
      node["amenity"="clinic"]({bbox_str});
      node["man_made"="mast"]["tower:type"="communication"]({bbox_str});
      node["man_made"="surveillance"]({bbox_str});
      node["aeroway"="helipad"]({bbox_str});
      way["aeroway"="helipad"]({bbox_str});
      node["amenity"="fire_station"]({bbox_str});
      way["amenity"="fire_station"]({bbox_str});
      node["emergency"="lifeboat_station"]({bbox_str});
      way["emergency"="lifeboat_station"]({bbox_str});
      node["emergency"="mountain_rescue"]({bbox_str});
      way["emergency"="mountain_rescue"]({bbox_str});
      node["ford"="yes"]["highway"]({bbox_str});
      way["ford"="yes"]["highway"]({bbox_str});
    );
    out center;
    """
    return (
        region_name,
        tile_bbox,
        query_overpass_fast(poi_query, f"POIs {region_name}", mirror_offset=tile_idx),
    )


def harvest_osm_layers() -> List[Dict[str, Any]]:
    """Extract OSM POIs, power lines & trunk bridges concurrently across distinct mirrors."""
    point_features: List[Dict[str, Any]] = []
    seen_ids = set()

    min_lon, min_lat, max_lon, max_lat = WALES_BBOX
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    power_query = f"""
    [out:json][timeout:18];
    way["power"="line"]({bbox_str});
    out geom;
    """
    bridge_query = f"""
    [out:json][timeout:18];
    way["bridge"="yes"]["highway"~"^(trunk|primary)$"]({bbox_str});
    out geom;
    """

    with ThreadPoolExecutor(max_workers=4) as pool:
        tile_futures = [
            pool.submit(_fetch_single_poi_tile, idx, name, bbox)
            for idx, (name, bbox) in enumerate(WALES_SUBTILES)
        ]
        power_future = pool.submit(query_overpass_fast, power_query, "PowerLines", 1)
        bridge_future = pool.submit(query_overpass_fast, bridge_query, "TrunkBridges", 2)

        for fut in as_completed(tile_futures):
            region_name, tile_bbox, poi_data = fut.result()
            if poi_data is not None:
                tile_count = 0
                for el in poi_data.get("elements", []):
                    el_type = el.get("type", "node")
                    el_id = f"{el_type}_{el.get('id', '')}"
                    if not el.get("id") or el_id in seen_ids:
                        continue

                    if el_type == "node":
                        lat = el.get("lat")
                        lon = el.get("lon")
                    else:
                        center = el.get("center", {})
                        lat = center.get("lat")
                        lon = center.get("lon")

                    if lat is None or lon is None:
                        continue

                    pt = build_point_feature(
                        el_id, float(lon), float(lat), el.get("tags", {})
                    )
                    if pt:
                        seen_ids.add(el_id)
                        point_features.append(pt)
                        tile_count += 1
                logging.info("Extracted %d points for %s.", tile_count, region_name)
            else:
                logging.warning(
                    "Tile %s unavailable; restoring cached points for this sector.",
                    region_name,
                )
                for cached_pt in restore_cached_points_in_bbox(tile_bbox):
                    cid = str(cached_pt.get("properties", {}).get("id", ""))
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        point_features.append(cached_pt)

        power_data = power_future.result()
        bridge_data = bridge_future.result()

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


def z_order_sort_key(feature: Dict[str, Any]) -> int:
    """
    Order features so MapLibre renders:
    0. Outer dispersion halos (`halo`)
    1. Intermediate dispersion plumes (`mid`)
    2. Core zone boundaries (`core`)
    3. Baseline CCTV points
    4. Resupply & Medical points
    5. Grid Down repeater points
    6. Acute Red (#FF3B30) hazard/telemetry points ON TOP
    """
    geom_type = feature.get("geometry", {}).get("type", "")
    props = feature.get("properties", {})
    category = props.get("category", "")
    ring = props.get("cloudRing", "")

    if geom_type in ("Polygon", "MultiPolygon"):
        if ring == "halo":
            return 0
        if ring == "mid":
            return 1
        return 2
    if category == "baseline":
        return 3
    if category in ("resupply", "medical"):
        return 4
    if category == "grid_down":
        return 5
    return 6


def main() -> None:
    osm_features = harvest_osm_layers()
    static_baseline = get_static_baseline_features()
    verified_hazards = build_verified_hazard_layers()
    priority_layers = build_priority_1_and_2_layers()
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_aurn = pool.submit(harvest_defra_aurn_telemetry)
        f_fz3 = pool.submit(
            harvest_datamapwales_wfs_layer,
            "inspire-nrw:NRW_FLOOD_MAP_FLOOD_ZONE_3_RIVERS_AND_SEA",
            "NRW_FZ3",
            "River & Coastal Flood Zone",
            "hazards",
            "FLOOD DANGER ZONE",
            "Official high-risk floodplain. Avoid low roads and riverbanks during heavy rain.",
            "NRW FLOOD ZONE 3 POLYGON",
            "#FF3B30",
            10.0,
            "High Fluvial/Tidal Probability | Routing Penalty: W_flood +10.0",
            "Statutory 1-in-30-yr floodplain. Avoid low-lying roads during severe weather.",
            100,
        )
        f_sssi = pool.submit(
            harvest_datamapwales_wfs_layer,
            "inspire-nrw:NRW_SSSI",
            "DMW_SSSI",
            "Protected Wildlife Reserve (SSSI)",
            "resupply",
            "NO FORAGING (PROTECTED)",
            "Protected wildlife reserve. Do NOT pick plants, uproot anything, or disturb habitat.",
            "SSSI PROTECTED ZONE (NO FORAGING)",
            "#FF3B30",
            0.0,
            "Statutory SSSI | Wildlife & Countryside Act 1981",
            "FORAGING PROHIBITED: High-Conservation SSSI Zone. No harvesting or uprooting permitted.",
            60,
        )
        f_crow = pool.submit(
            harvest_datamapwales_wfs_layer,
            "inspire-nrw:NRW_OPEN_ACCESS_LAND",
            "DMW_CROW",
            "Open Mountain & Moorland Access",
            "baseline",
            "OPEN WALKING LAND",
            "Public right to walk freely off-path across open mountain and moorland ground.",
            "CROW OPEN ACCESS LAND",
            "#94A3B8",
            0.0,
            "CRoW Act 2000 Statutory Access",
            "Legal right of foot access across uncultivated upland terrain.",
            60,
        )
        defra_aurn = f_aurn.result()
        dmw_flood_zones = f_fz3.result()
        dmw_sssi = f_sssi.result()
        dmw_crow = f_crow.result()

    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for feat in (
        static_baseline
        + verified_hazards
        + priority_layers
        + defra_aurn
        + dmw_flood_zones
        + dmw_sssi
        + dmw_crow
        + osm_features
    ):
        fid = str(feat.get("properties", {}).get("id", len(merged_by_id)))
        merged_by_id[fid] = feat

    all_features = sorted(merged_by_id.values(), key=z_order_sort_key)

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
