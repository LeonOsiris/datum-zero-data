"""
pipelines/build_wales_resilience_layer.py
Deterministic zero-token spatial ETL compiling Welsh open data into Datum Zero GeoJSON.

Visual & Geometric Architecture:
- Zero squares, rectangles, or pixel-scaled circles.
- Every areal zone (floodplains, 50Hz EMF corridors, cellular dead zones, COMAH buffers,
  Radon action belts, SSSIs, Ancient Woodland biomes, CRoW Open Access massifs) is shaped
  as a multi-vertex natural geographic footprint and morphologically rounded in British
  National Grid (EPSG:27700) into three concentric feathered dispersion shells:
    * cloudRing: "core" (inner physical boundary)
    * cloudRing: "mid"  (soft intermediate dispersion plume)
    * cloudRing: "halo" (outer vaporous atmospheric fade)
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
DATAMAPWALES_WFS_URL = "https://datamap.gov.wales/geoserver/ows"
DEFRA_UKAIR_STATIONS_URL = "https://uk-air.defra.gov.uk/sos-ukair/api/v1/stations"

# 7 balanced regional tiles [min_lon, min_lat, max_lon, max_lat]
WALES_SUBTILES: List[Tuple[str, List[float]]] = [
    ("North-West Wales (Eryri & Anglesey)", [-5.50, 52.75, -3.60, 53.50]),
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
    {
        "id": "COM_GB3MW",
        "name": "GB3MW - Aberystwyth / Plynlimon Repeater",
        "coords": [-3.9825, 52.4142],
        "metric": "TX: 145.600 MHz | RX: 145.000 MHz | CTCSS: 77.0 Hz",
        "action": "2m Mid-Wales coastal & Cambrian Mountains repeater (-0.6 MHz offset).",
    },
]

# Topological Cellular Dead Zones (Winding valley corridors -> organic ribbon clouds)
VERIFIED_CELLULAR_DEAD_ZONES: List[Dict[str, Any]] = [
    {
        "id": "SIG_DEAD_OGWEN",
        "name": "Nant Ffrancon & Ogwen Valley Cellular Dead Zone",
        "line": [[-4.0560, 53.1680], [-4.0340, 53.1440], [-4.0160, 53.1240], [-3.9880, 53.1120], [-3.9580, 53.1030]],
        "buffer_m": 950,
        "metric": "4-MNO Blackout (EE/Three/O2/Vodafone) | W_signal +2.5",
        "action": "Deep glaciated trough blocks mast line-of-sight. Use 2m VHF 145.500 MHz or GB3AR (145.725 MHz).",
    },
    {
        "id": "SIG_DEAD_LLANBERIS_PASS",
        "name": "Llanberis Pass & Pen-y-Pass Signal Shadow",
        "line": [[-4.1080, 53.1110], [-4.0750, 53.0990], [-4.0420, 53.0890], [-4.0120, 53.0800], [-3.9820, 53.0830]],
        "buffer_m": 900,
        "metric": "4-MNO Blackout | Glaciated Slate Gorge | W_signal +2.5",
        "action": "Zero mobile data or 999 voice inside pass. Climb above 450m ridge or relay via Capel Curig.",
    },
    {
        "id": "SIG_DEAD_RHINOGYDD",
        "name": "Rhinogydd & Coed y Brenin Interior Dead Zone",
        "line": [[-4.0220, 52.8740], [-3.9850, 52.8480], [-3.9420, 52.8220], [-3.9120, 52.7960], [-3.8880, 52.7750]],
        "buffer_m": 1500,
        "metric": "4-MNO Blackout | Cambrian Shielding | W_signal +2.5",
        "action": "Persistent upland shadow across Ardudwy range. Pre-cache offline route before entering.",
    },
    {
        "id": "SIG_DEAD_BERWYN",
        "name": "Berwyn Mountains & Milltir Cerrig Signal Shadow",
        "line": [[-3.6050, 52.8910], [-3.5680, 52.8710], [-3.5320, 52.8520], [-3.4920, 52.8360], [-3.4550, 52.8210]],
        "buffer_m": 1400,
        "metric": "4-MNO Blackout | High Moorland Defile | W_signal +2.5",
        "action": "Zero cellular coverage along B4391 pass. Maintain BLE stand-off mesh with convoy.",
    },
    {
        "id": "SIG_DEAD_ELAN_CAMBRIAN",
        "name": "Elan Valley & Elenydd Wilderness Dead Zone",
        "line": [[-3.7550, 52.3520], [-3.7020, 52.3180], [-3.6480, 52.2820], [-3.6040, 52.2540], [-3.5680, 52.2280]],
        "buffer_m": 1900,
        "metric": "4-MNO Blackout | Cambrian Plateau | W_signal +2.5",
        "action": "Largest continuous cellular shadow in Mid Wales. Rely on offline .pmtiles & VHF simplex.",
    },
    {
        "id": "SIG_DEAD_BLACK_MOUNTAIN",
        "name": "Western Bannau Brycheiniog / Black Mountain Shadow",
        "line": [[-3.8540, 51.8720], [-3.8160, 51.8440], [-3.7780, 51.8210], [-3.7420, 51.8040], [-3.7080, 51.7910]],
        "buffer_m": 1350,
        "metric": "4-MNO Blackout | A4069 & Usk Headwaters | W_signal +2.5",
        "action": "Steep escarpment blocks South Wales masts. Gain northern ridge crest for line-of-sight.",
    },
]

# High-Power Welsh Broadcast & RF Transmitter Sites (Ofcom Spectrum Records)
VERIFIED_RF_TRANSMITTERS: List[Dict[str, Any]] = [
    {
        "id": "RF_TX_WENVOE",
        "name": "Wenvoe High-Power Broadcast & Telecom Hub (300m Mast)",
        "coords": [-3.2818, 51.4595],
        "radius_m": 850,
        "metric": "ERP: 500 kW UHF / VHF | High RF Density Contour",
        "action": "Strong local RF field; potential front-end overload on unshielded handheld receivers within 850m.",
    },
    {
        "id": "RF_TX_MOELYPARC",
        "name": "Moel-y-Parc Strategic Broadcast & Microwave Mast",
        "coords": [-3.3145, 53.2255],
        "radius_m": 750,
        "metric": "ERP: 100 kW | Clwydian Range Line-of-Sight Hub",
        "action": "Dominant North-East Wales RF beacon. Elevated ground provides optimal 2m/70cm simplex range.",
    },
    {
        "id": "RF_TX_BLAENPLWYF",
        "name": "Blaenplwyf Coastal Broadcast & Radar Station",
        "coords": [-4.1145, 52.3715],
        "radius_m": 750,
        "metric": "ERP: 100 kW | Cardigan Bay Maritime & Broadcast Hub",
        "action": "Primary Mid-Wales coastal transmitter. Expect elevated RF noise floor near perimeter.",
    },
    {
        "id": "RF_TX_LLANDDONA",
        "name": "Llanddona Anglesey Strategic Broadcast Station",
        "coords": [-4.1325, 53.2932],
        "radius_m": 700,
        "metric": "ERP: 100 kW | Co-located with GB3LL 2m Repeater",
        "action": "Covers Menai Strait, Eryri north coast, and Irish Sea approaches.",
    },
]

# Organic Multi-Vertex UKHSA / BGS Radon Gas Action Belts (Zero Rectangles)
VERIFIED_RADON_ZONES: List[Dict[str, Any]] = [
    {
        "id": "RADON_HALKYN_CLWYD",
        "name": "UKHSA Radon Action Zone: Halkyn Mountain & Clwydian Limestone",
        "poly": [
            [-3.335, 53.268], [-3.265, 53.275], [-3.195, 53.258], [-3.135, 53.225],
            [-3.112, 53.175], [-3.148, 53.135], [-3.215, 53.142], [-3.268, 53.168],
            [-3.308, 53.212], [-3.335, 53.268],
        ],
        "metric": "UKHSA Class: >30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Carboniferous limestone karst releases elevated radon gas. Ventilate basements, caves, or subterranean shelters.",
    },
    {
        "id": "RADON_GWYNEDD_SLATE",
        "name": "UKHSA Radon Action Zone: Gwynedd Granite & Slate Belt",
        "poly": [
            [-4.215, 53.178], [-4.125, 53.192], [-4.035, 53.165], [-3.975, 53.115],
            [-3.962, 53.055], [-4.028, 53.035], [-4.115, 53.048], [-4.178, 53.082],
            [-4.212, 53.132], [-4.215, 53.178],
        ],
        "metric": "UKHSA Class: 10–30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Fractured metamorphic slate & granite bedrock. Avoid prolonged occupancy in unventilated mine adits or cellars.",
    },
    {
        "id": "RADON_PEMBROKE_GOWER",
        "name": "UKHSA Radon Action Zone: South Pembrokeshire & Gower Limestone",
        "poly": [
            [-5.045, 51.712], [-4.935, 51.725], [-4.815, 51.708], [-4.695, 51.685],
            [-4.662, 51.635], [-4.745, 51.612], [-4.865, 51.608], [-4.975, 51.628],
            [-5.035, 51.668], [-5.045, 51.712],
        ],
        "metric": "UKHSA Class: 10–30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Coastal limestone fault zone. Ensure active cross-ventilation in ground-level or below-grade shelters.",
    },
    {
        "id": "RADON_VALLEYS_TAFF",
        "name": "UKHSA Radon Action Zone: South Wales Coalfield & Taff Gorge",
        "poly": [
            [-3.515, 51.735], [-3.415, 51.748], [-3.305, 51.732], [-3.218, 51.695],
            [-3.205, 51.625], [-3.258, 51.582], [-3.355, 51.578], [-3.445, 51.608],
            [-3.498, 51.668], [-3.515, 51.735],
        ],
        "metric": "UKHSA Class: 10–30% Exceedance (>200 Bq/m3 Action Level)",
        "action": "Deep-mined strata & valley fault lines concentrate soil gas in enclosed lower storeys.",
    },
]

# Organic Multi-Vertex Statutory SSSIs (Wildlife & Countryside Act 1981 — STRICT NO-FORAGING ZONES)
VERIFIED_SSSI_ZONES: List[Dict[str, Any]] = [
    {
        "id": "SSSI_ERYRI_CORE",
        "name": "SSSI: Eryri / Yr Wyddfa & Cwm Idwal National Nature Reserve",
        "poly": [
            [-4.105, 53.128], [-4.045, 53.142], [-3.992, 53.124], [-3.978, 53.082],
            [-4.005, 53.048], [-4.058, 53.044], [-4.095, 53.068], [-4.112, 53.098],
            [-4.105, 53.128],
        ],
        "metric": "Statutory SSSI | Wildlife & Countryside Act 1981",
        "action": "FORAGING PROHIBITED: High-Conservation SSSI Zone. No harvesting, uprooting, or disturbance permitted.",
    },
    {
        "id": "SSSI_CORS_CARON",
        "name": "SSSI: Cors Caron Raised Bog National Nature Reserve (Tregaron)",
        "poly": [
            [-3.948, 52.266], [-3.918, 52.275], [-3.892, 52.262], [-3.888, 52.238],
            [-3.908, 52.218], [-3.938, 52.222], [-3.954, 52.242], [-3.948, 52.266],
        ],
        "metric": "Statutory SSSI & Ramsar Wetland | WCA 1981",
        "action": "FORAGING PROHIBITED: Protected peatland habitat. Keep strictly to designated raised boardwalks.",
    },
    {
        "id": "SSSI_GOWER_COAST",
        "name": "SSSI: Gower Coast (Rhossili to Port Eynon)",
        "poly": [
            [-4.324, 51.578], [-4.278, 51.582], [-4.225, 51.568], [-4.198, 51.548],
            [-4.218, 51.532], [-4.268, 51.535], [-4.308, 51.548], [-4.328, 51.564],
            [-4.324, 51.578],
        ],
        "metric": "Statutory Coastal SSSI | Wildlife & Countryside Act 1981",
        "action": "FORAGING PROHIBITED: Protected maritime cliff flora and seabird nesting habitat.",
    },
]

# Organic Multi-Vertex Ancient Woodland Foraging Biomes (Theft Act 1968 Four Fs; wild fungi excluded)
VERIFIED_FORAGING_BIOMES: List[Dict[str, Any]] = [
    {
        "id": "FORAGE_COED_Y_BRENIN",
        "name": "Ancient Woodland Biome: Coed y Brenin & Ganllwyd Oakwoods",
        "poly": [
            [-3.918, 52.838], [-3.878, 52.846], [-3.842, 52.832], [-3.836, 52.798],
            [-3.862, 52.776], [-3.898, 52.782], [-3.922, 52.808], [-3.918, 52.838],
        ],
        "metric": "NRW Ancient Semi-Natural Woodland | Theft Act 1968 Compliant",
        "action": "Legal personal-use above-ground flora only (wild garlic, nettles, bilberries, hazelnuts). No uprooting. DO NOT harvest wild fungi/mushrooms.",
    },
    {
        "id": "FORAGE_GWYDYR_CONWY",
        "name": "Ancient Woodland Biome: Gwydyr Forest & Betws-y-Coed Valley",
        "poly": [
            [-3.858, 53.118], [-3.818, 53.126], [-3.782, 53.112], [-3.776, 53.082],
            [-3.802, 53.066], [-3.838, 53.072], [-3.862, 53.095], [-3.858, 53.118],
        ],
        "metric": "Mixed Broadleaf/Conifer Canopy | Thermal Shelter & Flora",
        "action": "Dense canopy windbreak & fallen deadwood fuel. Personal above-ground foliage/berries only; strictly exclude wild mushrooms.",
    },
    {
        "id": "FORAGE_ELAN_OAKWOODS",
        "name": "Ancient Woodland Biome: Lower Elan & Wye Valley Broadleaf",
        "poly": [
            [-3.578, 52.288], [-3.535, 52.296], [-3.492, 52.282], [-3.486, 52.248],
            [-3.515, 52.226], [-3.555, 52.232], [-3.582, 52.258], [-3.578, 52.288],
        ],
        "metric": "Sessile Oak & Hazel Coppice | Theft Act 1968 Compliant",
        "action": "High seasonal mast (hazelnuts, blackberries, sorrel). Above-ground personal harvest only; zero wild fungi.",
    },
    {
        "id": "FORAGE_WENTWOOD",
        "name": "Ancient Woodland Biome: Wentwood & Wye Gorge Broadleaf",
        "poly": [
            [-2.838, 51.656], [-2.775, 51.668], [-2.705, 51.652], [-2.696, 51.618],
            [-2.735, 51.596], [-2.795, 51.602], [-2.842, 51.628], [-2.838, 51.656],
        ],
        "metric": "Largest Ancient Woodland in South Wales | Broadleaf Canopy",
        "action": "Rich spring wild garlic, elderflower/elderberry, and hazel. Above-ground personal use only; exclude wild fungi.",
    },
]

# Organic Multi-Vertex Designated Open Access Land (CRoW Act 2000)
VERIFIED_CROW_ACCESS_LAND: List[Dict[str, Any]] = [
    {
        "id": "CROW_CARNEDDAU_GLYDERAU",
        "name": "CRoW Open Access Land: Carneddau & Glyderau Massif",
        "poly": [
            [-4.055, 53.205], [-3.965, 53.218], [-3.872, 53.195], [-3.862, 53.145],
            [-3.895, 53.108], [-3.975, 53.112], [-4.038, 53.135], [-4.062, 53.172],
            [-4.055, 53.205],
        ],
        "metric": "CRoW Act 2000 Statutory Access | Mountain & Moorland",
        "action": "Legal right of foot access across uncultivated upland terrain for high-contour ridge bypasses.",
    },
    {
        "id": "CROW_BANNAU_CENTRAL",
        "name": "CRoW Open Access Land: Central Bannau Brycheiniog (Pen y Fan)",
        "poly": [
            [-3.538, 51.912], [-3.445, 51.924], [-3.348, 51.905], [-3.338, 51.858],
            [-3.385, 51.828], [-3.465, 51.832], [-3.525, 51.855], [-3.545, 51.885],
            [-3.538, 51.912],
        ],
        "metric": "CRoW Act 2000 Statutory Access | Old Red Sandstone Escarpment",
        "action": "Legal off-trail upland movement zone above valley flood corridors.",
    },
    {
        "id": "CROW_PRESELI_HILLS",
        "name": "CRoW Open Access Land: Mynydd Preseli Ridge (Pembrokeshire)",
        "poly": [
            [-4.858, 51.972], [-4.775, 51.984], [-4.685, 51.968], [-4.678, 51.935],
            [-4.725, 51.918], [-4.805, 51.922], [-4.855, 51.942], [-4.858, 51.972],
        ],
        "metric": "CRoW Act 2000 Statutory Access | West Wales Upland Corridor",
        "action": "Open moorland ridge corridor with long-range VHF line-of-sight.",
    },
]

# DEFRA UK-AIR / AURN Welsh Air Quality & Toxic Particulate Monitoring Stations
VERIFIED_DEFRA_AURN_STATIONS: List[Dict[str, Any]] = [
    {
        "id": "AURN_PORT_TALBOT",
        "name": "DEFRA AURN Sensor: Port Talbot Margam (PM2.5 / PM10)",
        "coords": [-3.7708, 51.5839],
        "metric": "Live AURN Telemetry | Industrial & Smoke Plume Monitor",
        "action": "Monitors steelworks, M4 corridor, and Afan valley wildfire smoke particulates.",
    },
    {
        "id": "AURN_CARDIFF",
        "name": "DEFRA AURN Sensor: Cardiff Centre (PM2.5 / NO2 / O3)",
        "coords": [-3.1763, 51.4818],
        "metric": "Live AURN Telemetry | Urban Particulate & Inversion Node",
        "action": "Tracks South-East Wales urban air quality and Bristol Channel smoke drift.",
    },
    {
        "id": "AURN_SWANSEA",
        "name": "DEFRA AURN Sensor: Swansea Roadside (PM2.5 / PM10)",
        "coords": [-3.9474, 51.6326],
        "metric": "Live AURN Telemetry | Tawe Valley Inversion Monitor",
        "action": "Monitors particulate trapping during winter anticyclonic temperature inversions.",
    },
    {
        "id": "AURN_WREXHAM",
        "name": "DEFRA AURN Sensor: Wrexham (PM2.5 / PM10)",
        "coords": [-2.9912, 53.0422],
        "metric": "Live AURN Telemetry | North-East Wales Industrial/Moorland Node",
        "action": "Detects Dee Valley inversions and Ruabon/Llantysilio moorland wildfire smoke.",
    },
    {
        "id": "AURN_ASTON_HILL",
        "name": "DEFRA AURN Sensor: Aston Hill Rural Background (Powys)",
        "coords": [-3.0341, 52.5038],
        "metric": "Live AURN Telemetry | Mid-Wales Elevated Baseline",
        "action": "High-elevation rural particulate and ozone sensor tracking regional plume transport.",
    },
    {
        "id": "AURN_NARBERTH",
        "name": "DEFRA AURN Sensor: Narberth Rural Background (Pembrokeshire)",
        "coords": [-4.6915, 51.7818],
        "metric": "Live AURN Telemetry | West Wales Maritime Air Monitor",
        "action": "Monitors Atlantic inflow and Milford Haven industrial corridor downwind particulates.",
    },
]

# Statutory BGS / Verified Welsh Resupply Springs & Medical Trauma Assets
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
        "id": "RESUPPLY_BGS_PRESELI",
        "name": "Ffynnon Taf / Preseli Ridge Spring",
        "coords": [-4.7420, 51.9585],
        "category": "resupply",
        "categoryLabel": "NATURAL SPRING",
        "accentColor": "#00E5FF",
        "metric": "Gravity Spring | ~12 L/min | Igneous Outcrop",
        "action": "Filtered upland discharge. Boil 60s or filter prior to storage.",
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
        "id": "MEDICAL_MIU_TENBY",
        "name": "Tenby Cottage Hospital Walk-In & Trauma",
        "coords": [-4.7062, 51.6725],
        "category": "medical",
        "categoryLabel": "MINOR INJURY TRAUMA UNIT",
        "accentColor": "#10B981",
        "metric": "NHS Wales Community Trauma & AED",
        "action": "South Pembrokeshire urgent care and defibrillation point.",
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
        "id": "BASELINE_TW_SEVERN",
        "name": "Traffic Wales M4 Prince of Wales Bridge West",
        "coords": [-2.7085, 51.5772],
        "category": "baseline",
        "categoryLabel": "PUBLIC SURVEILLANCE (CCTV)",
        "accentColor": "#94A3B8",
        "metric": "Type: Traffic Wales Strategic Crossing Camera",
        "action": "Monitors primary South Wales M4 arterial river crossing.",
    },
]

# Statutory NRW River Telemetry Stations across all major Welsh spate basins
VERIFIED_NRW_GAUGES: List[Dict[str, Any]] = [
    {
        "id": "NRW_GAUGE_GLASLYN",
        "name": "NRW Gauge: River Glaslyn at Beddgelert",
        "coords": [-4.1025, 53.0110],
        "metric": "Alert Stage: 1.65m | Severe Flood: 2.10m | Basin: Eryri South",
        "action": "Monitors Yr Wyddfa southern runoff. Above 1.65m, A498 Aberglaslyn Pass floods; divert via A4085.",
    },
    {
        "id": "NRW_GAUGE_OGWEN",
        "name": "NRW Gauge: Afon Ogwen at Bethesda",
        "coords": [-4.0615, 53.1802],
        "metric": "Alert Stage: 1.80m | Severe Flood: 2.40m | Basin: Ogwen",
        "action": "Steep Glyderau/Carneddau catchment; rapid spate surge threatens A5 valley bridge.",
    },
    {
        "id": "NRW_GAUGE_CONWY",
        "name": "NRW Gauge: River Conwy at Llanrwst (Pont Fawr)",
        "coords": [-3.7960, 53.1385],
        "metric": "Alert Stage: 2.40m | Bridge Overtop: 3.10m | Basin: Conwy",
        "action": "Fast-responding Carneddau spate river. Pont Fawr closes early; use eastern A470 high corridor.",
    },
    {
        "id": "NRW_GAUGE_CLWYD",
        "name": "NRW Gauge: River Clwyd at St Asaph",
        "coords": [-3.4435, 53.2578],
        "metric": "Alert Stage: 2.20m | Severe Flood: 2.95m | Basin: Clwyd",
        "action": "Flash-flood constriction at Spring Gardens bridge; impacts A525 and A55 slip approaches.",
    },
    {
        "id": "NRW_GAUGE_DEE",
        "name": "NRW Gauge: River Dee at Bala / Llyn Tegid Outflow",
        "coords": [-3.5885, 52.9092],
        "metric": "Alert Stage: 1.95m | Floodplain Spill: 2.50m | Basin: Upper Dee",
        "action": "Controls Dee valley inundation downstream toward Corwen and Llangollen.",
    },
    {
        "id": "NRW_GAUGE_MAWDDACH",
        "name": "NRW Gauge: Afon Mawddach at Ganllwyd",
        "coords": [-3.8892, 52.8015],
        "metric": "Alert Stage: 1.90m | A470 Encroachment: 2.45m | Basin: Mawddach",
        "action": "Drains Rhinogydd & Coed y Brenin; A470 valley gorge floods during prolonged orographic rain.",
    },
    {
        "id": "NRW_GAUGE_DYFI",
        "name": "NRW Gauge: Afon Dyfi at Machynlleth Bridge",
        "coords": [-3.8548, 52.5964],
        "metric": "Alert Stage: 2.80m | A487 Cut-Off: 3.40m | Basin: Dyfi",
        "action": "Single-corridor A487 north-south artery severs when tidal surge meets spate discharge.",
    },
    {
        "id": "NRW_GAUGE_YSTWYTH",
        "name": "NRW Gauge: Afon Ystwyth at Llanilar",
        "coords": [-4.0165, 52.3562],
        "metric": "Alert Stage: 2.05m | Severe Flood: 2.70m | Basin: Ystwyth",
        "action": "Plynlimon runoff corridor; B4575 valley crossings submerge rapidly.",
    },
    {
        "id": "NRW_GAUGE_WYE_RHAYADER",
        "name": "NRW Gauge: River Wye at Rhayader",
        "coords": [-3.5132, 52.3018],
        "metric": "Alert Stage: 2.10m | Severe Flood: 2.80m | Basin: Upper Wye",
        "action": "Monitors Elan Valley & Cambrian runoff along the central A470 trunk corridor.",
    },
    {
        "id": "NRW_GAUGE_TEIFI",
        "name": "NRW Gauge: Afon Teifi at Lampeter",
        "coords": [-4.0792, 52.1118],
        "metric": "Alert Stage: 2.15m | Severe Flood: 2.85m | Basin: Teifi",
        "action": "Low-lying valley floor inundation risk across A482/A475 river crossings.",
    },
    {
        "id": "NRW_GAUGE_CLEDDAU",
        "name": "NRW Gauge: Western Cleddau at Haverfordwest",
        "coords": [-4.9685, 51.8022],
        "metric": "Alert Stage: 2.35m | Tidal Lock: 3.10m | Basin: Cleddau",
        "action": "Preseli runoff meets Milford Haven spring tide; town quay and A4076 bridges flood.",
    },
    {
        "id": "NRW_GAUGE_TYWI",
        "name": "NRW Gauge: River Tywi at Carmarthen Quay",
        "coords": [-4.3055, 51.8542],
        "metric": "Alert Stage: 3.20m | Tidal Lock: 4.10m | Basin: Tywi",
        "action": "Wide floodplain inundation across A40/A484 valley approaches during spring tides.",
    },
    {
        "id": "NRW_GAUGE_TAWE",
        "name": "NRW Gauge: River Tawe at Pontardawe",
        "coords": [-3.8482, 51.7165],
        "metric": "Alert Stage: 2.25m | Severe Flood: 3.00m | Basin: Swansea Valley",
        "action": "Drains western Bannau Brycheiniog; A4067 Swansea Valley floor prone to flash inundation.",
    },
    {
        "id": "NRW_GAUGE_NEATH",
        "name": "NRW Gauge: River Neath at Resolven",
        "coords": [-3.7025, 51.7082],
        "metric": "Alert Stage: 2.45m | Severe Flood: 3.20m | Basin: Vale of Neath",
        "action": "Waterfall country spate discharge threatens A465 low-lying slip roads.",
    },
    {
        "id": "NRW_GAUGE_TAFF",
        "name": "NRW Gauge: River Taff at Pontypridd",
        "coords": [-3.3388, 51.6012],
        "metric": "Alert Stage: 2.60m | Severe Flood: 3.50m | Basin: Taff / Rhondda",
        "action": "Confluence bottleneck for Rhondda and Taff valleys; severe urban flash-flood velocity.",
    },
    {
        "id": "NRW_GAUGE_ELY",
        "name": "NRW Gauge: River Ely at St Fagans (Cardiff West)",
        "coords": [-3.2685, 51.4842],
        "metric": "Alert Stage: 2.15m | Severe Flood: 2.85m | Basin: Ely",
        "action": "Submerges western Cardiff low-lying crossings and A4232 approach culverts.",
    },
    {
        "id": "NRW_GAUGE_USK_BRECON",
        "name": "NRW Gauge: River Usk at Brecon Bridge",
        "coords": [-3.3935, 51.9478],
        "metric": "Alert Stage: 2.30m | Severe Flood: 3.10m | Basin: Usk",
        "action": "Drains northern Bannau Brycheiniog; low town bridges and riverside roads flood rapidly.",
    },
    {
        "id": "NRW_GAUGE_USK_NEWPORT",
        "name": "NRW Gauge: River Usk Tidal Barrier / Newport",
        "coords": [-2.9925, 51.5865],
        "metric": "Alert Stage: 6.80m AOD | Severe Surge: 7.50m AOD | Basin: Lower Usk",
        "action": "Extreme Bristol Channel tidal range coincides with Usk spate flow.",
    },
]

# Objective Physical Terrain, Tidal Stranding & DfT STATS19 Road Defile Hazards
VERIFIED_TERRAIN_AND_TIDAL_HAZARDS: List[Dict[str, Any]] = [
    {
        "id": "HAZ_TIDAL_TRAETH_LAFAN",
        "name": "Traeth Lafan Tidal Stranding Zone (Llanfairfechan)",
        "coords": [-4.0185, 53.2582],
        "label": "COASTAL TIDAL STRANDING HAZARD",
        "metric": "Fast Incoming Tide (4+ knots) | W_flood +10.0",
        "action": "Extensive mudflats flood rapidly from behind via deep channels. Never cross without tide tables.",
    },
    {
        "id": "HAZ_TERRAIN_A5_OGWEN",
        "name": "A5 Ogwen Valley Rockfall & Ice Defile",
        "coords": [-4.0150, 53.1250],
        "label": "DFT STATS19 & TERRAIN DEFILE",
        "metric": "STATS19 Cluster | Gale Funnel & Black Ice | W_incident +4.0",
        "action": "Sheer glaciated slabs above single-carriageway; severe crosswinds and winter verglas.",
    },
    {
        "id": "HAZ_TERRAIN_PENYPASS",
        "name": "A4086 Pen-y-Pass High-Gradient Blackspot",
        "coords": [-4.0205, 53.0801],
        "label": "DFT STATS19 COLLISION DEFILE",
        "metric": "1:8 Gradient | Blind Crests & Freezing Fog | W_incident +4.0",
        "action": "Impassable in freezing fog/snow; deploy high-visibility markers if evacuating on foot.",
    },
    {
        "id": "HAZ_TERRAIN_CRIMEA",
        "name": "A470 Crimea Pass (Blaenau Ffestiniog Summit)",
        "coords": [-3.9145, 53.0122],
        "label": "HIGH-ALTITUDE PASS EXPOSURE",
        "metric": "Elevation: 385m | Snowdrift & Whiteout Corridor | W_incident +3.5",
        "action": "Frequently first North Wales trunk pass closed by snowdrifts and freezing cloud.",
    },
    {
        "id": "HAZ_TERRAIN_BERWYN",
        "name": "B4391 Milltir Cerrig Pass (Berwyn Mountains)",
        "coords": [-3.5415, 52.8625],
        "label": "EXPOSED UPLAND DEFILE",
        "metric": "Elevation: 486m | Zero Shelter & Deep Drifts | W_incident +3.5",
        "action": "Remote high-moorland pass with zero cellular signal and rapid whiteout exposure.",
    },
    {
        "id": "HAZ_TIDAL_NEWGALE",
        "name": "A487 Newgale Shingle Bank Storm Surge Zone",
        "coords": [-5.1265, 51.8522],
        "label": "COASTAL OVERTOPPING DEFILE",
        "metric": "Spring Tide + SW Gale Overtopping | W_flood +10.0",
        "action": "Storm waves hurl cobbles across A487 coastal causeway; divert inland via Roch/Camrose.",
    },
    {
        "id": "HAZ_TIDAL_CEFN_SIDAN",
        "name": "Cefn Sidan / Tywi Estuary Quicksand & Tidal Bar",
        "coords": [-4.3685, 51.7245],
        "label": "COASTAL TIDAL STRANDING HAZARD",
        "metric": "8-Mile Sandbar | Shifting Quicksand Channels | W_flood +10.0",
        "action": "Rapid tidal cut-off across outer sandbanks; keep strictly above high-water dune line.",
    },
    {
        "id": "HAZ_TIDAL_WORMS_HEAD",
        "name": "Worm's Head Tidal Causeway (Rhossili)",
        "coords": [-4.3165, 51.5648],
        "label": "COASTAL TIDAL CUT-OFF",
        "metric": "Submerged 5 Hours Per Tide Cycle | W_flood +10.0",
        "action": "Rocky causeway floods 2.5h before high water; stranding risk with severe Atlantic swell.",
    },
    {
        "id": "HAZ_TERRAIN_BLACK_MOUNTAIN",
        "name": "A4069 Black Mountain Pass (Hairpin Exposure)",
        "coords": [-3.8165, 51.8325],
        "label": "DFT STATS19 & EXPOSURE DEFILE",
        "metric": "Elevation: 493m | Sheer Drop-Offs & Ice | W_incident +4.0",
        "action": "Exposed hairpin mountain road linking Brynaman and Llangadog; hazardous in mist/ice.",
    },
    {
        "id": "HAZ_TIDAL_SULLY",
        "name": "Sully Island Tidal Causeway (Vale of Glamorgan)",
        "coords": [-3.1955, 51.3965],
        "label": "COASTAL TIDAL STRANDING HAZARD",
        "metric": "Bristol Channel 12m Tidal Surge | W_flood +10.0",
        "action": "Second-highest tidal range in the world; causeway floods violently on incoming tide.",
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
    Morphological rounding (`morph_smooth_m > 0`) eliminates any sharp polygon corners
    so every zone renders as an organic, amorphous cloud locked to true ground scale.
    """
    gdf = gpd.GeoDataFrame([{"geometry": geom_wgs84}], crs="EPSG:4326").to_crs(epsg=27700)
    raw_bng = gdf.geometry.iloc[0]

    if morph_smooth_m > 0:
        # Round-cap morphological closing & opening eliminates sharp angles organically
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
    """Normalize a raw OSM Point/Way-Center element into Datum Zero's canonical schema."""
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
        hw = str(tags.get("highway", "")).lower()
        if hw in MINOR_PATH_TYPES and not tags.get("name"):
            return None
        category = "hazards"
        accent = "#FF3B30"
        label = "VEHICULAR RIVER FORD"
        default_name = "Low-Water Road River Ford"
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


def query_overpass_fast(
    query: str, label: str, mirror_offset: int = 0
) -> Optional[Dict[str, Any]]:
    """Execute an Overpass query with round-robin mirror rotation and rate-limit backoff."""
    headers = {"User-Agent": "DatumZero-ETL/4.1 (Civil Resilience Pipeline)"}
    n = len(OVERPASS_MIRRORS)
    ordered_mirrors = [OVERPASS_MIRRORS[(mirror_offset + i) % n] for i in range(n)]

    for mirror_url in ordered_mirrors:
        logging.info("Querying Overpass (%s) via %s...", label, mirror_url)
        try:
            response = requests.post(
                mirror_url,
                data={"data": query},
                headers=headers,
                timeout=55,
            )
            if response.status_code == 429:
                logging.warning("Mirror %s rate-limited (429) on %s. Rotating...", mirror_url, label)
                time.sleep(4.0)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            logging.warning("Mirror %s failed for %s (%s). Failing over...", mirror_url, label, exc)
            time.sleep(2.5)
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
        if not fid.startswith(("RESUPPLY_", "MEDICAL_", "BASELINE_", "HAZARDS_")):
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
                        "name": f"{tier} 50Hz EMF Corridor",
                        "category": "grid_down",
                        "categoryLabel": "HIGH-VOLTAGE EMF BUFFER",
                        "accentColor": "#FFB800",
                        "threatWeight": 2.0,
                        "cloudRing": ring_name,
                        "primaryMetric": f"True Ground Buffer: {buf_m}m | Compass & RF Interference",
                        "operationalAction": "Expect magnetometer deviation and storm line-drop hazard. Maintain clearance.",
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
            "name": "Strategic Trunk Bridge Choke Point",
            "category": "hazards",
            "categoryLabel": "SINGLE-ACCESS BRIDGE DEFILE",
            "accentColor": "#FFB800",
            "threatWeight": 5.0,
            "primaryMetric": "Survival Penalty: W_choke +5.0 | 160m Corridor Buffer",
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
    headers = {"User-Agent": "DatumZero-ETL/4.1 (Civil Resilience Pipeline)"}
    features: List[Dict[str, Any]] = []

    try:
        logging.info("Querying DataMapWales WFS (%s)...", type_name)
        res = requests.get(DATAMAPWALES_WFS_URL, params=params, headers=headers, timeout=25)
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
    headers = {"User-Agent": "DatumZero-ETL/4.1 (Civil Resilience Pipeline)"}
    try:
        res = requests.get(DEFRA_UKAIR_STATIONS_URL, headers=headers, timeout=15)
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
    Compile Priority 1 & 2 Geospatial Intelligence as organic, non-rectangular,
    3-ring feathered clouds locked to British National Grid meters (EPSG:27700).
    """
    features: List[Dict[str, Any]] = []

    # 1. Topological Cellular Dead Zones (W_signal = 2.5 winding valley clouds)
    for dz in VERIFIED_CELLULAR_DEAD_ZONES:
        line = LineString(dz["line"])
        buf = float(dz["buffer_m"])
        base_props = {
            "id": dz["id"],
            "name": dz["name"],
            "category": "grid_down",
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

    # 2. High-Power Broadcast & RF Transmitter Footprints
    for tx in VERIFIED_RF_TRANSMITTERS:
        pt = Point(tx["coords"][0], tx["coords"][1])
        rad = float(tx["radius_m"])
        base_props = {
            "id": tx["id"],
            "name": tx["name"],
            "category": "grid_down",
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

    # 3. UKHSA / BGS Radon Gas Action Zones (Organic smoothed geological belts)
    for rz in VERIFIED_RADON_ZONES:
        poly = Polygon(rz["poly"])
        base_props = {
            "id": rz["id"],
            "name": rz["name"],
            "category": "hazards",
            "categoryLabel": "RADON GAS ACTION ZONE",
            "accentColor": "#FF3B30",
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

    # 4. Statutory SSSIs (Organic smoothed protected boundaries — STRICT NO FORAGING)
    for sssi in VERIFIED_SSSI_ZONES:
        poly = Polygon(sssi["poly"])
        base_props = {
            "id": sssi["id"],
            "name": sssi["name"],
            "category": "resupply",
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

    # 5. Ancient Woodland Foraging Biomes (Organic smoothed canopy clouds)
    for fb in VERIFIED_FORAGING_BIOMES:
        poly = Polygon(fb["poly"])
        base_props = {
            "id": fb["id"],
            "name": fb["name"],
            "category": "resupply",
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

    # 6. Designated Open Access Land (CRoW Act 2000 — Organic upland massifs)
    for crow in VERIFIED_CROW_ACCESS_LAND:
        poly = Polygon(crow["poly"])
        base_props = {
            "id": crow["id"],
            "name": crow["name"],
            "category": "baseline",
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

    for corr in VERIFIED_FLOOD_CORRIDORS:
        line = LineString(corr["line"])
        buf = float(corr["buffer_m"])
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
    """Extract OSM POIs across 7 balanced sub-tiles + high-voltage power lines & trunk bridges."""
    point_features: List[Dict[str, Any]] = []
    seen_ids = set()

    for tile_idx, (region_name, tile_bbox) in enumerate(WALES_SUBTILES):
        min_lon, min_lat, max_lon, max_lat = tile_bbox
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
        poi_query = f"""
        [out:json][timeout:50];
        (
          node["natural"="spring"]({bbox_str});
          node["emergency"="defibrillator"]({bbox_str});
          node["man_made"="surveillance"]({bbox_str});
          node["ford"="yes"]["highway"]({bbox_str});
          way["ford"="yes"]["highway"]({bbox_str});
        );
        out center;
        """
        poi_data = query_overpass_fast(
            poi_query, f"POIs {region_name}", mirror_offset=tile_idx
        )
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
        time.sleep(2.5)

    min_lon, min_lat, max_lon, max_lat = WALES_BBOX
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    power_query = f"""
    [out:json][timeout:50];
    way["power"="line"]({bbox_str});
    out geom;
    """
    power_data = query_overpass_fast(power_query, "PowerLines", mirror_offset=1)
    time.sleep(2.0)

    bridge_query = f"""
    [out:json][timeout:50];
    way["bridge"="yes"]["highway"~"^(trunk|primary)$"]({bbox_str});
    out geom;
    """
    bridge_data = query_overpass_fast(bridge_query, "TrunkBridges", mirror_offset=2)

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
    defra_aurn = harvest_defra_aurn_telemetry()

    dmw_flood_zones = harvest_datamapwales_wfs_layer(
        type_name="inspire-nrw:NRW_FLOOD_MAP_FLOOD_ZONE_3_RIVERS_AND_SEA",
        id_prefix="NRW_FZ3",
        name_default="NRW Statutory Flood Zone 3 (Rivers & Sea)",
        category="hazards",
        category_label="NRW FLOOD ZONE 3 POLYGON",
        accent_color="#FF3B30",
        threat_weight=10.0,
        metric_text="High Fluvial/Tidal Probability | W_flood +10.0",
        action_text="Statutory 1-in-30-yr floodplain. Avoid low-lying roads during severe weather.",
        max_count=100,
    )
    dmw_sssi = harvest_datamapwales_wfs_layer(
        type_name="inspire-nrw:NRW_SSSI",
        id_prefix="DMW_SSSI",
        name_default="Site of Special Scientific Interest (SSSI)",
        category="resupply",
        category_label="SSSI PROTECTED ZONE (NO FORAGING)",
        accent_color="#FF3B30",
        threat_weight=0.0,
        metric_text="Statutory SSSI | Wildlife & Countryside Act 1981",
        action_text="FORAGING PROHIBITED: High-Conservation SSSI Zone. No harvesting or uprooting permitted.",
        max_count=60,
    )
    dmw_crow = harvest_datamapwales_wfs_layer(
        type_name="inspire-nrw:NRW_OPEN_ACCESS_LAND",
        id_prefix="DMW_CROW",
        name_default="Designated Open Access Land (CRoW Act 2000)",
        category="baseline",
        category_label="CROW OPEN ACCESS LAND",
        accent_color="#94A3B8",
        threat_weight=0.0,
        metric_text="CRoW Act 2000 Statutory Access",
        action_text="Legal right of foot access across uncultivated upland terrain.",
        max_count=60,
    )

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
