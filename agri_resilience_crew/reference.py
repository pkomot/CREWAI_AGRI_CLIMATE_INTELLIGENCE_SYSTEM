"""Static reference data: counties, terminal markets, crop agronomy and economics.

Coordinates are approximate county headquarters. Agronomic ranges follow widely
published Kenyan extension guidance (KALRO / MoALD factsheets) and are meant as
planning defaults, not field prescriptions. Economic figures (yield, cost of
production, farmgate share) are indicative planning assumptions for 2026 and
should be replaced with county extension data where available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Counties (all 47, IEBC order) -> (lat, lon) of county HQ
# --------------------------------------------------------------------------- #
COUNTY_COORDS: dict[str, tuple[float, float]] = {
    "Mombasa": (-4.04, 39.67),
    "Kwale": (-4.17, 39.45),
    "Kilifi": (-3.63, 39.85),
    "Tana River": (-1.50, 40.03),
    "Lamu": (-2.27, 40.90),
    "Taita Taveta": (-3.40, 38.56),
    "Garissa": (-0.45, 39.65),
    "Wajir": (1.75, 40.06),
    "Mandera": (3.94, 41.86),
    "Marsabit": (2.33, 37.99),
    "Isiolo": (0.35, 37.58),
    "Meru": (0.05, 37.65),
    "Tharaka-Nithi": (-0.30, 37.88),
    "Embu": (-0.53, 37.45),
    "Kitui": (-1.37, 38.01),
    "Machakos": (-1.52, 37.26),
    "Makueni": (-1.80, 37.62),
    "Nyandarua": (-0.27, 36.38),
    "Nyeri": (-0.42, 36.95),
    "Kirinyaga": (-0.50, 37.28),
    "Murang'a": (-0.72, 37.15),
    "Kiambu": (-1.17, 36.83),
    "Turkana": (3.12, 35.60),
    "West Pokot": (1.24, 35.12),
    "Samburu": (1.10, 36.70),
    "Trans Nzoia": (1.02, 35.00),
    "Uasin Gishu": (0.51, 35.27),
    "Elgeyo-Marakwet": (0.67, 35.51),
    "Nandi": (0.20, 35.10),
    "Baringo": (0.47, 35.97),
    "Laikipia": (0.02, 37.07),
    "Nakuru": (-0.30, 36.07),
    "Narok": (-1.08, 35.87),
    "Kajiado": (-1.85, 36.78),
    "Kericho": (-0.37, 35.28),
    "Bomet": (-0.78, 35.34),
    "Kakamega": (0.28, 34.75),
    "Vihiga": (0.07, 34.72),
    "Bungoma": (0.56, 34.56),
    "Busia": (0.46, 34.11),
    "Siaya": (0.06, 34.29),
    "Kisumu": (-0.09, 34.77),
    "Homa Bay": (-0.53, 34.46),
    "Migori": (-1.06, 34.47),
    "Kisii": (-0.68, 34.77),
    "Nyamira": (-0.56, 34.94),
    "Nairobi": (-1.29, 36.82),
}

COUNTIES: tuple[str, ...] = tuple(COUNTY_COORDS)

ASAL_COUNTIES: frozenset[str] = frozenset(
    {
        "Turkana", "Marsabit", "Mandera", "Wajir", "Garissa", "Isiolo", "Samburu",
        "Tana River", "Baringo", "West Pokot", "Kitui", "Makueni", "Kajiado", "Lamu",
        "Kilifi", "Kwale", "Taita Taveta", "Laikipia", "Narok",
    }
)  # fmt: skip

DESERT_LOCUST_FRONTLINE: frozenset[str] = frozenset(
    {"Mandera", "Wajir", "Marsabit", "Turkana", "Samburu", "Isiolo", "Garissa"}
)

SUB_COUNTIES: dict[str, tuple[str, ...]] = {
    "Nakuru": (
        "Molo", "Njoro", "Kuresoi North", "Kuresoi South", "Rongai", "Subukia",
        "Bahati", "Gilgil", "Naivasha", "Nakuru Town East", "Nakuru Town West",
    ),
    "Uasin Gishu": ("Soy", "Turbo", "Moiben", "Ainabkoi", "Kapseret", "Kesses"),
    "Meru": (
        "Imenti North", "Imenti South", "Imenti Central", "Buuri", "Tigania East",
        "Tigania West", "Igembe North", "Igembe Central", "Igembe South",
    ),
    "Nyandarua": ("Kinangop", "Kipipiri", "Ol Kalou", "Ol Joro Orok", "Ndaragwa"),
    "Trans Nzoia": ("Kwanza", "Endebess", "Saboti", "Kiminini", "Cherangany"),
}  # fmt: skip


def canonical_county(name: str) -> str | None:
    """Case/spacing/hyphen-insensitive lookup of a county name."""
    key = _norm(name)
    for county in COUNTIES:
        if _norm(county) == key:
            return county
    return None


def _norm(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


# --------------------------------------------------------------------------- #
# Terminal wholesale markets tracked for logistics arbitrage
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Market:
    name: str
    county: str
    lat: float
    lon: float
    # Indicative county cess + market entry + handling, KES per kg.
    fees_kes_per_kg: float


MARKETS: dict[str, Market] = {
    "Nairobi Wakulima": Market("Nairobi Wakulima", "Nairobi", -1.285, 36.830, 2.5),
    "Mombasa Kongowea": Market("Mombasa Kongowea", "Mombasa", -4.030, 39.680, 3.0),
    "Kisumu Kibuye": Market("Kisumu Kibuye", "Kisumu", -0.100, 34.760, 2.0),
    "Eldoret Main": Market("Eldoret Main", "Uasin Gishu", 0.520, 35.270, 2.0),
}

ROAD_FACTOR = 1.3  # straight-line -> road distance multiplier for Kenyan trunk roads


def road_distance_km(county: str, market: str) -> float:
    """Approximate road distance from a county HQ to a terminal market."""
    lat1, lon1 = COUNTY_COORDS[county]
    m = MARKETS[market]
    return round(_haversine_km(lat1, lon1, m.lat, m.lon) * ROAD_FACTOR, 1)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Crops
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CropProfile:
    name: str
    aliases: tuple[str, ...]
    ph_min: float
    ph_max: float
    rain_min_mm: float  # seasonal water requirement (rain-fed)
    rain_max_mm: float
    alt_min_m: float
    alt_max_m: float
    days_to_maturity: int
    yield_t_per_ha: float  # attainable rain-fed yield with good practice
    cost_kes_per_ha: float  # indicative cost of production
    farmgate_share: float  # farmgate / Nairobi wholesale price ratio
    base_price_kes_per_kg: float  # Nairobi wholesale reference level
    perishability: float  # fraction lost per 100 km transport (spoilage, bruising)
    waterlogging_sensitive: bool
    key_pests: tuple[str, ...]


CROPS: dict[str, CropProfile] = {
    "Irish Potatoes": CropProfile(
        "Irish Potatoes", ("potato", "potatoes", "irish potato", "waru"),
        5.0, 6.5, 500, 700, 1500, 3000, 105, 18.0, 320_000, 0.55, 48.0, 0.006, True,
        ("late blight", "bacterial wilt", "potato cyst nematode"),
    ),
    "Maize": CropProfile(
        "Maize", ("maize", "corn", "mahindi"),
        5.5, 7.0, 500, 800, 900, 2400, 150, 4.5, 110_000, 0.70, 52.0, 0.001, False,
        ("fall armyworm", "maize lethal necrosis", "stalk borer"),
    ),
    "Tomatoes": CropProfile(
        "Tomatoes", ("tomato", "tomatoes", "nyanya"),
        6.0, 7.0, 400, 600, 0, 2000, 85, 30.0, 520_000, 0.50, 75.0, 0.015, True,
        ("Tuta absoluta", "bacterial wilt", "early blight"),
    ),
    "Onions": CropProfile(
        "Onions", ("onion", "onions", "dry onion", "kitunguu"),
        6.0, 7.0, 350, 550, 0, 2000, 120, 20.0, 340_000, 0.55, 90.0, 0.004, True,
        ("onion thrips", "purple blotch", "downy mildew"),
    ),
    "Beans": CropProfile(
        "Beans", ("bean", "beans", "dry beans", "maharagwe"),
        6.0, 7.5, 300, 500, 900, 2200, 80, 1.4, 70_000, 0.72, 135.0, 0.001, True,
        ("bean fly", "anthracnose", "angular leaf spot"),
    ),
    "Cabbages": CropProfile(
        "Cabbages", ("cabbage", "cabbages", "kabichi"),
        6.0, 6.5, 380, 500, 800, 2500, 100, 40.0, 260_000, 0.45, 25.0, 0.010, False,
        ("diamondback moth", "black rot", "aphids"),
    ),
}  # fmt: skip


def canonical_crop(name: str) -> str | None:
    key = _norm(name)
    for crop, profile in CROPS.items():
        if _norm(crop) == key or any(_norm(a) == key for a in profile.aliases):
            return crop
    return None


# Lean-season months (price peaks) by crop, used for seasonal shape of synthetic series.
PRICE_PEAK_MONTH: dict[str, int] = {
    "Irish Potatoes": 4,
    "Maize": 6,
    "Tomatoes": 3,
    "Onions": 5,
    "Beans": 6,
    "Cabbages": 3,
}

# Transport tariff for a 10 t truck on Kenyan trunk roads, KES per tonne-km (2026 indicative).
TRANSPORT_KES_PER_TONNE_KM = 9.5
