"""Deterministic synthetic datasets that stand in for KAMIS, KMD and KALRO in mock mode.

KAMIS, KMD and KALRO do not publish stable public JSON APIs, so offline runs, CI
and tests use these generators. Every output is seeded from its query key, which
makes it reproducible. The shapes are realistic: seasonal price cycles, bimodal
and unimodal rainfall regimes, and acidic western soils. None of the numbers are
observations, and every result built on them carries ``synthetic=True`` in its
provenance.
"""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import math
import random

from agri_resilience_crew.reference import (
    ASAL_COUNTIES,
    COUNTIES,
    CROPS,
    DESERT_LOCUST_FRONTLINE,
    PRICE_PEAK_MONTH,
)


def seeded_rng(*key: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# --------------------------------------------------------------------------- #
# Agro-climatic zones and monthly rainfall normals (mm, Jan..Dec)
# --------------------------------------------------------------------------- #
ZONE_RAINFALL: dict[str, tuple[float, ...]] = {
    "western_highlands": (30, 45, 85, 160, 140, 90, 120, 140, 90, 70, 60, 35),
    "central_rift": (30, 40, 70, 140, 110, 60, 70, 80, 50, 50, 60, 40),
    "central_eastern_highlands": (35, 30, 110, 280, 120, 25, 20, 25, 30, 160, 200, 70),
    "lake_basin": (60, 70, 130, 200, 160, 70, 60, 80, 80, 90, 110, 80),
    "eastern_semi_arid": (30, 25, 80, 140, 40, 5, 5, 5, 10, 60, 150, 70),
    "northern_arid": (10, 10, 30, 70, 30, 5, 5, 5, 5, 25, 50, 15),
    "coast": (30, 15, 50, 180, 240, 110, 80, 70, 70, 100, 110, 70),
    "nairobi_metro": (50, 40, 90, 190, 130, 30, 15, 20, 25, 60, 150, 90),
}

_ZONE_MEMBERS: dict[str, tuple[str, ...]] = {
    "western_highlands": (
        "Uasin Gishu", "Trans Nzoia", "Nandi", "Kakamega", "Bungoma", "Vihiga",
        "Kericho", "Bomet", "Kisii", "Nyamira", "Elgeyo-Marakwet",
    ),
    "central_rift": ("Nakuru", "Nyandarua", "Narok", "Laikipia", "Baringo"),
    "central_eastern_highlands": (
        "Meru", "Nyeri", "Kirinyaga", "Murang'a", "Kiambu", "Embu", "Tharaka-Nithi",
    ),
    "lake_basin": ("Kisumu", "Siaya", "Homa Bay", "Migori", "Busia"),
    "eastern_semi_arid": (
        "Machakos", "Makueni", "Kitui", "Kajiado", "Taita Taveta", "Isiolo",
    ),
    "northern_arid": (
        "Turkana", "Marsabit", "Mandera", "Wajir", "Garissa", "Samburu", "West Pokot",
        "Tana River",
    ),
    "coast": ("Mombasa", "Kwale", "Kilifi", "Lamu"),
    "nairobi_metro": ("Nairobi",),
}  # fmt: skip

COUNTY_ZONE: dict[str, str] = {c: z for z, members in _ZONE_MEMBERS.items() for c in members}
assert set(COUNTY_ZONE) == set(COUNTIES), "every county needs an agro-climatic zone"

COUNTY_ALTITUDE_M: dict[str, float] = {
    "Mombasa": 50, "Kwale": 400, "Kilifi": 200, "Tana River": 100, "Lamu": 50,
    "Taita Taveta": 900, "Garissa": 150, "Wajir": 250, "Mandera": 300, "Marsabit": 1300,
    "Isiolo": 1100, "Meru": 1600, "Tharaka-Nithi": 1300, "Embu": 1400, "Kitui": 1100,
    "Machakos": 1600, "Makueni": 1100, "Nyandarua": 2400, "Nyeri": 1800, "Kirinyaga": 1400,
    "Murang'a": 1500, "Kiambu": 1700, "Turkana": 500, "West Pokot": 1500, "Samburu": 1500,
    "Trans Nzoia": 1900, "Uasin Gishu": 2100, "Elgeyo-Marakwet": 2300, "Nandi": 1900,
    "Baringo": 1300, "Laikipia": 1900, "Nakuru": 1900, "Narok": 1900, "Kajiado": 1600,
    "Kericho": 2000, "Bomet": 1900, "Kakamega": 1500, "Vihiga": 1600, "Bungoma": 1500,
    "Busia": 1200, "Siaya": 1200, "Kisumu": 1150, "Homa Bay": 1200, "Migori": 1300,
    "Kisii": 1700, "Nyamira": 1800, "Nairobi": 1700,
}  # fmt: skip

SUB_COUNTY_ALTITUDE_M: dict[str, float] = {
    "Molo": 2500, "Kuresoi North": 2400, "Kuresoi South": 2300, "Njoro": 2200,
    "Naivasha": 1900, "Gilgil": 2000, "Rongai": 1800, "Subukia": 1900,
    "Kinangop": 2600, "Ol Joro Orok": 2400, "Ainabkoi": 2400, "Moiben": 2200,
    "Buuri": 2000, "Imenti North": 1600, "Imenti South": 1500, "Igembe South": 1300,
    "Endebess": 1900, "Cherangany": 2100,
}  # fmt: skip

# Orographic adjustment relative to the zone normal (windward highlands wetter, leeward drier).
SUB_COUNTY_RAIN_FACTOR: dict[str, float] = {
    "Molo": 1.5, "Kuresoi North": 1.5, "Kuresoi South": 1.45, "Njoro": 1.25, "Naivasha": 0.8,
    "Gilgil": 0.8, "Rongai": 0.9, "Kinangop": 1.3, "Ol Joro Orok": 1.1, "Ndaragwa": 0.85,
    "Ainabkoi": 1.2, "Moiben": 1.1, "Buuri": 0.6, "Imenti North": 1.3, "Imenti South": 1.2,
    "Igembe South": 1.1, "Cherangany": 1.2,
}  # fmt: skip

CV_SEASONAL_RAIN = 0.30  # typical coefficient of variation used for SPI


def season_label(start: dt.date) -> str:
    month = start.month
    if month in (3, 4, 5):
        name = "MAM Long Rains"
    elif month in (6, 7, 8):
        name = "JJA Season"
    elif month in (10, 11, 12):
        name = "OND Short Rains"
    elif month == 9:
        name = "SON Transition / Short Rains"
    else:
        name = "JF Dry Season"
    return f"{name} {start.year}"


def monthly_normals(county: str, sub_county: str | None = None) -> tuple[float, ...]:
    factor = SUB_COUNTY_RAIN_FACTOR.get(sub_county or "", 1.0)
    return tuple(m * factor for m in ZONE_RAINFALL[COUNTY_ZONE[county]])


def normal_rainfall_mm(county: str, start: dt.date, days: int, sub_county: str | None = None) -> float:
    """Climatological rainfall for a date window, pro-rated by day."""
    monthly = monthly_normals(county, sub_county)
    total = 0.0
    for offset in range(days):
        day = start + dt.timedelta(days=offset)
        days_in_month = calendar.monthrange(day.year, day.month)[1]
        total += monthly[day.month - 1] / days_in_month
    return round(total, 1)


def tercile_forecast(county: str, start: dt.date) -> tuple[float, float, float]:
    """Synthetic probabilistic tercile forecast (above, near, below normal).

    A season-wide driver (an ENSO/IOD-like signal shared by all counties) is
    combined with a small county-level perturbation.
    """
    season = season_label(start)
    driver = seeded_rng("driver", season).uniform(-0.18, 0.18)
    local = seeded_rng("local", county, season).uniform(-0.06, 0.06)
    shift = driver + local
    above = min(0.7, max(0.1, 0.333 + shift))
    below = min(0.7, max(0.1, 0.333 - shift))
    near = max(0.1, 1.0 - above - below)
    total = above + near + below
    a, b = round(above / total, 3), round(below / total, 3)
    return a, round(1.0 - a - b, 3), b


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
MARKET_PRICE_FACTOR = {
    "Nairobi Wakulima": 1.00,
    "Mombasa Kongowea": 1.12,
    "Kisumu Kibuye": 1.05,
    "Eldoret Main": 0.92,
}

_DAILY_SIGMA = {
    "Tomatoes": 0.040,
    "Cabbages": 0.035,
    "Irish Potatoes": 0.025,
    "Onions": 0.020,
    "Maize": 0.010,
    "Beans": 0.008,
}
_SEASONAL_AMPLITUDE = {
    "Tomatoes": 0.28,
    "Cabbages": 0.25,
    "Irish Potatoes": 0.22,
    "Onions": 0.18,
    "Maize": 0.15,
    "Beans": 0.10,
}


def price_series(crop: str, market: str, end: dt.date, days: int) -> list[tuple[dt.date, float]]:
    """Daily wholesale price series (KES/kg) ending at ``end``.

    The series is generated from a fixed anchor so overlapping windows agree.
    """
    profile = CROPS[crop]
    anchor = dt.date(end.year - 1, 1, 1)
    total_days = (end - anchor).days + 1
    rng = seeded_rng("price", crop, market, anchor.year)
    sigma = _DAILY_SIGMA[crop]
    amp = _SEASONAL_AMPLITUDE[crop]
    peak_doy = dt.date(2001, PRICE_PEAK_MONTH[crop], 15).timetuple().tm_yday
    base = profile.base_price_kes_per_kg * MARKET_PRICE_FACTOR[market]

    deviation = 0.0
    out: list[tuple[dt.date, float]] = []
    for i in range(total_days):
        day = anchor + dt.timedelta(days=i)
        seasonal = 1 + amp * math.cos(2 * math.pi * (day.timetuple().tm_yday - peak_doy) / 365.25)
        # AR(1) mean-reverting log deviation (phi=0.94) with Gaussian shocks.
        deviation = 0.94 * deviation + rng.gauss(0.0, sigma)
        out.append((day, round(base * seasonal * math.exp(deviation), 2)))
    return out[-days:]


def origin_price_adjustment(county: str, crop: str) -> float:
    """Small county-level farmgate premium/discount (-8%..+8%)."""
    return seeded_rng("farmgate", county, crop).uniform(-0.08, 0.08)


# --------------------------------------------------------------------------- #
# Soils
# --------------------------------------------------------------------------- #
_ZONE_SOIL: dict[str, tuple[float, str, float, str]] = {
    # zone: (pH, texture, organic carbon %, drainage)
    "western_highlands": (5.3, "clay loam", 2.1, "good"),
    "central_rift": (6.1, "sandy loam", 1.8, "good"),
    "central_eastern_highlands": (5.4, "clay", 2.6, "good"),
    "lake_basin": (6.5, "clay", 1.2, "poor"),
    "eastern_semi_arid": (7.0, "sandy loam", 0.6, "excessive"),
    "northern_arid": (7.8, "sand", 0.3, "excessive"),
    "coast": (6.4, "sandy loam", 0.8, "good"),
    "nairobi_metro": (6.0, "clay loam", 1.5, "moderate"),
}

TEXTURE_MOISTURE: dict[str, tuple[float, float]] = {
    # texture: (field capacity %, permanent wilting point %) volumetric
    "sand": (12.0, 5.0),
    "sandy loam": (18.0, 8.0),
    "loam": (25.0, 12.0),
    "clay loam": (32.0, 18.0),
    "clay": (40.0, 25.0),
}


def soil_profile(county: str, sub_county: str | None) -> dict[str, float | str]:
    zone = COUNTY_ZONE[county]
    ph, texture, oc, drainage = _ZONE_SOIL[zone]
    rng = seeded_rng("soil", county, sub_county or "-")
    altitude = SUB_COUNTY_ALTITUDE_M.get(sub_county or "", COUNTY_ALTITUDE_M[county])
    return {
        "soil_ph": round(ph + rng.uniform(-0.3, 0.3), 2),
        "soil_texture": texture,
        "organic_carbon_pct": round(max(0.1, oc + rng.uniform(-0.3, 0.3)), 2),
        "drainage": drainage,
        "altitude_m": altitude,
    }


# --------------------------------------------------------------------------- #
# Zoning and pests (illustrative screening layers)
# --------------------------------------------------------------------------- #
UNIVERSAL_ZONING = (
    (
        "WRA",
        "Riparian reserve",
        "No cultivation or structures on riparian land (6 m minimum, up to 30 m from the "
        "highest-ever water mark depending on watercourse width); abstraction needs a WRA permit.",
        "all farms bordering rivers, streams or lakes",
        "restricted",
    ),
    (
        "NEMA",
        "Wetlands and river banks",
        "EMCA (Wetlands, River Banks, Lake Shores and Sea Shore Management) Regulations, 2009: "
        "conversion or drainage of wetlands needs NEMA approval and may need an EIA.",
        "valley-bottom and swamp plots",
        "restricted",
    ),
)

SITE_ZONING: dict[str, tuple[tuple[str, str, str, str, str], ...]] = {
    "Naivasha": (
        (
            "NEMA",
            "Lake Naivasha basin (Ramsar site)",
            "Irrigated horticulture and abstraction are subject to WRA permits and the basin "
            "management plan; agrochemical runoff controls apply.",
            "irrigated plots within the lake basin",
            "restricted",
        ),
    ),
    "Molo": (("KFS", "Mau Forest Complex", "No cultivation inside gazetted forest boundaries.", "forest-edge plots", "prohibited"),),
    "Kuresoi North": (("KFS", "Mau Forest Complex", "No cultivation inside gazetted forest boundaries.", "forest-edge plots", "prohibited"),),
    "Kuresoi South": (("KFS", "Mau Forest Complex", "No cultivation inside gazetted forest boundaries.", "forest-edge plots", "prohibited"),),
    "Njoro": (("KFS", "Eastern Mau forest block", "No cultivation inside gazetted forest boundaries.", "forest-edge plots", "prohibited"),),
    "Buuri": (("KFS", "Mt Kenya Forest Reserve", "No cultivation inside the reserve; PELIS plots only under KFS licence.", "forest-edge plots", "prohibited"),),
    "Imenti North": (("KFS", "Imenti Forest", "No cultivation inside gazetted forest boundaries.", "forest-edge plots", "prohibited"),),
    "Kinangop": (("County", "Steep slopes / Aberdare foothills", "Terracing and contour farming expected on steep slopes.", "plots on slopes", "advisory"),),
}  # fmt: skip

PEST_ACTIONS: dict[str, str] = {
    "late blight": "Use certified seed of tolerant varieties; begin a protective fungicide programme at canopy closure and tighten intervals in wet spells.",
    "bacterial wilt": "Rotate out of Solanaceae for 3+ seasons, use certified seed, rogue and destroy wilting plants.",
    "potato cyst nematode": "Use certified seed only; rotate with non-host crops; do not move soil from infested fields.",
    "fall armyworm": "Scout weekly from emergence; act at 10-20% damaged plants with PCPB-registered products at early instar; consider push-pull.",
    "maize lethal necrosis": "Plant certified MLN-tolerant seed; avoid maize-after-maize; control thrips and aphid vectors.",
    "stalk borer": "Early planting, destroy crop residues, apply registered granules into the funnel when damage appears.",
    "Tuta absoluta": "Use pheromone traps and insect netting; rotate registered active ingredients; remove infested leaves.",
    "early blight": "Mulch, stake and prune for airflow; protective fungicide in humid weather.",
    "onion thrips": "Monitor with blue sticky traps; overhead irrigation suppresses populations; rotate actives.",
    "purple blotch": "Wider spacing, avoid overhead irrigation late in the day, protective fungicide.",
    "downy mildew": "Avoid planting in low-lying humid sites; protective fungicide during cool wet weather.",
    "bean fly": "Seed dressing with registered insecticide; earth up to encourage adventitious roots.",
    "anthracnose": "Use clean certified seed; avoid working in wet fields; rotate.",
    "angular leaf spot": "Clean seed, resistant varieties, crop rotation.",
    "diamondback moth": "Scout twice weekly; use Bt-based products; rotate modes of action.",
    "black rot": "Hot-water treated seed; rotate away from brassicas for 2+ seasons.",
    "aphids": "Conserve natural enemies; spot-spray registered products when colonies build up.",
    "desert locust": "Report swarms to the county agriculture office / FAO DLIS; do not self-spray swarms.",
}  # fmt: skip

_WET_FAVOURED = {"late blight", "bacterial wilt", "early blight", "purple blotch", "downy mildew", "anthracnose", "angular leaf spot", "black rot"}  # fmt: skip
_DRY_FAVOURED = {"fall armyworm", "onion thrips", "Tuta absoluta", "aphids", "diamondback moth", "stalk borer", "bean fly"}  # fmt: skip


def pest_risk(pest: str, county: str, rainfall_anomaly_pct: float) -> str:
    if pest in _WET_FAVOURED:
        return "high" if rainfall_anomaly_pct >= 10 else "moderate" if rainfall_anomaly_pct > -15 else "low"
    if pest in _DRY_FAVOURED:
        return "high" if rainfall_anomaly_pct <= -15 else "moderate"
    if pest == "desert locust":
        return "moderate" if county in DESERT_LOCUST_FRONTLINE else "low"
    return "moderate"


def is_asal(county: str) -> bool:
    return county in ASAL_COUNTIES
