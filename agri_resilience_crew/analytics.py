"""Deterministic analytics. The LLM interprets these numbers but never computes them."""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections.abc import Sequence

from agri_resilience_crew.reference import (
    CROPS,
    MARKETS,
    TRANSPORT_KES_PER_TONNE_KM,
    road_distance_km,
)
from agri_resilience_crew.schemas import MarginResult, MarketPriceSignal, RiskLevel

MIN_FEASIBLE_MARGIN_PCT = 10.0


def ols_slope(values: Sequence[float]) -> float:
    """Least-squares slope per step; 0 for fewer than two points."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = statistics.fmean(values)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den


def pct_change(new: float, old: float) -> float:
    return 0.0 if old == 0 else (new - old) / old * 100


def price_signal(
    *,
    crop: str,
    market: str,
    origin_county: str,
    prices: Sequence[float],
    farmgate_price: float,
) -> MarketPriceSignal:
    """Velocity, volatility and haul-to-market arbitrage for one crop/market pair.

    ``net_margin_kes_per_kg`` is the gain from hauling produce to ``market``
    instead of selling at the farmgate, after transport, fees and spoilage.
    """
    if len(prices) < 30:
        raise ValueError("price_signal needs at least 30 daily observations")
    profile = CROPS[crop]
    latest = prices[-1]
    last30 = prices[-30:]
    log_returns = [math.log(b / a) for a, b in zip(last30, last30[1:], strict=False) if a > 0]
    volatility = statistics.pstdev(log_returns) * math.sqrt(30) * 100 if log_returns else 0.0
    sd = statistics.pstdev(prices)
    zscore = 0.0 if sd == 0 else (latest - statistics.fmean(prices)) / sd

    distance = road_distance_km(origin_county, market)
    transport = TRANSPORT_KES_PER_TONNE_KM * distance / 1000
    fees = MARKETS[market].fees_kes_per_kg
    spoilage = latest * profile.perishability * distance / 100
    net = latest - farmgate_price - transport - fees - spoilage
    net_pct = pct_change(farmgate_price + net, farmgate_price)

    return MarketPriceSignal(
        market=market,
        latest_price_kes_per_kg=round(latest, 2),
        mean_7d=round(statistics.fmean(prices[-7:]), 2),
        mean_30d=round(statistics.fmean(last30), 2),
        change_7d_pct=round(pct_change(latest, prices[-8]), 2),
        change_30d_pct=round(pct_change(latest, prices[-31] if len(prices) > 30 else prices[0]), 2),
        velocity_kes_per_kg_per_day=round(ols_slope(last30), 3),
        acceleration_kes_per_kg_per_day2=round(
            (ols_slope(prices[-14:]) - ols_slope(prices[-28:-14])) / 14, 4
        ),
        volatility_30d_pct=round(volatility, 2),
        zscore_vs_lookback=round(zscore, 2),
        road_distance_km=distance,
        transport_kes_per_kg=round(transport, 2),
        fees_kes_per_kg=fees,
        spoilage_kes_per_kg=round(spoilage, 2),
        farmgate_price_kes_per_kg=round(farmgate_price, 2),
        net_margin_kes_per_kg=round(net, 2),
        net_margin_pct=round(net_pct, 2),
        feasible=net_pct >= MIN_FEASIBLE_MARGIN_PCT,
    )


def trend_label(velocity: float, latest_price: float) -> str:
    """Classify a 30-day slope: >0.2%/day of price is rising, <-0.2%/day falling."""
    rel = 0.0 if latest_price == 0 else velocity / latest_price * 100
    return "rising" if rel > 0.2 else "falling" if rel < -0.2 else "flat"


def gross_margin(
    crop: str,
    *,
    yield_t_per_ha: float,
    price_kes_per_kg: float,
    area_ha: float = 1.0,
    cost_kes_per_ha: float | None = None,
) -> MarginResult:
    cost_per_ha = cost_kes_per_ha if cost_kes_per_ha is not None else CROPS[crop].cost_kes_per_ha
    revenue = yield_t_per_ha * 1000 * price_kes_per_kg * area_ha
    cost = cost_per_ha * area_ha
    gm = revenue - cost
    return MarginResult(
        crop=crop,
        area_ha=area_ha,
        yield_t_per_ha=round(yield_t_per_ha, 2),
        price_kes_per_kg=round(price_kes_per_kg, 2),
        revenue_kes=round(revenue, 0),
        cost_kes=round(cost, 0),
        gross_margin_kes=round(gm, 0),
        gross_margin_pct=round(0.0 if revenue == 0 else gm / revenue * 100, 2),
        breakeven_price_kes_per_kg=round(cost_per_ha / (yield_t_per_ha * 1000), 2),
    )


# --------------------------------------------------------------------------- #
# Climate risk
# --------------------------------------------------------------------------- #
def classify_climate_risk(anomaly_pct: float, spi: float) -> tuple[RiskLevel, RiskLevel, RiskLevel]:
    """Return (overall, dry-spell, flood) risk from a rainfall anomaly and SPI."""
    if anomaly_pct <= -30 or spi <= -1.5:
        dry = RiskLevel.SEVERE
    elif anomaly_pct <= -15 or spi <= -1.0:
        dry = RiskLevel.HIGH
    elif anomaly_pct <= -5:
        dry = RiskLevel.MODERATE
    else:
        dry = RiskLevel.LOW

    if anomaly_pct >= 40 or spi >= 2.0:
        flood = RiskLevel.SEVERE
    elif anomaly_pct >= 25 or spi >= 1.5:
        flood = RiskLevel.HIGH
    elif anomaly_pct >= 10:
        flood = RiskLevel.MODERATE
    else:
        flood = RiskLevel.LOW

    order = list(RiskLevel)
    overall = max(dry, flood, key=order.index)
    return overall, dry, flood


# --------------------------------------------------------------------------- #
# Land suitability (FAO-style limiting-factor scoring)
# --------------------------------------------------------------------------- #
def _band_score(value: float, lo: float, hi: float, tolerance: float) -> float:
    if lo <= value <= hi:
        return 1.0
    gap = lo - value if value < lo else value - hi
    return max(0.0, 1.0 - gap / tolerance)


def suitability(
    crop: str,
    *,
    soil_ph: float,
    altitude_m: float,
    season_rainfall_mm: float,
    drainage: str,
) -> tuple[float, str, list[str]]:
    """Score crop/site suitability 0..1 and return (score, FAO class, limiting factors)."""
    p = CROPS[crop]
    factors: dict[str, float] = {
        "soil pH": _band_score(soil_ph, p.ph_min, p.ph_max, 1.5),
        "altitude": _band_score(altitude_m, p.alt_min_m, p.alt_max_m, 600),
    }
    if season_rainfall_mm < p.rain_min_mm:
        factors["rainfall deficit"] = max(0.0, season_rainfall_mm / p.rain_min_mm)
    elif season_rainfall_mm > p.rain_max_mm * 1.5 and p.waterlogging_sensitive:
        factors["excess rainfall / waterlogging"] = max(
            0.3, 1 - (season_rainfall_mm - p.rain_max_mm * 1.5) / p.rain_max_mm
        )
    else:
        factors["rainfall"] = 1.0
    if drainage == "poor" and p.waterlogging_sensitive:
        factors["poor drainage"] = 0.7
    elif drainage == "excessive":
        factors["low water holding capacity"] = 0.85

    # Geometric mean, capped by the single worst factor + 0.15 (limiting-factor rule).
    values = [max(v, 1e-3) for v in factors.values()]
    geo = math.exp(sum(math.log(v) for v in values) / len(values))
    score = round(min(geo, min(values) + 0.15), 3)
    klass = "S1" if score >= 0.8 else "S2" if score >= 0.6 else "S3" if score >= 0.4 else "N"
    limiting = [f"{name} ({value:.2f})" for name, value in factors.items() if value < 0.8]
    return score, klass, limiting


def expected_yield(crop: str, score: float, rainfall_anomaly_pct: float) -> float:
    """Attainable yield scaled by suitability and a mild rainfall-anomaly response."""
    base = CROPS[crop].yield_t_per_ha * max(0.15, score)
    climate = 1 + max(-0.4, min(0.1, rainfall_anomaly_pct / 100 * 0.6))
    return round(base * climate, 2)


def date_add(day: dt.date, days: int) -> dt.date:
    return day + dt.timedelta(days=days)
