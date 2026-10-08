from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from agri_resilience_crew import analytics
from agri_resilience_crew.reference import COUNTIES, road_distance_km
from agri_resilience_crew.schemas import CrewInputs, CropAction, CropPlan, KamisPriceQuery, RiskLevel


def test_all_47_counties_present() -> None:
    assert len(COUNTIES) == 47


def test_inputs_normalise_county_crop_and_default_sub_counties() -> None:
    inputs = CrewInputs(county="nakuru", crops=["potatoes", "Irish Potatoes", "mahindi"])
    assert inputs.county == "Nakuru"
    assert inputs.crops == ["Irish Potatoes", "Maize"]  # aliases resolved and de-duplicated
    assert inputs.sub_counties == ["Molo", "Njoro", "Kuresoi North"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"county": "Atlantis", "crops": ["Maize"]},
        {"county": "Nakuru", "crops": ["Coffee"]},
        {"county": "Nakuru", "crops": ["Maize"], "sub_counties": ["Kibera"]},
        {"county": "Nakuru", "crops": []},
        {"county": "Nakuru", "crops": ["Maize"], "horizon_days": 365},
    ],
)
def test_inputs_reject_invalid(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        CrewInputs(**kwargs)


def test_tool_query_rejects_unknown_market() -> None:
    with pytest.raises(ValidationError):
        KamisPriceQuery(crop="Maize", origin_county="Nakuru", markets=["Gikomba"])


def test_crop_plan_rejects_incoherent_windows() -> None:
    base = dict(
        crop="Maize", action=CropAction.PLANT, target_market=None, expected_yield_t_per_ha=3,
        expected_price_kes_per_kg=40, expected_revenue_kes_per_ha=120_000, expected_cost_kes_per_ha=110_000,
        expected_gross_margin_kes_per_ha=10_000, expected_margin_pct=8.3, confidence=0.5, rationale="x",
        planting_window_start=dt.date(2026, 10, 10), planting_window_end=dt.date(2026, 10, 30),
        harvest_window_end=dt.date(2027, 4, 1),
    )  # fmt: skip
    CropPlan(**base, harvest_window_start=dt.date(2027, 3, 1))
    with pytest.raises(ValidationError, match="harvest must start after planting"):
        CropPlan(**base, harvest_window_start=dt.date(2026, 10, 1))
    with pytest.raises(ValidationError, match="switch_to"):
        CropPlan(**{**base, "action": CropAction.SWITCH}, harvest_window_start=dt.date(2027, 3, 1))


def test_ols_slope_and_pct_change() -> None:
    assert analytics.ols_slope([1, 2, 3, 4]) == pytest.approx(1.0)
    assert analytics.ols_slope([5]) == 0.0
    assert analytics.pct_change(110, 100) == pytest.approx(10.0)


def test_price_signal_rising_series_and_logistics() -> None:
    prices = [40 + 0.5 * i for i in range(60)]
    sig = analytics.price_signal(
        crop="Irish Potatoes",
        market="Nairobi Wakulima",
        origin_county="Nakuru",
        prices=prices,
        farmgate_price=25,
    )
    assert sig.velocity_kes_per_kg_per_day == pytest.approx(0.5)
    assert analytics.trend_label(sig.velocity_kes_per_kg_per_day, sig.latest_price_kes_per_kg) == "rising"
    distance = road_distance_km("Nakuru", "Nairobi Wakulima")
    assert 120 < distance < 250  # Nakuru-Nairobi is ~160 km by road
    expected_net = 69.5 - 25 - sig.transport_kes_per_kg - sig.fees_kes_per_kg - sig.spoilage_kes_per_kg
    assert sig.net_margin_kes_per_kg == pytest.approx(expected_net, abs=0.02)
    assert sig.feasible


def test_price_signal_requires_30_days() -> None:
    with pytest.raises(ValueError):
        analytics.price_signal(
            crop="Maize", market="Eldoret Main", origin_county="Nakuru", prices=[1.0] * 10, farmgate_price=1
        )


def test_gross_margin_arithmetic() -> None:
    m = analytics.gross_margin("Irish Potatoes", yield_t_per_ha=15, price_kes_per_kg=30)
    assert m.revenue_kes == 450_000
    assert m.gross_margin_kes == 130_000
    assert m.breakeven_price_kes_per_kg == pytest.approx(21.33, abs=0.01)


@pytest.mark.parametrize(
    ("anomaly", "spi", "overall"),
    [(0, 0, RiskLevel.LOW), (-8, -0.3, RiskLevel.MODERATE), (-20, -0.7, RiskLevel.HIGH),
     (-35, -1.2, RiskLevel.SEVERE), (30, 1.0, RiskLevel.HIGH)],
)  # fmt: skip
def test_climate_risk_classification(anomaly: float, spi: float, overall: RiskLevel) -> None:
    assert analytics.classify_climate_risk(anomaly, spi)[0] is overall


def test_suitability_limiting_factors() -> None:
    good = analytics.suitability(
        "Irish Potatoes", soil_ph=5.8, altitude_m=2400, season_rainfall_mm=600, drainage="good"
    )
    assert good[1] == "S1" and good[2] == []
    bad = analytics.suitability(
        "Tomatoes", soil_ph=4.5, altitude_m=2600, season_rainfall_mm=150, drainage="poor"
    )
    assert bad[1] == "N"
    assert any("altitude" in f for f in bad[2])
