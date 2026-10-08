from __future__ import annotations

import datetime as dt

import pytest
from conftest import AS_OF

from agri_resilience_crew import engine
from agri_resilience_crew.guardrails import validate_playbook
from agri_resilience_crew.schemas import AgriStrategyOutput, CrewInputs, CropAction
from agri_resilience_crew.sources import DataGateway


@pytest.mark.parametrize(
    ("county", "crops"),
    [
        ("Nakuru", ["Irish Potatoes", "Maize", "Tomatoes", "Onions"]),
        ("Uasin Gishu", ["Maize", "Irish Potatoes"]),
        ("Meru", ["Irish Potatoes", "Tomatoes"]),
        ("Turkana", ["Beans"]),
        ("Kisumu", ["Tomatoes", "Cabbages"]),
    ],
)
def test_synthesis_produces_valid_playbook(gateway: DataGateway, county: str, crops: list[str]) -> None:
    inputs = CrewInputs(county=county, crops=crops, as_of=AS_OF)
    bundle = engine.gather_research(gateway, inputs)
    out = engine.synthesize(bundle)
    AgriStrategyOutput.model_validate(out.model_dump())  # round-trips through the schema
    assert [p.crop for p in out.crop_plans] == inputs.crops
    assert out.engine == "deterministic"
    assert all(p.synthetic for p in out.data_provenance if p.source in {"KAMIS", "KMD", "KALRO"})
    # The deterministic engine must pass the same guardrail the LLM is held to.
    assert validate_playbook(out, bundle, synthetic=True) == []


def test_switch_targets_are_well_suited(gateway: DataGateway) -> None:
    inputs = CrewInputs(county="Nakuru", crops=["Tomatoes", "Onions"], sub_counties=["Molo"], as_of=AS_OF)
    bundle = engine.gather_research(gateway, inputs)
    out = engine.synthesize(bundle)
    suitability = {ev.crop: ev.soil.suitability_class for ev in bundle.alternatives}
    for plan in out.crop_plans:
        assert plan.action in {CropAction.SWITCH, CropAction.AVOID}  # 2,500 m is too high for both
        if plan.switch_to:
            assert suitability[plan.switch_to] in {"S1", "S2"}


def test_dry_outlook_records_market_vs_climate_conflict(bundle: engine.ResearchBundle) -> None:
    out = engine.synthesize(bundle)
    assert out.climate_risk_level.value in {"high", "severe"}  # synthetic OND 2026 is dry in Nakuru
    assert out.conflicts_resolved
    assert any("rainfall" in m.risk.lower() for m in out.risk_mitigations)


def test_guardrail_catches_tampered_numbers_and_coverage(bundle: engine.ResearchBundle) -> None:
    good = engine.synthesize(bundle)
    plan = good.crop_plans[0]
    tampered = good.model_copy(
        update={
            "crop_plans": [
                plan.model_copy(update={"expected_revenue_kes_per_ha": plan.expected_revenue_kes_per_ha * 3}),
            ],
            "data_provenance": [p.model_copy(update={"synthetic": False}) for p in good.data_provenance],
        }
    )
    errors = validate_playbook(tampered, bundle, synthetic=True)
    joined = "\n".join(errors)
    assert "revenue" in joined
    assert "missing requested crops" in joined
    assert "synthetic=true" in joined


def test_guardrail_rejects_hallucinated_yield_and_price(bundle: engine.ResearchBundle) -> None:
    good = engine.synthesize(bundle)
    plan = good.crop_plans[0]
    yld, price = plan.expected_yield_t_per_ha * 4, 999.0
    revenue = yld * 1000 * price
    fake = plan.model_copy(
        update={
            "expected_yield_t_per_ha": yld,
            "expected_price_kes_per_kg": price,
            "expected_revenue_kes_per_ha": revenue,
            "expected_gross_margin_kes_per_ha": revenue - plan.expected_cost_kes_per_ha,
            "expected_margin_pct": (revenue - plan.expected_cost_kes_per_ha) / revenue * 100,
        }
    )
    out = good.model_copy(update={"crop_plans": [fake, *good.crop_plans[1:]]})
    errors = "\n".join(validate_playbook(out, bundle, synthetic=True))
    assert "outside +/-50%" in errors and "evidence band" in errors


def test_planting_window_without_onset_assumes_irrigation(bundle: engine.ResearchBundle) -> None:
    climate = bundle.primary_climate.model_copy(update={"expected_onset": None, "expected_cessation": None})
    start, end, rain_fed = engine.planting_window(climate, AS_OF)
    assert not rain_fed and start == AS_OF + dt.timedelta(days=7) and end > start
