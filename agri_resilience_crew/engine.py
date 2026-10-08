"""Deterministic research and synthesis engine.

Three jobs:

1. Offline mode (no LLM key): produce the full :class:`AgriStrategyOutput` from tool data.
2. LLM mode: pre-compute a :class:`ResearchBundle` that the guardrail uses to
   reject hallucinated numbers in the crew's playbook.
3. Fallback: if the crew fails or its output fails validation, the run still
   returns a typed playbook, marked ``engine="deterministic"``.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from agri_resilience_crew import analytics
from agri_resilience_crew.reference import CROPS, MARKETS
from agri_resilience_crew.schemas import (
    AgriStrategyOutput,
    ClimateOutlook,
    CrewInputs,
    CropAction,
    CropPlan,
    DataProvenance,
    KamisPriceReport,
    MarketPriceSignal,
    PricingAction,
    RiskLevel,
    RiskMitigation,
    SoilSuitabilityReport,
)
from agri_resilience_crew.sources import DataGateway, DataUnavailableError

_RISK_ORDER = list(RiskLevel)
STORABLE = {"Irish Potatoes": "diffused-light store (6-8 weeks)", "Maize": "hermetic bags (months)", "Onions": "cured, ventilated store (2-3 months)", "Beans": "hermetic bags (months)"}  # fmt: skip


class CropEvidence(BaseModel):
    crop: str
    market: KamisPriceReport | None
    soil: SoilSuitabilityReport


class ResearchBundle(BaseModel):
    inputs: CrewInputs
    climate: list[ClimateOutlook]
    crops: list[CropEvidence]
    alternatives: list[CropEvidence]
    gaps: list[str]

    @property
    def primary_climate(self) -> ClimateOutlook:
        return self.climate[0]


def gather_research(gateway: DataGateway, inputs: CrewInputs) -> ResearchBundle:
    """Fetch every source in parallel (bounded) and assemble the evidence bundle."""
    sites: list[str | None] = list(inputs.sub_counties) or [None]
    gaps: list[str] = []

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="agri-research") as pool:
        climate_futs = [
            pool.submit(gateway.climate_outlook, inputs.county, s, inputs.as_of, inputs.horizon_days)
            for s in sites
        ]
        climate = [f.result() for f in climate_futs]
        anomaly = climate[0].rainfall_anomaly_pct
        primary = sites[0]

        candidates = list(dict.fromkeys([*inputs.crops, *CROPS]))
        market_futs = {
            c: pool.submit(gateway.kamis_report, c, inputs.county, list(MARKETS), inputs.as_of, 90)
            for c in candidates
        }
        soil_futs = {
            c: pool.submit(gateway.soil_suitability, inputs.county, primary, c, inputs.as_of, anomaly)
            for c in candidates
        }
        evidence: dict[str, CropEvidence] = {}
        for crop in candidates:
            try:
                market = market_futs[crop].result()
            except DataUnavailableError as exc:
                gaps.append(str(exc))
                market = None
            evidence[crop] = CropEvidence(crop=crop, market=market, soil=soil_futs[crop].result())

    return ResearchBundle(
        inputs=inputs,
        climate=climate,
        crops=[evidence[c] for c in inputs.crops],
        alternatives=[evidence[c] for c in candidates if c not in inputs.crops],
        gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Economics per crop
# --------------------------------------------------------------------------- #
def best_signal(report: KamisPriceReport | None) -> MarketPriceSignal | None:
    if report is None:
        return None
    if report.best_market:
        return next(s for s in report.signals if s.market == report.best_market)
    return max(report.signals, key=lambda s: s.net_margin_kes_per_kg, default=None)


def realised_price(signal: MarketPriceSignal | None) -> float:
    """Net-back price to the farmer, with a volatility haircut.

    That is the best market price minus transport, fees and spoilage, cut by
    half the 30-day volatility. When hauling is not worth it, the farmgate price is used.
    """
    if signal is None:
        return 0.0
    netback = max(
        signal.farmgate_price_kes_per_kg, signal.farmgate_price_kes_per_kg + signal.net_margin_kes_per_kg
    )
    return round(netback * (1 - signal.volatility_30d_pct / 200), 2)


def crop_margin_pct(ev: CropEvidence) -> float:
    price = realised_price(best_signal(ev.market))
    if price <= 0:
        return -100.0
    return analytics.gross_margin(
        ev.crop, yield_t_per_ha=max(ev.soil.expected_yield_t_per_ha, 0.01), price_kes_per_kg=price
    ).gross_margin_pct


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def planting_window(climate: ClimateOutlook, as_of: dt.date) -> tuple[dt.date, dt.date, bool]:
    """Return (start, end, rain_fed). Without a usable onset, planting assumes irrigation."""
    if climate.expected_onset is None:
        return as_of + dt.timedelta(days=7), as_of + dt.timedelta(days=28), False
    start = max(as_of, climate.expected_onset - dt.timedelta(days=3))
    end = max(start + dt.timedelta(days=14), climate.expected_onset + dt.timedelta(days=21))
    rain_fed = climate.expected_cessation is None or end < climate.expected_cessation
    return start, end, rain_fed


def decide(ev: CropEvidence, climate: ClimateOutlook, margin_pct: float) -> CropAction:
    # Zoning constraints apply to specific plots (riparian, forest edge), so they are
    # surfaced as compliance flags rather than vetoing the crop county-wide.
    if ev.soil.suitability_class == "N" or margin_pct < 0:
        return CropAction.SWITCH
    if climate.risk_level is RiskLevel.SEVERE:
        return CropAction.HOLD
    if (
        margin_pct >= 35
        and ev.soil.suitability_class in {"S1", "S2"}
        and _RISK_ORDER.index(climate.risk_level) <= _RISK_ORDER.index(RiskLevel.MODERATE)
    ):
        return CropAction.EXPAND
    return CropAction.PLANT


def synthesize(bundle: ResearchBundle) -> AgriStrategyOutput:
    inputs = bundle.inputs
    climate = bundle.primary_climate
    plant_start, plant_end, rain_fed = planting_window(climate, inputs.as_of)

    plans: list[CropPlan] = []
    conflicts: list[str] = []
    mitigations: list[RiskMitigation] = []
    pricing: list[PricingAction] = []
    compliance: set[str] = set()

    alt_ranked = sorted(bundle.alternatives, key=crop_margin_pct, reverse=True)

    for ev in bundle.crops:
        signal = best_signal(ev.market)
        price = realised_price(signal)
        yld = ev.soil.expected_yield_t_per_ha
        margin = analytics.gross_margin(
            ev.crop, yield_t_per_ha=max(yld, 0.01), price_kes_per_kg=max(price, 0.01)
        )
        action = decide(ev, climate, margin.gross_margin_pct)
        switch_to = None
        if action is CropAction.SWITCH:
            better = [
                a
                for a in alt_ranked
                if a.soil.suitability_class in {"S1", "S2"}
                and crop_margin_pct(a) > max(10, margin.gross_margin_pct)
            ]
            if better:
                switch_to = better[0].crop
            else:
                action = CropAction.AVOID

        maturity = CROPS[ev.crop].days_to_maturity
        confidence = _confidence(ev, climate, synthetic_data=any(p.synthetic for p in _provenance(bundle)))
        trend = (
            analytics.trend_label(signal.velocity_kes_per_kg_per_day, signal.latest_price_kes_per_kg)
            if signal
            else "flat"
        )

        rationale = (
            f"{ev.soil.suitability_class} suitability ({ev.soil.suitability_score:.2f}"
            f"{'; limits: ' + ', '.join(ev.soil.limiting_factors) if ev.soil.limiting_factors else ''}); "
            f"{climate.season} outlook {climate.rainfall_anomaly_pct:+.0f}% rainfall ({climate.risk_level.value} risk); "
            f"price trend {trend} at {signal.market if signal else 'n/a'}; "
            f"net-back KES {price:.1f}/kg vs break-even KES {margin.breakeven_price_kes_per_kg:.1f}/kg."
        )
        if switch_to:
            alt = better[0]
            rationale += (
                f" Switch candidate {switch_to}: {alt.soil.suitability_class} suitability, "
                f"projected gross margin {crop_margin_pct(alt):.0f}%."
            )
        if not rain_fed:
            rationale += " Rain-fed window is unreliable: plan supplementary irrigation."
        plans.append(
            CropPlan(
                crop=ev.crop,
                action=action,
                switch_to=switch_to,
                planting_window_start=plant_start,
                planting_window_end=plant_end,
                harvest_window_start=plant_start + dt.timedelta(days=maturity),
                harvest_window_end=plant_end + dt.timedelta(days=maturity + 14),
                target_market=signal.market if signal and signal.feasible else None,
                expected_yield_t_per_ha=yld,
                expected_price_kes_per_kg=price,
                expected_revenue_kes_per_ha=margin.revenue_kes,
                expected_cost_kes_per_ha=margin.cost_kes,
                expected_gross_margin_kes_per_ha=margin.gross_margin_kes,
                expected_margin_pct=margin.gross_margin_pct,
                confidence=confidence,
                rationale=rationale,
            )
        )

        # Conflicts between specialist signals.
        if trend == "rising" and _RISK_ORDER.index(climate.risk_level) >= _RISK_ORDER.index(RiskLevel.HIGH):
            conflicts.append(
                f"{ev.crop}: KAMIS shows rising prices but KMD flags {climate.risk_level.value} climate "
                f"risk; resolved as '{action.value}' with staggered planting and irrigation backup."
            )
        if signal and signal.feasible and ev.soil.suitability_class in {"S3", "N"}:
            conflicts.append(
                f"{ev.crop}: profitable haul to {signal.market} but site suitability is "
                f"{ev.soil.suitability_class}; agronomy overrides market, so the action is '{action.value}'"
                + (f" (switch to {switch_to})" if switch_to else "")
                + "."
            )

        # Pest and zoning mitigations.
        for pest in ev.soil.pest_advisories:
            if pest.risk in {RiskLevel.HIGH, RiskLevel.SEVERE}:
                mitigations.append(
                    RiskMitigation(
                        risk=f"{pest.pest} on {ev.crop}",
                        likelihood=pest.risk,
                        impact=RiskLevel.HIGH,
                        mitigation=pest.action,
                    )
                )
        for z in ev.soil.zoning_constraints:
            if z.severity != "advisory":
                compliance.add(f"[{z.authority}] {z.zone}: {z.rule}")

        # Pricing playbook.
        if signal:
            pricing.append(
                PricingAction(
                    week_start=inputs.as_of,
                    week_end=inputs.as_of + dt.timedelta(days=6),
                    crop=ev.crop,
                    action=f"Lock input prices and line up {signal.market} buyers; benchmark against KAMIS weekly.",
                    trigger=f"Current {signal.market} price KES {signal.latest_price_kes_per_kg:.1f}/kg ({trend}).",
                )
            )
            harvest = plant_start + dt.timedelta(days=maturity)
            floor = round(margin.breakeven_price_kes_per_kg * 1.25, 1)
            store = STORABLE.get(ev.crop)
            pricing.append(
                PricingAction(
                    week_start=harvest,
                    week_end=harvest + dt.timedelta(days=13),
                    crop=ev.crop,
                    action=(
                        f"Sell in tranches at {signal.market if signal.feasible else 'farmgate'}"
                        + (
                            f"; hold unsold stock in {store}"
                            if store
                            else "; perishable, so pre-agree offtake"
                        )
                    ),
                    trigger=f"Net-back >= KES {floor}/kg (1.25x break-even); otherwise "
                    + ("store and re-check weekly." if store else "sell to contracted offtaker."),
                )
            )

    mitigations.extend(_climate_mitigations(climate))
    for ev in bundle.crops:
        sig = best_signal(ev.market)
        if sig and sig.volatility_30d_pct >= 15:
            mitigations.append(
                RiskMitigation(
                    risk=f"{ev.crop} price volatility {sig.volatility_30d_pct:.0f}% (30d)",
                    likelihood=RiskLevel.HIGH,
                    impact=RiskLevel.MODERATE,
                    mitigation="Stagger planting in 2-3 blocks two weeks apart and pre-sell part of the crop to an offtaker.",
                )
            )

    return AgriStrategyOutput(
        county=inputs.county,
        sub_counties=inputs.sub_counties,
        season=climate.season,
        as_of=inputs.as_of,
        horizon_days=inputs.horizon_days,
        climate_risk_level=climate.risk_level,
        climate_summary=_climate_summary(bundle),
        market_summary=_market_summary(bundle),
        crop_plans=plans,
        risk_mitigations=mitigations
        or [
            RiskMitigation(
                risk="No elevated risks detected",
                likelihood=RiskLevel.LOW,
                impact=RiskLevel.LOW,
                mitigation="Continue weekly monitoring of KMD and KAMIS bulletins.",
            )
        ],
        pricing_playbook=sorted(pricing, key=lambda p: p.week_start),
        compliance_flags=sorted(compliance),
        conflicts_resolved=conflicts,
        data_provenance=_provenance(bundle),
        engine="deterministic",
    )


def _confidence(ev: CropEvidence, climate: ClimateOutlook, *, synthetic_data: bool) -> float:
    c = 0.85 * (0.5 + 0.5 * ev.soil.suitability_score)
    c -= 0.08 * _RISK_ORDER.index(climate.risk_level)
    if ev.market is None:
        c -= 0.2
    if synthetic_data:
        c -= 0.15
    return round(min(0.9, max(0.1, c)), 2)


def _climate_mitigations(climate: ClimateOutlook) -> list[RiskMitigation]:
    out: list[RiskMitigation] = []
    if _RISK_ORDER.index(climate.dry_spell_risk) >= _RISK_ORDER.index(RiskLevel.MODERATE):
        out.append(
            RiskMitigation(
                risk=f"Below-normal rainfall ({climate.rainfall_anomaly_pct:+.0f}%, SPI {climate.spi_3month:+.2f})",
                likelihood=climate.dry_spell_risk,
                impact=RiskLevel.HIGH,
                mitigation="Plant early-maturing/drought-tolerant varieties, mulch, and prepare drip or furrow irrigation and on-farm water pans before onset.",
            )
        )
    if _RISK_ORDER.index(climate.flood_risk) >= _RISK_ORDER.index(RiskLevel.MODERATE):
        out.append(
            RiskMitigation(
                risk=f"Above-normal rainfall / waterlogging ({climate.rainfall_anomaly_pct:+.0f}%)",
                likelihood=climate.flood_risk,
                impact=RiskLevel.HIGH,
                mitigation="Use raised beds and drainage furrows; tighten fungal-disease spray intervals; avoid valley-bottom plots.",
            )
        )
    if climate.prob_below_normal >= 0.45:
        out.append(
            RiskMitigation(
                risk="Late or erratic onset of rains",
                likelihood=RiskLevel.HIGH,
                impact=RiskLevel.MODERATE,
                mitigation="Wait for 20-25 mm cumulative rain before dry-planting; stagger planting; keep seed in reserve for gap-filling.",
            )
        )
    if climate.heat_stress_days >= 15:
        out.append(
            RiskMitigation(
                risk=f"{climate.heat_stress_days} heat-stress days expected",
                likelihood=RiskLevel.MODERATE,
                impact=RiskLevel.MODERATE,
                mitigation="Irrigate in early morning, use shade nets on nurseries, and time flowering away from the hottest weeks.",
            )
        )
    return out


def _climate_summary(bundle: ResearchBundle) -> str:
    parts = []
    for c in bundle.climate:
        site = c.sub_county or c.county
        parts.append(
            f"{site}: {c.expected_rainfall_mm:.0f} mm expected vs {c.normal_rainfall_mm:.0f} mm normal "
            f"({c.rainfall_anomaly_pct:+.0f}%), {c.risk_level.value} risk"
            + (f", onset ~{c.expected_onset:%d %b}" if c.expected_onset else ", no reliable onset")
        )
    p = bundle.primary_climate
    return (
        f"{p.season}: tercile probabilities above/near/below = {p.prob_above_normal:.0%}/"
        f"{p.prob_near_normal:.0%}/{p.prob_below_normal:.0%}. " + "; ".join(parts) + "."
    )


def _market_summary(bundle: ResearchBundle) -> str:
    parts = []
    for ev in bundle.crops:
        sig = best_signal(ev.market)
        if sig is None:
            parts.append(f"{ev.crop}: no KAMIS data")
            continue
        trend = analytics.trend_label(sig.velocity_kes_per_kg_per_day, sig.latest_price_kes_per_kg)
        parts.append(
            f"{ev.crop}: best market {sig.market} at KES {sig.latest_price_kes_per_kg:.1f}/kg "
            f"({trend}, {sig.velocity_kes_per_kg_per_day:+.2f} KES/kg/day, vol {sig.volatility_30d_pct:.0f}%), "
            f"haul margin KES {sig.net_margin_kes_per_kg:.1f}/kg over farmgate"
        )
    return "; ".join(parts) + "."


def _provenance(bundle: ResearchBundle) -> list[DataProvenance]:
    seen: dict[tuple[str, str], DataProvenance] = {}
    for c in bundle.climate:
        seen.setdefault((c.provenance.source, c.provenance.mode), c.provenance)
    for ev in [*bundle.crops, *bundle.alternatives]:
        if ev.market:
            seen.setdefault((ev.market.provenance.source, ev.market.provenance.mode), ev.market.provenance)
        for p in ev.soil.provenance:
            seen.setdefault((p.source, p.mode), p)
    return list(seen.values())
