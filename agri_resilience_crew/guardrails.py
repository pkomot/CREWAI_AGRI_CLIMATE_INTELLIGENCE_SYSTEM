"""Task guardrails that check the LLM playbook against deterministic tool evidence.

``output_pydantic`` guarantees the playbook's *shape*. These guardrails check its
*substance*: arithmetic consistency, crop coverage, numbers anchored to tool
evidence and honest provenance. When a check fails, CrewAI re-runs the task
with the error list as feedback, up to ``guardrail_max_retries`` times.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

from crewai.tasks.task_output import TaskOutput

from agri_resilience_crew.engine import ResearchBundle
from agri_resilience_crew.schemas import AgriStrategyOutput

YIELD_TOLERANCE = 0.5  # +/-50% of the KALRO-derived expected yield
PRICE_BAND = (0.5, 1.5)  # x [lowest farmgate, highest observed wholesale]


def _close(a: float, b: float, rel: float = 0.05, abs_tol: float = 1_000.0) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def validate_playbook(output: AgriStrategyOutput, bundle: ResearchBundle, *, synthetic: bool) -> list[str]:
    errors: list[str] = []
    inputs = bundle.inputs
    if output.county != inputs.county:
        errors.append(f"county must be {inputs.county}, got {output.county}")
    planned = {p.crop for p in output.crop_plans}
    missing = set(inputs.crops) - planned
    extra = planned - set(inputs.crops)
    if missing:
        errors.append(f"crop_plans missing requested crops: {sorted(missing)}")
    if extra:
        errors.append(f"crop_plans contain unrequested crops {sorted(extra)}; use switch_to instead")

    evidence = {ev.crop: ev for ev in bundle.crops}
    for plan in output.crop_plans:
        ev = evidence.get(plan.crop)
        expected_rev = plan.expected_yield_t_per_ha * 1000 * plan.expected_price_kes_per_kg
        if not _close(plan.expected_revenue_kes_per_ha, expected_rev):
            errors.append(
                f"{plan.crop}: revenue {plan.expected_revenue_kes_per_ha:,.0f} != yield x 1000 x price "
                f"({expected_rev:,.0f}); use gross_margin_calculator"
            )
        gm = plan.expected_revenue_kes_per_ha - plan.expected_cost_kes_per_ha
        if not _close(plan.expected_gross_margin_kes_per_ha, gm):
            errors.append(
                f"{plan.crop}: gross margin {plan.expected_gross_margin_kes_per_ha:,.0f} != revenue - cost ({gm:,.0f})"
            )
        if plan.expected_revenue_kes_per_ha > 0:
            pct = gm / plan.expected_revenue_kes_per_ha * 100
            if abs(pct - plan.expected_margin_pct) > 2:
                errors.append(f"{plan.crop}: expected_margin_pct {plan.expected_margin_pct:.1f} != {pct:.1f}")
        if plan.planting_window_start < inputs.as_of - dt.timedelta(days=7):
            errors.append(f"{plan.crop}: planting window starts in the past ({plan.planting_window_start})")
        if ev is None:
            continue
        ref_yield = ev.soil.expected_yield_t_per_ha
        if ref_yield > 0 and abs(plan.expected_yield_t_per_ha - ref_yield) > YIELD_TOLERANCE * ref_yield:
            errors.append(
                f"{plan.crop}: yield {plan.expected_yield_t_per_ha} t/ha is outside +/-50% of the KALRO "
                f"estimate {ref_yield} t/ha; justify from tool data or correct it"
            )
        if ev.market and ev.market.signals:
            lo = min(s.farmgate_price_kes_per_kg for s in ev.market.signals) * PRICE_BAND[0]
            hi = max(s.latest_price_kes_per_kg for s in ev.market.signals) * PRICE_BAND[1]
            if not lo <= plan.expected_price_kes_per_kg <= hi:
                errors.append(
                    f"{plan.crop}: price KES {plan.expected_price_kes_per_kg}/kg outside the evidence band "
                    f"KES {lo:.1f}-{hi:.1f}/kg from KAMIS"
                )

    if not output.data_provenance:
        errors.append("data_provenance must list the KAMIS, KMD and KALRO sources used")
    elif synthetic and not any(p.synthetic for p in output.data_provenance):
        errors.append("data is synthetic in mock mode; data_provenance must set synthetic=true")
    return errors


def playbook_guardrail(
    bundle: ResearchBundle, *, synthetic: bool
) -> Callable[[TaskOutput], tuple[bool, Any]]:
    """Build a CrewAI guardrail closed over this run's evidence bundle."""

    # No return annotation: CrewAI inspects it at runtime and rejects string (postponed) annotations.
    def guardrail(result: TaskOutput):  # noqa: ANN202
        payload = result.pydantic
        if not isinstance(payload, AgriStrategyOutput):
            try:
                payload = AgriStrategyOutput.model_validate_json(result.raw)
            except Exception as exc:  # noqa: BLE001
                return False, f"Output is not a valid AgriStrategyOutput JSON object: {exc}"
        errors = validate_playbook(payload, bundle, synthetic=synthetic)
        if errors:
            return False, "Fix these issues and return the corrected playbook:\n- " + "\n- ".join(errors)
        result.pydantic = payload.model_copy(update={"engine": "crewai"})
        return True, result

    return guardrail
