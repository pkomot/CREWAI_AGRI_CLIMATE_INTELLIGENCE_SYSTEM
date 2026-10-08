"""CLI: python -m agri_resilience_crew --county Nakuru --crop "Irish Potatoes" --crop Maize

Without an LLM key the deterministic engine produces the playbook offline.
With a key, the research crew runs, then the hierarchical strategy crew does;
on crew failure the run falls back to the deterministic playbook.
"""

from __future__ import annotations

import argparse
import logging
import uuid

from agri_resilience_crew import engine
from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.memory import SeasonLedger, build_memory, remember_playbook, seed_entity_memory
from agri_resilience_crew.schemas import AgriStrategyOutput, CrewInputs, RiskLevel
from agri_resilience_crew.sources import DataGateway

logger = logging.getLogger("agri_resilience_crew")


def run(inputs: CrewInputs, settings: CrewSettings) -> AgriStrategyOutput:
    run_id = uuid.uuid4().hex[:12]
    gateway = DataGateway(settings)
    ledger = SeasonLedger(settings.ledger_db)
    try:
        bundle = engine.gather_research(gateway, inputs)
        output: AgriStrategyOutput | None = None
        if settings.online:
            output = _run_crews(bundle, settings, gateway, ledger, run_id)
        if output is None:
            output = engine.synthesize(bundle)
        ledger.record(run_id, output)
        return output
    finally:
        gateway.close()


def _run_crews(bundle, settings, gateway, ledger, run_id) -> AgriStrategyOutput | None:
    from agri_resilience_crew.crew import AgriResilienceCrew, build_llm, kickoff_inputs

    try:
        llm = build_llm(settings.llm_model or "", settings)
        memory = build_memory(settings, llm)
        if memory is not None:
            seed_entity_memory(memory, bundle, run_id)
        crew = AgriResilienceCrew(settings, gateway, bundle, memory=memory, llm=llm)
        history = ledger.prompt_context(bundle.inputs.county, bundle.inputs.crops)

        crew.research_crew().kickoff(
            inputs=kickoff_inputs(bundle, historical_context=history, risk_mandate="")
        )
        # Conditional routing: a high/severe climate assessment adds a contingency mandate.
        climate = crew.climate_impact_assessment().output
        risk = getattr(getattr(climate, "pydantic", None), "risk_level", None)
        mandate = (
            "CLIMATE CONTINGENCY: risk is high/severe. Include a drought/flood contingency per crop "
            "and prefer hold/switch over expand unless irrigation is assured."
            if risk in {RiskLevel.HIGH, RiskLevel.SEVERE}
            else ""
        )
        result = crew.strategy_crew().kickoff(
            inputs=kickoff_inputs(bundle, historical_context=history, risk_mandate=mandate)
        )
        output = result.pydantic if isinstance(result.pydantic, AgriStrategyOutput) else None
        if output is not None and memory is not None:
            remember_playbook(memory, output)
        return output
    except Exception:  # noqa: BLE001 - always return a playbook
        logger.exception("crew run failed; falling back to deterministic engine")
        return None


def cli() -> None:
    p = argparse.ArgumentParser(description="AgriResilience-Crew 90-day playbook")
    p.add_argument("--county", default="Nakuru")
    p.add_argument("--crop", action="append", dest="crops")
    p.add_argument("--sub-county", action="append", dest="sub_counties", default=[])
    p.add_argument("--farm-size-ha", type=float, default=1.0)
    p.add_argument("--offline", action="store_true", help="force the deterministic engine")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = CrewSettings.from_env(**({"llm_model": None} if args.offline else {}))
    inputs = CrewInputs(
        county=args.county,
        crops=args.crops or ["Irish Potatoes"],
        sub_counties=args.sub_counties,
        farm_size_ha=args.farm_size_ha,
    )
    print(run(inputs, settings).model_dump_json(indent=2))


if __name__ == "__main__":
    cli()
