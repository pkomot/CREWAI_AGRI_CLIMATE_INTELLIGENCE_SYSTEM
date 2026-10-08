"""Runs both real CrewAI crews end-to-end against a scripted LLM (no network)."""

from __future__ import annotations

from crewai import Process
from fake_llm import ScriptedLLM

from agri_resilience_crew import engine
from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.crew import AgriResilienceCrew, kickoff_inputs
from agri_resilience_crew.schemas import (
    AgriStrategyOutput,
    AgronomicFeasibilityAudit,
    ClimateImpactAssessment,
    MarketVolatilityAnalysis,
)
from agri_resilience_crew.sources import DataGateway


def _answers(bundle: engine.ResearchBundle, *, bad_first_playbook: bool = False) -> dict:
    c = bundle.primary_climate
    climate = ClimateImpactAssessment(
        county=c.county, season=c.season, risk_level=c.risk_level, expected_rainfall_mm=c.expected_rainfall_mm,
        rainfall_anomaly_pct=c.rainfall_anomaly_pct, planting_window_start=None, planting_window_end=None,
        sub_county_notes=["CTX-CLIMATE"], key_risks=["late onset"], summary="dry season",
    )  # fmt: skip
    market = MarketVolatilityAnalysis(origin_county=c.county, crops=[], summary="CTX-MARKET")
    agro = AgronomicFeasibilityAudit(county=c.county, crops=[], nema_compliant=True, summary="CTX-AGRO")
    good = engine.synthesize(bundle).model_copy(update={"engine": "crewai"})
    plan = good.crop_plans[0]
    bad = good.model_copy(
        update={
            "crop_plans": [
                plan.model_copy(update={"expected_revenue_kes_per_ha": plan.expected_revenue_kes_per_ha * 2}),
                *good.crop_plans[1:],
            ]
        }
    )

    def playbook(n: int) -> str:
        return (bad if bad_first_playbook and n == 0 else good).model_dump_json()

    return {
        "Task 1:": lambda n: climate.model_dump_json(),
        "Task 2:": lambda n: market.model_dump_json(),
        "Task 3:": lambda n: agro.model_dump_json(),
        "Task 4:": playbook,
    }


def _crew(settings: CrewSettings, gateway: DataGateway, bundle, llm) -> AgriResilienceCrew:
    online = settings.model_copy(update={"llm_model": "scripted", "manager_llm_model": "scripted"})
    return AgriResilienceCrew(online, gateway, bundle, llm=llm, manager_llm=llm)


def test_topology_and_agent_guardrails(settings, gateway, bundle) -> None:
    crew = _crew(settings, gateway, bundle, ScriptedLLM(_answers(bundle)))
    research, strategy = crew.research_crew(), crew.strategy_crew()

    assert research.process is Process.sequential
    assert [t.async_execution for t in research.tasks] == [True, True, False]
    assert strategy.process is Process.hierarchical
    assert strategy.manager_agent is crew.agri_strategy_lead
    assert strategy.manager_agent not in strategy.agents
    assert not strategy.manager_agent.tools and strategy.manager_agent.allow_delegation
    assert strategy.tasks[0].output_pydantic is AgriStrategyOutput
    for agent in [*research.agents, strategy.manager_agent]:
        assert agent.max_iter == 15 and agent.max_rpm == 30
    assert research.max_rpm == strategy.max_rpm == 30


def test_research_then_hierarchical_strategy_end_to_end(settings, gateway, bundle) -> None:
    llm = ScriptedLLM(_answers(bundle))
    crew = _crew(settings, gateway, bundle, llm)

    research = crew.research_crew().kickoff(
        inputs=kickoff_inputs(bundle, historical_context="LEDGER-CTX", risk_mandate="")
    )
    assert [type(t.pydantic) for t in research.tasks_output] == [
        ClimateImpactAssessment,
        MarketVolatilityAnalysis,
        AgronomicFeasibilityAudit,
    ]

    result = crew.strategy_crew().kickoff(
        inputs=kickoff_inputs(bundle, historical_context="LEDGER-CTX", risk_mandate="CONTINGENCY-MANDATE")
    )
    assert isinstance(result.pydantic, AgriStrategyOutput)
    assert result.pydantic.engine == "crewai"

    strategy_prompts = [text for marker, text in llm.calls if marker == "Task 4:"]
    # Research outputs cross the crew boundary as task context; inputs are interpolated.
    for needle in ("CTX-CLIMATE", "CTX-MARKET", "CTX-AGRO", "CONTINGENCY-MANDATE", "LEDGER-CTX"):
        assert needle in strategy_prompts[0], needle


def test_guardrail_forces_retry_on_inconsistent_playbook(settings, gateway, bundle) -> None:
    llm = ScriptedLLM(_answers(bundle, bad_first_playbook=True))
    crew = _crew(settings, gateway, bundle, llm)
    crew.research_crew().kickoff(inputs=kickoff_inputs(bundle, historical_context="", risk_mandate=""))
    result = crew.strategy_crew().kickoff(
        inputs=kickoff_inputs(bundle, historical_context="", risk_mandate="")
    )

    strategy_prompts = [text for marker, text in llm.calls if marker == "Task 4:"]
    assert len(strategy_prompts) >= 2  # first answer rejected, second accepted
    assert any("gross_margin_calculator" in p for p in strategy_prompts[1:])  # feedback reached the model
    assert isinstance(result.pydantic, AgriStrategyOutput)
    plan = result.pydantic.crop_plans[0]
    assert (
        plan.expected_revenue_kes_per_ha
        == engine.synthesize(bundle).crop_plans[0].expected_revenue_kes_per_ha
    )
