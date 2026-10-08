"""AgriResilience-Crew: agents, tasks and the two crews built from YAML with ``@CrewBase``.

Topology (why two crews):

CrewAI's ``Process.hierarchical`` runs *every* task through the manager agent,
and an agent has a single executor. Two ``async_execution=True`` tasks in one
hierarchical crew would therefore run concurrently on the same manager and
overwrite each other's executor state. So the work is split into two phases:

1. ``research_crew``: ``Process.sequential``. Climate (Task 1) and Market (Task 2)
   are independent and run concurrently (``async_execution=True``) on different
   specialist agents. The Agronomy/NEMA audit (Task 3) is synchronous: it joins
   both and uses their outputs as context.
2. ``strategy_crew``: ``Process.hierarchical``, with ``agri_strategy_lead`` as the
   custom ``manager_agent``. The manager receives Tasks 1-3 as context, delegates
   follow-up questions to the specialists (``allow_delegation=True``) and returns
   ``AgriStrategyOutput`` through ``output_pydantic`` and an evidence guardrail.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool

from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.engine import ResearchBundle
from agri_resilience_crew.guardrails import playbook_guardrail
from agri_resilience_crew.mcp import mcp_servers_by_agent
from agri_resilience_crew.schemas import (
    AgriStrategyOutput,
    AgronomicFeasibilityAudit,
    ClimateImpactAssessment,
    MarketVolatilityAnalysis,
)
from agri_resilience_crew.sources import DataGateway
from agri_resilience_crew.tools import build_tools

_NATIVE_RETRY_PROVIDERS = ("anthropic/", "openai/", "azure/")


def build_llm(model: str, settings: CrewSettings, *, temperature: float = 0.2) -> LLM:
    """CrewAI LLM with timeouts and SDK-level retries where the provider supports them.

    BaseLLM also wraps every call in jittered exponential backoff on HTTP 429.
    """
    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
    if model.startswith(_NATIVE_RETRY_PROVIDERS):
        kwargs["timeout"] = settings.llm_timeout_s
        kwargs["max_retries"] = settings.llm_max_retries
    return LLM(**kwargs)


def _web_search_tools() -> list[BaseTool]:
    """Optional web search for the climate analyst (Serper), when a key is configured."""
    if not os.environ.get("SERPER_API_KEY"):
        return []
    from crewai_tools import SerperDevTool

    return [SerperDevTool(country="ke", n_results=5)]


@CrewBase
class AgriResilienceCrew:
    """Climate-adaptive agricultural market intelligence crew for Kenyan counties."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        settings: CrewSettings,
        gateway: DataGateway,
        bundle: ResearchBundle,
        *,
        memory: Any = None,
        llm: LLM | Any | None = None,
        manager_llm: LLM | Any | None = None,
    ) -> None:
        if llm is None and settings.llm_model is None:
            raise ValueError("AgriResilienceCrew needs an LLM; use the deterministic engine offline")
        self.settings = settings
        self.bundle = bundle
        self.memory = memory
        self.llm = llm or build_llm(settings.llm_model or "", settings)
        self.manager_llm = manager_llm or build_llm(
            settings.manager_llm_model or settings.llm_model or "", settings, temperature=0.1
        )
        self.tools = build_tools(gateway, bundle.inputs.as_of)
        self.mcps = mcp_servers_by_agent(settings.mcp_servers)

    # ------------------------------------------------------------------ #
    # Agents
    # ------------------------------------------------------------------ #
    def _agent_kwargs(self, key: str) -> dict[str, Any]:
        s = self.settings
        return {
            "llm": self.llm,
            "max_iter": s.max_iter,
            "max_rpm": s.max_rpm,
            "max_retry_limit": s.max_retry_limit,
            "max_execution_time": s.agent_max_execution_time_s,
            "respect_context_window": True,
            "inject_date": True,
            "mcps": self.mcps.get(key) or None,
            "verbose": s.verbose,
        }

    @agent
    def kmd_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["kmd_analyst"],  # type: ignore[index]
            tools=[self.tools["kmd"], self.tools["reference"], *_web_search_tools()],
            allow_delegation=False,
            **self._agent_kwargs("kmd_analyst"),
        )

    @agent
    def kamis_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["kamis_analyst"],  # type: ignore[index]
            tools=[self.tools["kamis"], self.tools["margin"], self.tools["reference"]],
            allow_delegation=False,
            **self._agent_kwargs("kamis_analyst"),
        )

    @agent
    def kalro_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["kalro_specialist"],  # type: ignore[index]
            tools=[self.tools["kalro"], self.tools["reference"]],
            # May ask the climate/market specialists for clarification during the audit.
            allow_delegation=True,
            **self._agent_kwargs("kalro_specialist"),
        )

    @cached_property
    def agri_strategy_lead(self) -> Agent:
        """Manager agent. Deliberately not ``@agent``: a manager must not be in ``agents``,
        and CrewAI forbids tools on a hierarchical manager (it delegates instead)."""
        kwargs = self._agent_kwargs("agri_strategy_lead")
        kwargs["llm"] = self.manager_llm
        return Agent(
            config=self.agents_config["agri_strategy_lead"],  # type: ignore[index]
            allow_delegation=True,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Tasks
    # ------------------------------------------------------------------ #
    @task
    def climate_impact_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["climate_impact_assessment"],  # type: ignore[index]
            output_pydantic=ClimateImpactAssessment,
            async_execution=True,
        )

    @task
    def market_price_volatility_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["market_price_volatility_analysis"],  # type: ignore[index]
            output_pydantic=MarketVolatilityAnalysis,
            async_execution=True,
        )

    @task
    def agronomic_feasibility_audit(self) -> Task:
        return Task(
            config=self.tasks_config["agronomic_feasibility_audit"],  # type: ignore[index]
            output_pydantic=AgronomicFeasibilityAudit,
        )

    @task
    def strategy_synthesis(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_synthesis"],  # type: ignore[index]
            output_pydantic=AgriStrategyOutput,
            guardrail=playbook_guardrail(self.bundle, synthetic=self.settings.mode == "mock"),
            guardrail_max_retries=3,
        )

    # ------------------------------------------------------------------ #
    # Crews
    # ------------------------------------------------------------------ #
    def _crew_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_rpm": self.settings.max_rpm,
            "verbose": self.settings.verbose,
            "cache": True,
        }
        if self.memory is not None:
            kwargs["memory"] = self.memory
        return kwargs

    @crew
    def research_crew(self) -> Crew:
        return Crew(
            name="AgriResilience-Research",
            agents=[self.kmd_analyst(), self.kamis_analyst(), self.kalro_specialist()],
            tasks=[
                self.climate_impact_assessment(),
                self.market_price_volatility_analysis(),
                self.agronomic_feasibility_audit(),
            ],
            process=Process.sequential,
            **self._crew_kwargs(),
        )

    @crew
    def strategy_crew(self) -> Crew:
        return Crew(
            name="AgriResilience-Strategy",
            agents=[self.kmd_analyst(), self.kamis_analyst(), self.kalro_specialist()],
            tasks=[self.strategy_synthesis()],
            process=Process.hierarchical,
            manager_agent=self.agri_strategy_lead,
            **self._crew_kwargs(),
        )


def kickoff_inputs(bundle: ResearchBundle, *, historical_context: str, risk_mandate: str) -> dict[str, Any]:
    inputs = bundle.inputs.as_crew_inputs()
    inputs["historical_context"] = historical_context
    inputs["risk_mandate"] = risk_mandate
    return inputs
