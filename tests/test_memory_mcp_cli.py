from __future__ import annotations

import pytest
from conftest import AS_OF
from crewai.mcp import MCPServerHTTP, MCPServerStdio

from agri_resilience_crew import __main__ as cli
from agri_resilience_crew import engine
from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.mcp import mcp_servers_by_agent
from agri_resilience_crew.memory import SeasonLedger, build_memory
from agri_resilience_crew.schemas import CrewInputs


def test_season_ledger_round_trip_and_prompt_context(tmp_path, bundle) -> None:
    ledger = SeasonLedger(tmp_path / "ledger.sqlite")
    assert "No prior playbooks" in ledger.prompt_context("Nakuru", ["Maize"])
    out = engine.synthesize(bundle)
    ledger.record("run1", out)
    assert ledger.history("Nakuru")[0] == out
    ctx = ledger.prompt_context("Nakuru", ["Maize"])
    assert "Maize" in ctx and "{" not in ctx  # safe for CrewAI {placeholder} interpolation


def test_memory_disabled_without_embedder(settings) -> None:
    assert (
        build_memory(settings.model_copy(update={"memory_enabled": True, "embedder": None}), llm=None) is None
    )


def test_embedder_defaults_follow_available_keys() -> None:
    assert CrewSettings.from_env({"ANTHROPIC_API_KEY": "x"}).embedder is None
    assert CrewSettings.from_env({"GEMINI_API_KEY": "g"}).embedder["provider"] == "google-generativeai"
    s = CrewSettings.from_env({"ANTHROPIC_API_KEY": "x"})
    assert s.llm_model == "anthropic/claude-sonnet-5-5" and s.manager_llm_model == "anthropic/claude-opus-5-5"
    assert (
        CrewSettings.from_env({"ANTHROPIC_API_KEY": "x", "AGRI_CREW_LLM_MODEL": "offline"}).llm_model is None
    )


def test_mcp_servers_mapped_to_agents(monkeypatch) -> None:
    monkeypatch.setenv("KB_TOKEN", "s3cret")
    servers = mcp_servers_by_agent(
        [
            {"type": "http", "url": "https://kb.example/mcp", "headers": {"Authorization": "Bearer ${KB_TOKEN}"},
             "agents": ["kalro_specialist"], "tools": ["search"]},
            {"type": "stdio", "command": "uvx", "args": ["geo-mcp"]},
        ]
    )  # fmt: skip
    http = servers["kalro_specialist"][0]
    assert isinstance(http, MCPServerHTTP) and http.headers == {"Authorization": "Bearer s3cret"}
    assert isinstance(servers["kmd_analyst"][0], MCPServerStdio)
    assert servers["agri_strategy_lead"] == []
    with pytest.raises(ValueError, match="unknown agents"):
        mcp_servers_by_agent([{"type": "http", "url": "https://x", "agents": ["intern"]}])


def test_cli_run_offline_records_ledger(settings) -> None:
    inputs = CrewInputs(county="Meru", crops=["Irish Potatoes"], as_of=AS_OF)
    out = cli.run(inputs, settings)
    assert out.engine == "deterministic" and out.county == "Meru"
    assert SeasonLedger(settings.ledger_db).history("Meru")[0].crop_plans == out.crop_plans
