"""Runtime configuration for AgriResilience-Crew.

All settings are read from environment variables (prefix ``AGRI_CREW_``) so the
same code runs locally, in CI and in production without modification.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_ANTHROPIC_MODEL = "anthropic/claude-sonnet-5-5"
DEFAULT_ANTHROPIC_MANAGER_MODEL = "anthropic/claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini/gemini-3-pro-preview"

MOCK_KAMIS_BASE_URL = "https://kamis.kilimo.mock/api/v1"
MOCK_KMD_BASE_URL = "https://meteo.go.mock/api/v1"
MOCK_KALRO_BASE_URL = "https://kalro.mock/api/v1"


class CrewSettings(BaseModel):
    """Immutable, validated runtime settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["mock", "live"] = Field(
        default="mock",
        description="'mock' serves synthetic datasets in-process; 'live' calls real endpoints.",
    )

    # LLMs (CrewAI 'provider/model' strings). None -> offline deterministic engine.
    llm_model: str | None = None
    manager_llm_model: str | None = None
    llm_timeout_s: float = Field(default=120.0, gt=0, le=900)
    llm_max_retries: int = Field(default=4, ge=0, le=10)

    # Agent guardrails (spec: max_iter=15, max_rpm=30).
    max_iter: int = Field(default=15, ge=1, le=50)
    max_rpm: int = Field(default=30, ge=1, le=600)
    max_retry_limit: int = Field(default=3, ge=0, le=10)
    agent_max_execution_time_s: int = Field(default=600, ge=30)

    # Tool I/O resilience.
    tool_timeout_s: float = Field(default=8.0, gt=0, le=120)
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_initial_backoff_s: float = Field(default=0.5, gt=0, le=30)
    retry_max_backoff_s: float = Field(default=8.0, gt=0, le=120)
    cache_ttl_s: float = Field(default=6 * 3600.0, gt=0)

    kamis_base_url: str = MOCK_KAMIS_BASE_URL
    kmd_base_url: str = MOCK_KMD_BASE_URL
    kalro_base_url: str = MOCK_KALRO_BASE_URL
    api_bearer_token: str | None = Field(default=None, repr=False)

    # Memory.
    memory_enabled: bool = True
    storage_dir: Path = Path(".agri_crew")
    embedder: dict[str, Any] | None = Field(
        default=None,
        description="CrewAI embedder spec, e.g. {'provider': 'google', 'config': {...}}.",
    )

    # MCP servers: JSON list of {"type": "http"|"sse"|"stdio", ...}.
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)

    verbose: bool = False

    @model_validator(mode="after")
    def _live_mode_requires_real_endpoints(self) -> CrewSettings:
        if self.mode == "live":
            mocked = [
                url for url in (self.kamis_base_url, self.kmd_base_url, self.kalro_base_url) if ".mock" in url
            ]
            if mocked:
                raise ValueError(
                    "AGRI_CREW_MODE=live requires real data endpoints; still mocked: " + ", ".join(mocked)
                )
        return self

    @property
    def online(self) -> bool:
        return self.llm_model is not None

    @property
    def memory_dir(self) -> Path:
        return self.storage_dir / "memory"

    @property
    def ledger_db(self) -> Path:
        return self.storage_dir / "season_ledger.sqlite"

    @property
    def flow_db(self) -> Path:
        return self.storage_dir / "flow_state.sqlite"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> CrewSettings:
        """Build settings from environment variables, ignoring unset values."""
        env = os.environ if env is None else env
        mapping = {
            "mode": "AGRI_CREW_MODE",
            "llm_timeout_s": "AGRI_CREW_LLM_TIMEOUT_S",
            "llm_max_retries": "AGRI_CREW_LLM_MAX_RETRIES",
            "max_iter": "AGRI_CREW_MAX_ITER",
            "max_rpm": "AGRI_CREW_MAX_RPM",
            "max_retry_limit": "AGRI_CREW_MAX_RETRY_LIMIT",
            "agent_max_execution_time_s": "AGRI_CREW_AGENT_MAX_EXECUTION_TIME_S",
            "tool_timeout_s": "AGRI_CREW_TOOL_TIMEOUT_S",
            "retry_max_attempts": "AGRI_CREW_RETRY_MAX_ATTEMPTS",
            "retry_initial_backoff_s": "AGRI_CREW_RETRY_INITIAL_BACKOFF_S",
            "retry_max_backoff_s": "AGRI_CREW_RETRY_MAX_BACKOFF_S",
            "cache_ttl_s": "AGRI_CREW_CACHE_TTL_S",
            "kamis_base_url": "AGRI_CREW_KAMIS_BASE_URL",
            "kmd_base_url": "AGRI_CREW_KMD_BASE_URL",
            "kalro_base_url": "AGRI_CREW_KALRO_BASE_URL",
            "api_bearer_token": "AGRI_CREW_API_BEARER_TOKEN",
            "storage_dir": "AGRI_CREW_STORAGE_DIR",
        }
        values: dict[str, Any] = {f: env[v] for f, v in mapping.items() if env.get(v)}

        if env.get("AGRI_CREW_MEMORY"):
            values["memory_enabled"] = env["AGRI_CREW_MEMORY"].lower() not in {"0", "false", "off"}
        if env.get("AGRI_CREW_VERBOSE"):
            values["verbose"] = env["AGRI_CREW_VERBOSE"].lower() in {"1", "true", "on"}
        if env.get("AGRI_CREW_EMBEDDER"):
            values["embedder"] = json.loads(env["AGRI_CREW_EMBEDDER"])
        if env.get("AGRI_CREW_MCP_SERVERS"):
            values["mcp_servers"] = json.loads(env["AGRI_CREW_MCP_SERVERS"])

        llm_model, manager_model = _resolve_models(env)
        if llm_model:
            values["llm_model"] = llm_model
            values["manager_llm_model"] = manager_model or llm_model
        if "embedder" not in values:
            values["embedder"] = _default_embedder(env)

        values.update(overrides)
        return cls.model_validate(values)


def _resolve_models(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    explicit = env.get("AGRI_CREW_LLM_MODEL")
    manager = env.get("AGRI_CREW_MANAGER_LLM_MODEL")
    if explicit and explicit.lower() == "offline":
        return None, None
    if explicit:
        return explicit, manager
    if env.get("ANTHROPIC_API_KEY"):
        return DEFAULT_ANTHROPIC_MODEL, manager or DEFAULT_ANTHROPIC_MANAGER_MODEL
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return DEFAULT_GEMINI_MODEL, manager
    return None, None


def _default_embedder(env: Mapping[str, str]) -> dict[str, Any] | None:
    """Pick an embedder that matches available credentials (memory needs one)."""
    google_key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
    if google_key:
        return {
            "provider": "google-generativeai",
            "config": {"api_key": google_key, "model_name": "gemini-embedding-001"},
        }
    if env.get("OPENAI_API_KEY"):
        return {"provider": "openai", "config": {"model_name": "text-embedding-3-small"}}
    if env.get("VOYAGE_API_KEY"):
        return {
            "provider": "voyageai",
            "config": {"api_key": env["VOYAGE_API_KEY"], "model": "voyage-3.5"},
        }
    return None
