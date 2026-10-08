from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
sys.path.insert(0, str(Path(__file__).parent))

from agri_resilience_crew import engine  # noqa: E402
from agri_resilience_crew.config import CrewSettings  # noqa: E402
from agri_resilience_crew.schemas import CrewInputs  # noqa: E402
from agri_resilience_crew.sources import DataGateway  # noqa: E402

AS_OF = dt.date(2026, 10, 7)


@pytest.fixture
def settings(tmp_path: Path) -> CrewSettings:
    return CrewSettings.from_env({}, storage_dir=tmp_path / "store", memory_enabled=False)


@pytest.fixture
def gateway(settings: CrewSettings):
    gw = DataGateway(settings)
    yield gw
    gw.close()


@pytest.fixture
def nakuru_inputs() -> CrewInputs:
    return CrewInputs(county="Nakuru", crops=["Irish Potatoes", "Maize"], as_of=AS_OF)


@pytest.fixture
def bundle(gateway: DataGateway, nakuru_inputs: CrewInputs) -> engine.ResearchBundle:
    return engine.gather_research(gateway, nakuru_inputs)
