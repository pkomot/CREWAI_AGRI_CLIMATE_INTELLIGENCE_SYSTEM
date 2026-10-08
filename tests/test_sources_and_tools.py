from __future__ import annotations

import json
import time

import httpx
import pytest
from conftest import AS_OF
from crewai.tools import ToolFailure

from agri_resilience_crew import synthetic
from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.resilience import (
    BackoffPolicy,
    ToolTimeoutError,
    TTLCache,
    retry_with_backoff,
    run_with_timeout,
)
from agri_resilience_crew.sources import DataGateway, DataUnavailableError
from agri_resilience_crew.tools import build_tools

LIVE = dict(
    mode="live",
    kamis_base_url="https://kamis.example/api",
    kmd_base_url="https://kmd.example/api",
    kalro_base_url="https://kalro.example/api",
    retry_initial_backoff_s=0.01,
    retry_max_backoff_s=0.02,
    memory_enabled=False,
)


# --------------------------------------------------------------------------- #
# Resilience primitives
# --------------------------------------------------------------------------- #
def test_retry_backs_off_exponentially_then_succeeds() -> None:
    attempts, sleeps = [], []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    policy = BackoffPolicy(max_attempts=3, initial_s=1.0, jitter=0.0)
    assert retry_with_backoff(flaky, policy, label="t", sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_retry_does_not_retry_client_errors() -> None:
    request = httpx.Request("GET", "https://x")
    err = httpx.HTTPStatusError("404", request=request, response=httpx.Response(404, request=request))
    calls = []

    def fail() -> None:
        calls.append(1)
        raise err

    with pytest.raises(httpx.HTTPStatusError):
        retry_with_backoff(fail, BackoffPolicy(max_attempts=5), label="t", sleep=lambda _: None)
    assert len(calls) == 1


def test_run_with_timeout_unblocks_caller() -> None:
    start = time.monotonic()
    with pytest.raises(ToolTimeoutError):
        run_with_timeout(lambda: time.sleep(2), timeout_s=0.1)
    assert time.monotonic() - start < 1.0


def test_ttl_cache_expiry_and_stale_reads() -> None:
    now = [0.0]
    cache: TTLCache[int] = TTLCache(10, clock=lambda: now[0])
    cache.set("k", 1)
    now[0] = 11
    assert cache.get("k") is None
    assert cache.get("k", allow_stale=True) == 1


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #
def test_mock_data_is_deterministic(gateway: DataGateway) -> None:
    a = gateway.kamis_report("Maize", "Nakuru", ["Nairobi Wakulima", "Eldoret Main"], AS_OF, 90)
    b = DataGateway(gateway.settings).kamis_report(
        "Maize", "Nakuru", ["Nairobi Wakulima", "Eldoret Main"], AS_OF, 90
    )
    assert a == b
    assert a.provenance.synthetic and a.provenance.mode == "mock"


def test_terciles_sum_to_one_for_every_county() -> None:
    for county in synthetic.COUNTY_ZONE:
        assert sum(synthetic.tercile_forecast(county, AS_OF)) == pytest.approx(1.0, abs=0.002)


def test_live_mode_requires_real_endpoints() -> None:
    with pytest.raises(ValueError, match="still mocked"):
        CrewSettings.from_env({"AGRI_CREW_MODE": "live"})


def _live_gateway(handler) -> DataGateway:
    settings = CrewSettings.from_env({}, **LIVE)
    return DataGateway(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_live_outlook_parses_adapter_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/outlook"
        return httpx.Response(200, json={"prob_above": 0.2, "prob_near": 0.3, "prob_below": 0.5})

    outlook = _live_gateway(handler).climate_outlook("Nakuru", None, AS_OF, 90)
    assert outlook.prob_below_normal == 0.5
    assert outlook.rainfall_anomaly_pct == pytest.approx(-18.0)
    assert outlook.provenance.mode == "live" and not outlook.provenance.synthetic


def test_live_failure_serves_stale_cache_then_errors_without_cache() -> None:
    state = {"up": True, "calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if not state["up"]:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "soil_ph": 5.6,
                "soil_texture": "loam",
                "organic_carbon_pct": 2.0,
                "drainage": "good",
                "altitude_m": 2300,
            },
        )

    gw = _live_gateway(handler)
    gw.soil_suitability("Nakuru", "Molo", "Irish Potatoes", AS_OF, 0)
    gw._cache._ttl_s = -1  # expire everything: next read must hit the network
    state["up"] = False
    report = gw.soil_suitability("Nakuru", "Molo", "Irish Potatoes", AS_OF, 0)
    assert report.provenance[0].mode == "cache"
    with pytest.raises(DataUnavailableError):
        gw.soil_suitability("Meru", None, "Maize", AS_OF, 0)
    assert state["calls"] >= 1 + 3 + 3  # first fetch, then 3 retries for each failure


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_tools_return_json_payloads(gateway: DataGateway) -> None:
    tools = build_tools(gateway, AS_OF)
    kmd = json.loads(tools["kmd"].run(county="Nakuru", sub_county="Molo"))
    assert kmd["season"] == "OND Short Rains 2026"
    kamis = json.loads(tools["kamis"].run(crop="potatoes", origin_county="Nakuru"))
    assert {s["market"] for s in kamis["signals"]} == {
        "Nairobi Wakulima",
        "Mombasa Kongowea",
        "Kisumu Kibuye",
        "Eldoret Main",
    }
    kalro = json.loads(
        tools["kalro"].run(county="Nakuru", sub_county="Naivasha", crop="Tomatoes", rainfall_anomaly_pct=-10)
    )
    assert any("Naivasha" in z["zone"] for z in kalro["zoning_constraints"])
    assert (
        json.loads(tools["margin"].run(crop="Maize", yield_t_per_ha=4, price_kes_per_kg=40))[
            "gross_margin_kes"
        ]
        == 50_000
    )
    assert "Molo" in tools["reference"].run(county="Nakuru")


def test_tool_reports_unavailable_data_as_tool_failure() -> None:
    gw = _live_gateway(lambda request: httpx.Response(503))
    result = build_tools(gw, AS_OF)["kmd"].run(county="Nakuru")
    assert isinstance(result, ToolFailure) and result.code == "data_unavailable"
