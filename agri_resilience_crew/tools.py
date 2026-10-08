"""Custom CrewAI tools over Kenyan public agricultural datasets.

Every tool validates its arguments with a Pydantic ``args_schema``. Data I/O is
time-boxed and retried with exponential backoff in :class:`DataGateway`. An
unrecoverable data gap comes back as a :class:`ToolFailure`, so the agent sees
a clear message and the framework records the failure.
"""

from __future__ import annotations

import datetime as dt
import json

from crewai.tools import BaseTool, ToolFailure, tool
from crewai.tools.tool_failure import ToolFailureReason
from pydantic import ConfigDict, Field

from agri_resilience_crew import analytics
from agri_resilience_crew.reference import COUNTIES, CROPS, MARKETS, SUB_COUNTIES, road_distance_km
from agri_resilience_crew.schemas import (
    KalroSuitabilityQuery,
    KamisPriceQuery,
    KmdForecastQuery,
    MarginQuery,
)
from agri_resilience_crew.sources import DataGateway, DataUnavailableError


class _GatewayTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway: DataGateway = Field(exclude=True)
    as_of: dt.date = Field(default_factory=dt.date.today)

    @staticmethod
    def _unavailable(exc: DataUnavailableError) -> ToolFailure:
        return ToolFailure(
            message=f"Data source unavailable: {exc}. State the gap explicitly; do not invent figures.",
            reason=ToolFailureReason.TOOL_REPORTED,
            code="data_unavailable",
            retryable=True,
        )


class KAMISPriceFetcherTool(_GatewayTool):
    name: str = "kamis_price_fetcher"
    description: str = (
        "Fetch daily wholesale prices (KES/kg) from KAMIS for one crop across the Nairobi "
        "Wakulima, Mombasa Kongowea, Kisumu Kibuye and Eldoret markets. Returns price velocity "
        "(30-day OLS slope), acceleration, 30-day volatility, z-score and the haul-to-market "
        "margin over farmgate after transport, cess/fees and spoilage from the origin county."
    )
    args_schema: type[KamisPriceQuery] = KamisPriceQuery

    def _run(self, **kwargs: object) -> str | ToolFailure:
        q = KamisPriceQuery.model_validate(kwargs)
        try:
            report = self.gateway.kamis_report(
                q.crop, q.origin_county, q.markets, self.as_of, q.lookback_days
            )
        except DataUnavailableError as exc:
            return self._unavailable(exc)
        return report.model_dump_json()


class KMDWeatherScraperTool(_GatewayTool):
    name: str = "kmd_climate_outlook"
    description: str = (
        "Query the KMD seasonal outlook for a county (optionally a sub-county): tercile "
        "probabilities, expected vs normal rainfall (mm), anomaly %, 3-month SPI, onset and "
        "cessation estimates, dry-spell/flood risk, heat-stress days and an overall risk level."
    )
    args_schema: type[KmdForecastQuery] = KmdForecastQuery

    def _run(self, **kwargs: object) -> str | ToolFailure:
        q = KmdForecastQuery.model_validate(kwargs)
        try:
            outlook = self.gateway.climate_outlook(q.county, q.sub_county, self.as_of, q.horizon_days)
        except DataUnavailableError as exc:
            return self._unavailable(exc)
        return outlook.model_dump_json()


class KALROSoilSuitabilityTool(_GatewayTool):
    name: str = "kalro_soil_suitability"
    description: str = (
        "Match county/sub-county soil (pH, texture, organic carbon, drainage, altitude) against "
        "a crop's agronomic requirements. Returns an FAO-style suitability class (S1-N), "
        "limiting factors, soil-moisture planting threshold, expected yield, soil amendments, "
        "pest advisories (e.g. fall armyworm, late blight, desert locust) and NEMA/WRA/KFS "
        "zoning constraints. Pass rainfall_anomaly_pct from the KMD outlook when known."
    )
    args_schema: type[KalroSuitabilityQuery] = KalroSuitabilityQuery

    def _run(self, **kwargs: object) -> str | ToolFailure:
        q = KalroSuitabilityQuery.model_validate(kwargs)
        try:
            report = self.gateway.soil_suitability(
                q.county, q.sub_county, q.crop, self.as_of, q.rainfall_anomaly_pct
            )
        except DataUnavailableError as exc:
            return self._unavailable(exc)
        return report.model_dump_json()


class GrossMarginCalculatorTool(BaseTool):
    name: str = "gross_margin_calculator"
    description: str = (
        "Compute revenue, cost of production, gross margin (KES) and break-even price for a "
        "crop given yield (t/ha), price (KES/kg) and area. Use this instead of mental arithmetic."
    )
    args_schema: type[MarginQuery] = MarginQuery

    def _run(self, **kwargs: object) -> str:
        q = MarginQuery.model_validate(kwargs)
        return analytics.gross_margin(
            q.crop,
            yield_t_per_ha=q.yield_t_per_ha,
            price_kes_per_kg=q.price_kes_per_kg,
            area_ha=q.area_ha,
            cost_kes_per_ha=q.cost_kes_per_ha,
        ).model_dump_json()


@tool("kenya_agri_reference")
def kenya_agri_reference(county: str = "") -> str:
    """List supported crops, terminal markets, Kenya's 47 counties and known sub-counties.
    Pass a county name to also get its sub-counties and road distances to each market."""
    payload: dict[str, object] = {
        "crops": list(CROPS),
        "markets": list(MARKETS),
        "counties": list(COUNTIES),
    }
    if county and county in COUNTIES:
        payload["sub_counties"] = list(SUB_COUNTIES.get(county, ()))
        payload["road_distance_km"] = {m: road_distance_km(county, m) for m in MARKETS}
    return json.dumps(payload)


def build_tools(gateway: DataGateway, as_of: dt.date) -> dict[str, BaseTool]:
    return {
        "kamis": KAMISPriceFetcherTool(gateway=gateway, as_of=as_of),
        "kmd": KMDWeatherScraperTool(gateway=gateway, as_of=as_of),
        "kalro": KALROSoilSuitabilityTool(gateway=gateway, as_of=as_of),
        "margin": GrossMarginCalculatorTool(),
        "reference": kenya_agri_reference,
    }
