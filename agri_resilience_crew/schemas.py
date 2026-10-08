"""Pydantic v2 models: crew inputs, tool I/O, intermediate task outputs and the playbook."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from agri_resilience_crew.reference import (
    CROPS,
    MARKETS,
    SUB_COUNTIES,
    canonical_county,
    canonical_crop,
)


class RiskLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class DataProvenance(BaseModel):
    source: Literal["KAMIS", "KMD", "KALRO", "NEMA", "MEMORY", "LEDGER", "MCP"]
    mode: Literal["mock", "live", "cache", "unavailable"]
    as_of: dt.date
    synthetic: bool = Field(description="True when the figures come from the synthetic dataset.")
    note: str = ""


def _county(value: str) -> str:
    county = canonical_county(value)
    if county is None:
        raise ValueError(f"'{value}' is not one of Kenya's 47 counties")
    return county


def _crop(value: str) -> str:
    crop = canonical_crop(value)
    if crop is None:
        raise ValueError(f"unsupported crop '{value}'; supported: {', '.join(CROPS)}")
    return crop


County = Annotated[str, AfterValidator(_county)]
Crop = Annotated[str, AfterValidator(_crop)]


# --------------------------------------------------------------------------- #
# Crew inputs
# --------------------------------------------------------------------------- #
class CrewInputs(BaseModel):
    """Validated kickoff inputs. Rejects unknown counties, sub-counties and crops."""

    model_config = ConfigDict(extra="forbid")

    county: County
    crops: list[str] = Field(min_length=1, max_length=6)
    sub_counties: list[str] = Field(default_factory=list, max_length=12)
    horizon_days: int = Field(default=90, ge=30, le=120)
    farm_size_ha: float = Field(default=1.0, gt=0, le=10_000)
    as_of: dt.date = Field(default_factory=dt.date.today)

    @field_validator("crops")
    @classmethod
    def _normalise_crops(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(_crop(c) for c in value))

    @model_validator(mode="after")
    def _check_sub_counties(self) -> CrewInputs:
        known = SUB_COUNTIES.get(self.county)
        if not self.sub_counties and known:
            self.sub_counties = list(known[:3])
        if known:
            lookup = {s.lower(): s for s in known}
            bad = [s for s in self.sub_counties if s.lower() not in lookup]
            if bad:
                raise ValueError(
                    f"unknown sub-county for {self.county}: {', '.join(bad)}; "
                    f"expected one of: {', '.join(known)}"
                )
            self.sub_counties = [lookup[s.lower()] for s in self.sub_counties]
        return self

    def as_crew_inputs(self) -> dict[str, str | int | float]:
        """Flattened, string-safe values for CrewAI ``{placeholder}`` interpolation."""
        return {
            "county": self.county,
            "crops": ", ".join(self.crops),
            "sub_counties": ", ".join(self.sub_counties) or "county-wide",
            "horizon_days": self.horizon_days,
            "farm_size_ha": self.farm_size_ha,
            "as_of": self.as_of.isoformat(),
            "markets": ", ".join(MARKETS),
        }


# --------------------------------------------------------------------------- #
# Tool argument schemas
# --------------------------------------------------------------------------- #
class KamisPriceQuery(BaseModel):
    crop: Crop = Field(description="Commodity, e.g. 'Irish Potatoes', 'Maize', 'Tomatoes'.")
    origin_county: County = Field(description="County the produce is sourced from.")
    markets: list[str] = Field(
        default_factory=lambda: list(MARKETS),
        description=f"Terminal markets to compare. Any of: {', '.join(MARKETS)}.",
    )
    lookback_days: int = Field(default=90, ge=14, le=365)

    @field_validator("markets")
    @classmethod
    def _known_markets(cls, value: list[str]) -> list[str]:
        unknown = [m for m in value if m not in MARKETS]
        if unknown:
            raise ValueError(f"unknown markets {unknown}; choose from {list(MARKETS)}")
        return value or list(MARKETS)


class KmdForecastQuery(BaseModel):
    county: County = Field(description="Kenyan county, e.g. 'Nakuru'.")
    sub_county: str | None = Field(default=None, description="Optional sub-county.")
    horizon_days: int = Field(default=90, ge=30, le=120)


class KalroSuitabilityQuery(BaseModel):
    county: County = Field(description="Kenyan county.")
    crop: Crop = Field(description="Candidate crop.")
    sub_county: str | None = Field(default=None)
    rainfall_anomaly_pct: float | None = Field(
        default=None,
        ge=-100,
        le=300,
        description="Seasonal rainfall anomaly (%) from the KMD outlook; omit to assume normal.",
    )


class MarginQuery(BaseModel):
    crop: Crop
    yield_t_per_ha: float = Field(gt=0, le=150)
    price_kes_per_kg: float = Field(gt=0, le=2000, description="Net farmgate-equivalent price.")
    area_ha: float = Field(default=1.0, gt=0, le=10_000)
    cost_kes_per_ha: float | None = Field(default=None, gt=0, le=5_000_000)


# --------------------------------------------------------------------------- #
# Tool results
# --------------------------------------------------------------------------- #
class MarketPriceSignal(BaseModel):
    market: str
    latest_price_kes_per_kg: float
    mean_7d: float
    mean_30d: float
    change_7d_pct: float
    change_30d_pct: float
    velocity_kes_per_kg_per_day: float = Field(description="OLS slope over the last 30 days.")
    acceleration_kes_per_kg_per_day2: float = Field(
        description="Slope of last 14 days minus slope of the prior 14 days, per day."
    )
    volatility_30d_pct: float = Field(description="Std-dev of daily log returns x sqrt(30).")
    zscore_vs_lookback: float
    road_distance_km: float
    transport_kes_per_kg: float
    fees_kes_per_kg: float
    spoilage_kes_per_kg: float
    farmgate_price_kes_per_kg: float
    net_margin_kes_per_kg: float
    net_margin_pct: float
    feasible: bool


class KamisPriceReport(BaseModel):
    crop: str
    origin_county: str
    as_of: dt.date
    signals: list[MarketPriceSignal]
    best_market: str | None
    provenance: DataProvenance


class ClimateOutlook(BaseModel):
    county: str
    sub_county: str | None
    season: str
    horizon_days: int
    window_start: dt.date
    window_end: dt.date
    prob_above_normal: float = Field(ge=0, le=1)
    prob_near_normal: float = Field(ge=0, le=1)
    prob_below_normal: float = Field(ge=0, le=1)
    normal_rainfall_mm: float
    expected_rainfall_mm: float
    rainfall_anomaly_pct: float
    spi_3month: float = Field(description="Standardised Precipitation Index (3-month).")
    expected_onset: dt.date | None
    expected_cessation: dt.date | None
    dry_spell_risk: RiskLevel
    flood_risk: RiskLevel
    heat_stress_days: int
    risk_level: RiskLevel
    provenance: DataProvenance

    @model_validator(mode="after")
    def _probabilities_sum_to_one(self) -> ClimateOutlook:
        total = self.prob_above_normal + self.prob_near_normal + self.prob_below_normal
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"tercile probabilities must sum to 1 (got {total:.3f})")
        return self


class PestAdvisory(BaseModel):
    pest: str
    crop: str
    risk: RiskLevel
    action: str


class ZoningConstraint(BaseModel):
    authority: Literal["NEMA", "WRA", "KFS", "County"]
    zone: str
    rule: str
    applies_to: str
    severity: Literal["advisory", "restricted", "prohibited"]


class SoilSuitabilityReport(BaseModel):
    county: str
    sub_county: str | None
    crop: str
    soil_ph: float
    soil_texture: str
    organic_carbon_pct: float
    altitude_m: float
    field_capacity_pct: float
    wilting_point_pct: float
    min_soil_moisture_pct_for_planting: float
    suitability_score: float = Field(ge=0, le=1)
    suitability_class: Literal["S1", "S2", "S3", "N"]
    limiting_factors: list[str]
    recommended_amendments: list[str]
    expected_yield_t_per_ha: float
    days_to_maturity: int
    pest_advisories: list[PestAdvisory]
    zoning_constraints: list[ZoningConstraint]
    provenance: list[DataProvenance]


class MarginResult(BaseModel):
    crop: str
    area_ha: float
    yield_t_per_ha: float
    price_kes_per_kg: float
    revenue_kes: float
    cost_kes: float
    gross_margin_kes: float
    gross_margin_pct: float
    breakeven_price_kes_per_kg: float


# --------------------------------------------------------------------------- #
# Intermediate task outputs (research phase)
# --------------------------------------------------------------------------- #
class ClimateImpactAssessment(BaseModel):
    county: str
    season: str
    risk_level: RiskLevel
    expected_rainfall_mm: float
    rainfall_anomaly_pct: float
    planting_window_start: dt.date | None
    planting_window_end: dt.date | None
    sub_county_notes: list[str]
    key_risks: list[str]
    summary: str


class CropMarketView(BaseModel):
    crop: str
    best_market: str | None
    latest_price_kes_per_kg: float
    velocity_kes_per_kg_per_day: float
    volatility_30d_pct: float
    net_margin_kes_per_kg: float
    trend: Literal["rising", "flat", "falling"]
    note: str


class MarketVolatilityAnalysis(BaseModel):
    origin_county: str
    crops: list[CropMarketView]
    summary: str


class CropFeasibility(BaseModel):
    crop: str
    suitability_class: Literal["S1", "S2", "S3", "N"]
    expected_yield_t_per_ha: float
    blocking_constraints: list[str]
    required_actions: list[str]


class AgronomicFeasibilityAudit(BaseModel):
    county: str
    crops: list[CropFeasibility]
    nema_compliant: bool
    summary: str


# --------------------------------------------------------------------------- #
# Final playbook
# --------------------------------------------------------------------------- #
class CropAction(StrEnum):
    PLANT = "plant"
    EXPAND = "expand"
    HOLD = "hold"
    SWITCH = "switch"
    AVOID = "avoid"


class CropPlan(BaseModel):
    crop: str
    action: CropAction
    switch_to: str | None = None
    planting_window_start: dt.date
    planting_window_end: dt.date
    harvest_window_start: dt.date
    harvest_window_end: dt.date
    target_market: str | None
    expected_yield_t_per_ha: float = Field(ge=0)
    expected_price_kes_per_kg: float = Field(ge=0)
    expected_revenue_kes_per_ha: float = Field(ge=0)
    expected_cost_kes_per_ha: float = Field(ge=0)
    expected_gross_margin_kes_per_ha: float
    expected_margin_pct: float
    confidence: float = Field(ge=0, le=1)
    rationale: str

    @model_validator(mode="after")
    def _coherent(self) -> CropPlan:
        if self.planting_window_end < self.planting_window_start:
            raise ValueError(f"{self.crop}: planting window ends before it starts")
        if self.harvest_window_start <= self.planting_window_start:
            raise ValueError(f"{self.crop}: harvest must start after planting")
        if self.harvest_window_end < self.harvest_window_start:
            raise ValueError(f"{self.crop}: harvest window ends before it starts")
        if self.action is CropAction.SWITCH and not self.switch_to:
            raise ValueError(f"{self.crop}: action 'switch' requires switch_to")
        return self


class RiskMitigation(BaseModel):
    risk: str
    likelihood: RiskLevel
    impact: RiskLevel
    mitigation: str
    owner: str = "farm manager"


class PricingAction(BaseModel):
    week_start: dt.date
    week_end: dt.date
    crop: str
    action: str
    trigger: str


class AgriStrategyOutput(BaseModel):
    """Typed 90-day planting, harvesting and pricing playbook (final task output)."""

    county: str
    sub_counties: list[str]
    season: str
    as_of: dt.date
    horizon_days: int
    climate_risk_level: RiskLevel
    climate_summary: str
    market_summary: str
    crop_plans: list[CropPlan] = Field(min_length=1)
    risk_mitigations: list[RiskMitigation] = Field(min_length=1)
    pricing_playbook: list[PricingAction]
    compliance_flags: list[str]
    conflicts_resolved: list[str]
    data_provenance: list[DataProvenance]
    engine: Literal["crewai", "deterministic"] = "crewai"
    disclaimer: str = (
        "Decision support only. Validate against current KAMIS, KMD and county extension "
        "advisories before committing capital."
    )

    @model_validator(mode="after")
    def _within_horizon(self) -> AgriStrategyOutput:
        limit = self.as_of + dt.timedelta(days=self.horizon_days)
        for plan in self.crop_plans:
            if plan.action in {CropAction.PLANT, CropAction.EXPAND, CropAction.SWITCH} and (
                plan.planting_window_start > limit
            ):
                raise ValueError(f"{plan.crop}: planting starts after the {self.horizon_days}-day horizon")
        return self
