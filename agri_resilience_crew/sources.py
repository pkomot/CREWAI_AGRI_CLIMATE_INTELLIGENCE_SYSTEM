"""Data gateway for KAMIS (prices), KMD (climate) and KALRO/NEMA (soils, zoning).

Each source has two backends:

* ``mock``: deterministic synthetic data from :mod:`agri_resilience_crew.synthetic`.
* ``live``: HTTP JSON adapters, called with a hard timeout, exponential-backoff
  retry and a last-known-good cache. KAMIS, KMD and KALRO do not publish stable
  public JSON APIs, so ``live`` expects a thin adapter service (scraper or
  data-sharing endpoint) that implements the contracts documented in the README.

Raw data comes from the backend. Every derived metric is computed locally in
:mod:`agri_resilience_crew.analytics`, so mock and live runs share one analytics path.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

from agri_resilience_crew import analytics, synthetic
from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.reference import CROPS, MARKETS
from agri_resilience_crew.resilience import (
    BackoffPolicy,
    TTLCache,
    retry_with_backoff,
    run_with_timeout,
)
from agri_resilience_crew.schemas import (
    ClimateOutlook,
    DataProvenance,
    KamisPriceReport,
    PestAdvisory,
    RiskLevel,
    SoilSuitabilityReport,
    ZoningConstraint,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DataUnavailableError(RuntimeError):
    """Raised when a source failed and no cached data exists."""


class DataGateway:
    """Thread-safe facade over the three public-data sources."""

    def __init__(self, settings: CrewSettings, *, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._policy = BackoffPolicy(
            max_attempts=settings.retry_max_attempts,
            initial_s=settings.retry_initial_backoff_s,
            max_s=settings.retry_max_backoff_s,
        )
        self._cache: TTLCache[Any] = TTLCache(settings.cache_ttl_s)
        headers = {"Accept": "application/json", "User-Agent": "agri-resilience-crew/0.1"}
        if settings.api_bearer_token:
            headers["Authorization"] = f"Bearer {settings.api_bearer_token}"
        self._client = client or httpx.Client(timeout=settings.tool_timeout_s, headers=headers)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ #
    # Core fetch: timeout + backoff + cache fallback
    # ------------------------------------------------------------------ #
    def _fetch(self, key: str, label: str, loader: Callable[[], T]) -> tuple[T, str]:
        """Return (data, mode) where mode is 'mock', 'live' or 'cache'."""
        mode = self.settings.mode
        fresh = self._cache.get(key)
        if fresh is not None:
            return fresh, mode

        def guarded() -> T:
            return run_with_timeout(loader, self.settings.tool_timeout_s)

        try:
            data = retry_with_backoff(guarded, self._policy, label=label)
        except Exception as exc:
            stale = self._cache.get(key, allow_stale=True)
            if stale is not None:
                logger.warning("%s failed (%r); serving last-known-good cache", label, exc)
                return stale, "cache"
            raise DataUnavailableError(f"{label} unavailable: {exc}") from exc
        self._cache.set(key, data)
        return data, mode

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _provenance(self, source: str, mode: str, as_of: dt.date, note: str = "") -> DataProvenance:
        return DataProvenance(
            source=source,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            as_of=as_of,
            synthetic=self.settings.mode == "mock",
            note=note,
        )

    # ------------------------------------------------------------------ #
    # KAMIS
    # ------------------------------------------------------------------ #
    def price_series(self, crop: str, market: str, end: dt.date, days: int) -> tuple[list[float], str]:
        def load() -> list[float]:
            if self.settings.mode == "mock":
                return [p for _, p in synthetic.price_series(crop, market, end, days)]
            rows = self._get_json(
                f"{self.settings.kamis_base_url}/prices",
                {
                    "commodity": crop,
                    "market": market,
                    "from": (end - dt.timedelta(days=days - 1)).isoformat(),
                    "to": end.isoformat(),
                },
            )
            prices = [float(r["wholesale_kes_per_kg"]) for r in rows]
            if len(prices) < 30:
                raise DataUnavailableError(f"KAMIS returned {len(prices)} rows for {crop}@{market}")
            return prices

        return self._fetch(f"kamis|{crop}|{market}|{end}|{days}", f"KAMIS {crop}@{market}", load)

    def kamis_report(
        self, crop: str, origin_county: str, markets: list[str], end: dt.date, lookback_days: int
    ) -> KamisPriceReport:
        modes: set[str] = set()
        series: dict[str, list[float]] = {}
        for market in markets:
            try:
                prices, mode = self.price_series(crop, market, end, lookback_days)
            except DataUnavailableError as exc:
                logger.warning("skipping %s: %s", market, exc)
                continue
            series[market] = prices
            modes.add(mode)
        if not series:
            raise DataUnavailableError(f"no KAMIS price data for {crop}")

        reference = series.get("Nairobi Wakulima") or next(iter(series.values()))
        farmgate = (
            reference[-1]
            * CROPS[crop].farmgate_share
            * (1 + synthetic.origin_price_adjustment(origin_county, crop))
        )
        signals = [
            analytics.price_signal(
                crop=crop,
                market=market,
                origin_county=origin_county,
                prices=prices,
                farmgate_price=farmgate,
            )
            for market, prices in series.items()
        ]
        feasible = [s for s in signals if s.feasible]
        best = max(feasible or signals, key=lambda s: s.net_margin_kes_per_kg)
        mode = "cache" if "cache" in modes else self.settings.mode
        return KamisPriceReport(
            crop=crop,
            origin_county=origin_county,
            as_of=end,
            signals=signals,
            best_market=best.market if best.feasible else None,
            provenance=self._provenance(
                "KAMIS", mode, end, note=f"{len(series)}/{len(markets)} markets, {lookback_days}d window"
            ),
        )

    # ------------------------------------------------------------------ #
    # KMD
    # ------------------------------------------------------------------ #
    def climate_outlook(
        self, county: str, sub_county: str | None, start: dt.date, horizon_days: int
    ) -> ClimateOutlook:
        def load() -> dict[str, Any]:
            if self.settings.mode == "mock":
                above, near, below = synthetic.tercile_forecast(county, start)
                return {"prob_above": above, "prob_near": near, "prob_below": below}
            return self._get_json(
                f"{self.settings.kmd_base_url}/outlook",
                {
                    "county": county,
                    "sub_county": sub_county or "",
                    "start": start.isoformat(),
                    "days": horizon_days,
                },
            )

        raw, mode = self._fetch(
            f"kmd|{county}|{sub_county}|{start}|{horizon_days}", f"KMD outlook {county}", load
        )
        above, near, below = float(raw["prob_above"]), float(raw["prob_near"]), float(raw["prob_below"])
        normal = synthetic.normal_rainfall_mm(county, start, horizon_days, sub_county)
        anomaly = float(raw.get("rainfall_anomaly_pct", round((above - below) * 60, 1)))
        expected = float(raw.get("expected_rainfall_mm", round(normal * (1 + anomaly / 100), 1)))
        spi = round(anomaly / 100 / synthetic.CV_SEASONAL_RAIN, 2)
        overall, dry, flood = analytics.classify_climate_risk(anomaly, spi)
        onset, cessation = _onset_cessation(county, sub_county, start, horizon_days, above, below)

        heat_base = 25 if synthetic.is_asal(county) else 3
        heat = (
            heat_base
            + (8 if anomaly < -15 else 0)
            + synthetic.seeded_rng("heat", county, start).randint(0, 4)
        )
        return ClimateOutlook(
            county=county,
            sub_county=sub_county,
            season=synthetic.season_label(start),
            horizon_days=horizon_days,
            window_start=start,
            window_end=start + dt.timedelta(days=horizon_days - 1),
            prob_above_normal=above,
            prob_near_normal=near,
            prob_below_normal=below,
            normal_rainfall_mm=normal,
            expected_rainfall_mm=expected,
            rainfall_anomaly_pct=anomaly,
            spi_3month=spi,
            expected_onset=onset,
            expected_cessation=cessation,
            dry_spell_risk=dry,
            flood_risk=flood,
            heat_stress_days=heat,
            risk_level=overall,
            provenance=self._provenance("KMD", mode, start),
        )

    # ------------------------------------------------------------------ #
    # KALRO + NEMA
    # ------------------------------------------------------------------ #
    def soil_suitability(
        self,
        county: str,
        sub_county: str | None,
        crop: str,
        start: dt.date,
        rainfall_anomaly_pct: float | None,
    ) -> SoilSuitabilityReport:
        def load() -> dict[str, Any]:
            if self.settings.mode == "mock":
                return synthetic.soil_profile(county, sub_county)
            return self._get_json(
                f"{self.settings.kalro_base_url}/soils",
                {"county": county, "sub_county": sub_county or ""},
            )

        soil, mode = self._fetch(f"kalro|{county}|{sub_county}", f"KALRO soils {county}", load)
        profile = CROPS[crop]
        anomaly = rainfall_anomaly_pct if rainfall_anomaly_pct is not None else 0.0
        season_rain = synthetic.normal_rainfall_mm(county, start, profile.days_to_maturity, sub_county) * (
            1 + anomaly / 100
        )
        score, klass, limiting = analytics.suitability(
            crop,
            soil_ph=float(soil["soil_ph"]),
            altitude_m=float(soil["altitude_m"]),
            season_rainfall_mm=season_rain,
            drainage=str(soil["drainage"]),
        )
        texture = str(soil["soil_texture"])
        fc, wp = synthetic.TEXTURE_MOISTURE.get(texture, synthetic.TEXTURE_MOISTURE["loam"])

        amendments: list[str] = []
        ph = float(soil["soil_ph"])
        if ph < profile.ph_min:
            lime = round((profile.ph_min - ph + 0.3) * 2.0, 1)
            amendments.append(
                f"Apply ~{lime} t/ha agricultural lime 4-6 weeks before planting (confirm with a soil test)."
            )
        if ph > profile.ph_max:
            amendments.append("Use acidifying fertilisers (e.g. ammonium sulphate) and organic matter.")
        if float(soil["organic_carbon_pct"]) < 1.0:
            amendments.append("Incorporate 5-10 t/ha well-decomposed manure to raise organic carbon.")
        if soil["drainage"] == "poor" and profile.waterlogging_sensitive:
            amendments.append("Plant on raised beds or ridges; open drainage furrows.")

        pests = [
            PestAdvisory(
                pest=pest,
                crop=crop,
                risk=RiskLevel(synthetic.pest_risk(pest, county, anomaly)),
                action=synthetic.PEST_ACTIONS[pest],
            )
            for pest in profile.key_pests
        ]
        if synthetic.pest_risk("desert locust", county, anomaly) != "low":
            pests.append(
                PestAdvisory(
                    pest="desert locust",
                    crop=crop,
                    risk=RiskLevel.MODERATE,
                    action=synthetic.PEST_ACTIONS["desert locust"],
                )
            )

        zoning = [
            ZoningConstraint(authority=a, zone=z, rule=r, applies_to=t, severity=s)  # type: ignore[arg-type]
            for a, z, r, t, s in (
                *synthetic.UNIVERSAL_ZONING,
                *synthetic.SITE_ZONING.get(sub_county or "", ()),
            )
        ]
        return SoilSuitabilityReport(
            county=county,
            sub_county=sub_county,
            crop=crop,
            soil_ph=ph,
            soil_texture=texture,
            organic_carbon_pct=float(soil["organic_carbon_pct"]),
            altitude_m=float(soil["altitude_m"]),
            field_capacity_pct=fc,
            wilting_point_pct=wp,
            min_soil_moisture_pct_for_planting=round(wp + 0.5 * (fc - wp), 1),
            suitability_score=score,
            suitability_class=klass,  # type: ignore[arg-type]
            limiting_factors=limiting,
            recommended_amendments=amendments,
            expected_yield_t_per_ha=analytics.expected_yield(crop, score, anomaly),
            days_to_maturity=profile.days_to_maturity,
            pest_advisories=pests,
            zoning_constraints=zoning,
            provenance=[
                self._provenance("KALRO", mode, start),
                self._provenance(
                    "NEMA", "mock", start, note="illustrative zoning screen; confirm with county NEMA office"
                ),
            ],
        )


RAINY_MONTH_MM = 50.0  # a month at or above this normal counts as part of the rainy season


def _onset_cessation(
    county: str,
    sub_county: str | None,
    start: dt.date,
    horizon_days: int,
    above: float,
    below: float,
) -> tuple[dt.date | None, dt.date | None]:
    """Estimate rainy-season onset and cessation inside the forecast window."""
    monthly = synthetic.monthly_normals(county, sub_county)
    threshold = RAINY_MONTH_MM
    end = start + dt.timedelta(days=horizon_days - 1)
    shift = 10 if below >= 0.45 else -5 if above >= 0.45 else 0

    rainy: list[dt.date] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        if monthly[cursor.month - 1] >= threshold:
            rainy.append(cursor)
        cursor = (cursor + dt.timedelta(days=32)).replace(day=1)
    if not rainy:
        return None, None
    onset = rainy[0] + dt.timedelta(days=7 + shift)
    last = rainy[-1]
    month_end = (last + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    # A dry outlook (positive shift) ends the season early; a wet one extends it.
    return onset, month_end - dt.timedelta(days=shift)


__all__ = ["DataGateway", "DataUnavailableError", "MARKETS"]
