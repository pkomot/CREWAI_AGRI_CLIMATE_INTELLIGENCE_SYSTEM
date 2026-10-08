"""Crew memory: CrewAI unified ``Memory`` plus a SQLite season ledger.

CrewAI 1.x replaced the separate short-term, long-term and entity memories with
a single LanceDB-backed ``Memory`` that is organised by hierarchical scopes.
The spec's three memory tiers map onto it as follows:

* Short-term (cross-task context): both crews share one ``Memory`` instance;
  CrewAI auto-saves task results under ``/agri-resilience`` and agents recall
  them in later tasks, including across the research -> strategy phase boundary.
* Entity: explicit records under ``/entities/{crops,counties,markets}/<slug>``,
  seeded from the tool evidence before kickoff.
* Long-term (cross-season learning): ``/seasons/<county>`` records, plus the
  :class:`SeasonLedger` SQLite table. The ledger keeps every playbook and feeds
  a compact history back into the prompts on the next run.

Memory needs an embedder. If none is configured, vector memory is disabled with
a warning, while the SQLite ledger keeps working.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agri_resilience_crew.config import CrewSettings
from agri_resilience_crew.schemas import AgriStrategyOutput

if TYPE_CHECKING:
    from crewai.memory.unified_memory import Memory

    from agri_resilience_crew.engine import ResearchBundle

logger = logging.getLogger(__name__)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_memory(settings: CrewSettings, llm: Any) -> Memory | None:
    """Create the unified Memory, or None when disabled or no embedder is available."""
    if not settings.memory_enabled:
        return None
    if settings.embedder is None:
        logger.warning(
            "crew memory disabled: no embedder configured. Set GEMINI_API_KEY/GOOGLE_API_KEY, "
            "OPENAI_API_KEY, VOYAGE_API_KEY or AGRI_CREW_EMBEDDER (Anthropic has no embeddings API)."
        )
        return None
    from crewai.memory.unified_memory import Memory

    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    return Memory(
        llm=llm,
        storage=str(settings.memory_dir / "lancedb"),
        embedder=settings.embedder,
        root_scope="/agri-resilience",
        recency_half_life_days=120,  # roughly one agricultural season
    )


def seed_entity_memory(memory: Memory, bundle: ResearchBundle, run_id: str) -> None:
    """Write explicit entity facts so recall works without LLM scope inference."""
    inputs = bundle.inputs
    records: list[dict[str, Any]] = []
    for ev in bundle.crops:
        soil = ev.soil
        records.append(
            {
                "content": (
                    f"{ev.crop} in {inputs.county} ({soil.sub_county or 'county-wide'}): suitability "
                    f"{soil.suitability_class} ({soil.suitability_score:.2f}), soil pH {soil.soil_ph}, "
                    f"expected yield {soil.expected_yield_t_per_ha} t/ha as of {inputs.as_of}."
                ),
                "scope": f"/entities/crops/{slug(ev.crop)}",
                "categories": ["crop", "agronomy", slug(inputs.county)],
            }
        )
        if ev.market and ev.market.best_market:
            sig = next(s for s in ev.market.signals if s.market == ev.market.best_market)
            records.append(
                {
                    "content": (
                        f"{ev.crop} best terminal market from {inputs.county} on {inputs.as_of}: "
                        f"{sig.market} at KES {sig.latest_price_kes_per_kg}/kg, velocity "
                        f"{sig.velocity_kes_per_kg_per_day:+} KES/kg/day, haul margin KES {sig.net_margin_kes_per_kg}/kg."
                    ),
                    "scope": f"/entities/markets/{slug(sig.market)}",
                    "categories": ["market", "price", slug(ev.crop)],
                }
            )
    for c in bundle.climate:
        records.append(
            {
                "content": (
                    f"{c.season} outlook for {c.sub_county or c.county}, {c.county}: "
                    f"{c.rainfall_anomaly_pct:+.0f}% rainfall anomaly, {c.risk_level.value} risk."
                ),
                "scope": f"/entities/counties/{slug(c.county)}/{slug(c.sub_county or 'all')}",
                "categories": ["climate", slug(c.county)],
            }
        )
    for rec in records:
        try:
            memory.remember(
                rec["content"],
                scope=rec["scope"],
                categories=rec["categories"],
                importance=0.7,
                metadata={"run_id": run_id, "as_of": inputs.as_of.isoformat()},
                source=f"run:{run_id}",
            )
        except Exception:  # noqa: BLE001 - memory is an enhancement, never a blocker
            logger.warning("entity memory write failed for %s", rec["scope"], exc_info=True)
            return


def remember_playbook(memory: Memory, output: AgriStrategyOutput) -> None:
    plans = "; ".join(
        f"{p.crop}: {p.action.value} ({p.expected_margin_pct:.0f}% margin)" for p in output.crop_plans
    )
    try:
        memory.remember(
            f"{output.season} playbook for {output.county} (climate risk {output.climate_risk_level.value}): {plans}.",
            scope=f"/seasons/{slug(output.county)}",
            categories=["playbook", "season", slug(output.county)],
            importance=0.8,
            metadata={"as_of": output.as_of.isoformat(), "engine": output.engine},
        )
    except Exception:  # noqa: BLE001
        logger.warning("long-term memory write failed", exc_info=True)


class SeasonLedger:
    """SQLite store of every playbook, used for cross-season trend context."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS playbooks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    county TEXT NOT NULL,
                    season TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    climate_risk TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS ix_playbooks_county ON playbooks(county, as_of)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, run_id: str, output: AgriStrategyOutput) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO playbooks(run_id, county, season, as_of, engine, climate_risk, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    output.county,
                    output.season,
                    output.as_of.isoformat(),
                    output.engine,
                    output.climate_risk_level.value,
                    output.model_dump_json(),
                ),
            )

    def history(self, county: str, limit: int = 4) -> list[AgriStrategyOutput]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT payload FROM playbooks WHERE county = ? ORDER BY as_of DESC, id DESC LIMIT ?",
                (county, limit),
            ).fetchall()
        return [AgriStrategyOutput.model_validate(json.loads(r[0])) for r in rows]

    def prompt_context(self, county: str, crops: list[str]) -> str:
        """Compact, brace-free history string safe for CrewAI prompt interpolation."""
        past = self.history(county)
        if not past:
            return "No prior playbooks for this county in the season ledger."
        lines = []
        for p in past:
            relevant = [c for c in p.crop_plans if c.crop in crops] or p.crop_plans
            plan = ", ".join(
                f"{c.crop} {c.action.value} at KES {c.expected_price_kes_per_kg:.0f}/kg ({c.expected_margin_pct:.0f}% margin)"
                for c in relevant
            )
            lines.append(f"{p.as_of} {p.season} [{p.climate_risk_level.value} climate risk]: {plan}")
        return "Prior playbooks (most recent first): " + " | ".join(lines)
