# CREWAI_AGRI_CLIMATE_INTELLIGENCE_SYSTEM

**AgriResilience-Crew** is a CrewAI multi-agent system for climate-adaptive agricultural market
intelligence in Kenya. Four agents combine KMD climate outlooks, KAMIS wholesale prices and
KALRO soil suitability with NEMA/WRA/KFS zoning. From these they produce a typed **90-day
planting, harvesting and pricing playbook** (`AgriStrategyOutput`) for any of Kenya's 47 counties.

```
                  ┌──────────────── Phase 1: research_crew (Process.sequential) ──────────────┐
CrewInputs ──►    │  T1 Climate Impact (kmd_analyst)        ─┐ async_execution=True (parallel) │
 (validated)      │  T2 Market Price Volatility (kamis_analyst) ┘                              │
                  │  T3 Agronomic Feasibility & NEMA Audit (kalro_specialist, context T1+T2)  │
                  └───────────────────────────────────────┬────────────────────────────────────┘
                     risk router: high/severe climate ──► adds a contingency mandate
                  ┌─────────────── Phase 2: strategy_crew (Process.hierarchical) ──────────────┐
                  │  manager_agent = agri_strategy_lead ── delegates follow-ups to specialists  │
                  │  T4 Strategy Synthesis  → output_pydantic=AgriStrategyOutput + guardrail    │
                  └───────────────────────────────────────┬────────────────────────────────────┘
                              SeasonLedger (SQLite) + unified Memory (LanceDB) ◄┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,anthropic]"                          # Python 3.11–3.13
python -m agri_resilience_crew --county Nakuru --crop "Irish Potatoes" --crop Maize
pytest -q
```

With no LLM key, the run uses the **deterministic engine** and still returns a full typed
playbook. Set `ANTHROPIC_API_KEY` (or `GEMINI_API_KEY`) to run the crews. Pass `--offline` to
force the deterministic engine. A sample output is in
[`examples/nakuru_playbook.json`](examples/nakuru_playbook.json).

Useful flags: `--sub-county Molo --sub-county Naivasha`, `--farm-size-ha 2.5`. Every setting is
an environment variable; see [`.env.example`](.env.example).

## Data: what is real and what is synthetic

KAMIS, KMD and KALRO **do not publish stable public JSON APIs**, so data access has two modes:

| Mode | Behaviour |
|---|---|
| `mock` (default) | Deterministic synthetic data ([`synthetic.py`](agri_resilience_crew/synthetic.py)). It has realistic shapes: seasonal price cycles with AR(1) noise, bimodal/unimodal rainfall regimes per agro-climatic zone, and acidic western soils. **The numbers are not observations.** Every result carries `synthetic=true` in `data_provenance`, and the guardrail rejects a playbook that hides this. |
| `live` | HTTP JSON adapters with a hard timeout, exponential-backoff retry and a last-known-good cache. You provide a thin adapter service (a scraper or data-sharing endpoint) that implements the contracts below. |

Live adapter contracts (`AGRI_CREW_*_BASE_URL`):

| Source | Request | Response |
|---|---|---|
| KAMIS | `GET {kamis}/prices?commodity=&market=&from=&to=` | `[{"date": "2026-10-01", "wholesale_kes_per_kg": 41.5}, …]` (≥ 30 rows) |
| KMD | `GET {kmd}/outlook?county=&sub_county=&start=&days=` | `{"prob_above": .2, "prob_near": .3, "prob_below": .5}`, optionally with `rainfall_anomaly_pct` and `expected_rainfall_mm` |
| KALRO | `GET {kalro}/soils?county=&sub_county=` | `{"soil_ph": 5.6, "soil_texture": "loam", "organic_carbon_pct": 2.0, "drainage": "good", "altitude_m": 2300}` |

The NEMA/WRA/KFS zoning layer is an **illustrative screen**, built on the real rules (Water Act riparian
reserves, the EMCA 2009 wetlands regulations, gazetted forests). Confirm site-specific constraints with
the county NEMA office. Zoning is screened for the first (primary) sub-county only.

All derived metrics are computed in Python ([`analytics.py`](agri_resilience_crew/analytics.py)), never by the LLM:
price velocity (30-day OLS slope), acceleration, volatility, haul-to-market margin net of transport, cess
and spoilage, FAO-style S1–N suitability, and gross margin. The agents interpret these numbers; they do not compute them.

## How the spec maps to the code

| Requirement | Where / how |
|---|---|
| Hierarchical crew with a manager | `strategy_crew`: `Process.hierarchical`, `manager_agent=agri_strategy_lead` (no tools; it delegates) ([`crew.py`](agri_resilience_crew/crew.py)) |
| Agents & tasks | YAML in [`config/`](agri_resilience_crew/config) loaded with `@CrewBase`, `@agent`, `@task`, `@crew` |
| `allow_delegation=True` on leads | Manager, plus `kalro_specialist`, which may ask the climate/market specialists for clarification |
| `async_execution=True` on independent research | T1 and T2 run concurrently on separate agents. T3 joins them. **See the design note below.** |
| Conditional task graph | After phase 1, a high/severe `ClimateImpactAssessment.risk_level` injects a contingency mandate into T4 ([`__main__.py`](agri_resilience_crew/__main__.py)) |
| `KAMISPriceFetcherTool`, `KMDWeatherScraperTool`, `KALROSoilSuitabilityTool` | `BaseTool` subclasses with Pydantic `args_schema` ([`tools.py`](agri_resilience_crew/tools.py)). Also `GrossMarginCalculatorTool` and an `@tool` reference lookup. An optional `SerperDevTool` web search is added when `SERPER_API_KEY` is set |
| Tool timeouts | Hard wall-clock deadline per data call (`AGRI_CREW_TOOL_TIMEOUT_S`, default 8 s). A data gap returns a `ToolFailure`, never an invented number ([`resilience.py`](agri_resilience_crew/resilience.py)) |
| `max_iter=15`, `max_rpm=30`, exponential backoff | Set on every agent and both crews. Tool I/O uses jittered exponential backoff. LLM calls use SDK `max_retries` plus CrewAI's built-in 429 backoff. `max_retry_limit=3` per agent |
| `output_pydantic=AgriStrategyOutput` | T4, with an **evidence guardrail** that rejects non-matching arithmetic, missing crops, yields/prices outside the tool evidence band, and dishonest provenance. CrewAI retries the task with the errors as feedback (`guardrail_max_retries=3`) ([`guardrails.py`](agri_resilience_crew/guardrails.py)) |
| Short-term / long-term / entity memory | CrewAI 1.x unified `Memory` (LanceDB) with scopes: entity records under `/entities/{crops,counties,markets}`, season records under `/seasons/<county>`; both crews share one `Memory` instance. Long-term history also lives in a SQLite `SeasonLedger`, which feeds prior playbooks back into the prompts ([`memory.py`](agri_resilience_crew/memory.py)) |
| MCP servers | `AGRI_CREW_MCP_SERVERS` (HTTP / SSE / stdio), mapped per agent through `Agent.mcps`, with tool allow-lists and `${ENV}` expansion for secrets ([`mcp.py`](agri_resilience_crew/mcp.py)) |
| Input validation | `CrewInputs` canonicalises and rejects unknown counties (all 47), sub-counties and crops (aliases such as `waru` and `mahindi` work) |

### Design notes and deviations from the original spec

* **Two crews instead of one.** In CrewAI, `Process.hierarchical` routes *every* task through the
  manager agent, and an agent has a single executor. Two `async_execution=True` tasks in one
  hierarchical crew would therefore run at the same time on the same manager and overwrite each
  other's executor state. Splitting the work keeps real parallelism (phase 1) *and* manager-led
  delegation (phase 2). The research outputs reach T4 as task context across the crew boundary;
  `tests/test_crew_wiring.py` checks this.
* **Memory.** The spec named ChromaDB short-term memory, SQLite long-term memory and a separate entity
  memory. CrewAI 1.x replaced all three with one LanceDB-backed `Memory`. This project maps the three
  tiers onto scopes and adds its own SQLite ledger. Memory needs an embedder, and **Anthropic has no
  embeddings API**, so with only `ANTHROPIC_API_KEY` set, vector memory is disabled with a warning and
  the SQLite ledger still runs. Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, `VOYAGE_API_KEY` or `AGRI_CREW_EMBEDDER` to enable it.
* **No `langchain_google_genai`.** CrewAI's native `LLM` class covers Anthropic (default:
  `claude-sonnet-5-5` for specialists, `claude-opus-5-5` for the manager) and Gemini.
* **Always returns a playbook.** If the crew fails, or its output never passes the guardrail, the
  deterministic engine's playbook is returned with `engine: "deterministic"`.

## Tests

`pytest -q` runs 48 offline tests. They cover input validation, analytics, backoff/timeout/cache
fallback (including live-mode HTTP via `httpx.MockTransport`), every tool, the engine across five
counties, guardrail rejections, MCP mapping, the season ledger, and an **end-to-end run of both real
CrewAI crews** against a scripted LLM. That run includes a guardrail-triggered retry.

> Decision support only. Validate against current KAMIS, KMD and county extension advisories before committing capital.
