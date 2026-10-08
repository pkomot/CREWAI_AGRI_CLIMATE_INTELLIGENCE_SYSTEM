"""MCP (Model Context Protocol) wiring for external vector knowledge bases and spatial data.

Servers are declared with ``AGRI_CREW_MCP_SERVERS``, a JSON list such as::

    [
      {"type": "http", "url": "https://kb.example.org/mcp",
       "headers": {"Authorization": "Bearer ${AGRI_KB_TOKEN}"},
       "agents": ["kalro_specialist", "agri_strategy_lead"], "tools": ["search_extension_notes"]},
      {"type": "stdio", "command": "uvx", "args": ["geo-mcp-server"],
       "agents": ["kmd_analyst"]}
    ]

``${VAR}`` placeholders in headers/env are expanded from the environment at load
time, so secrets never live in the JSON. ``agents`` restricts a server to the
named agents (default: all specialists); ``tools`` is an allow-list.
"""

from __future__ import annotations

import os
from typing import Any

from crewai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio, create_static_tool_filter

SPECIALISTS = ("kmd_analyst", "kamis_analyst", "kalro_specialist")
ALL_AGENTS = (*SPECIALISTS, "agri_strategy_lead")

MCPServer = MCPServerHTTP | MCPServerSSE | MCPServerStdio


def _expand(mapping: dict[str, str] | None) -> dict[str, str] | None:
    if mapping is None:
        return None
    return {k: os.path.expandvars(v) for k, v in mapping.items()}


def build_server(spec: dict[str, Any]) -> MCPServer:
    kind = spec.get("type", "http")
    tool_filter = create_static_tool_filter(allowed_tool_names=spec["tools"]) if spec.get("tools") else None
    common: dict[str, Any] = {"tool_filter": tool_filter, "cache_tools_list": True}
    if kind == "http":
        return MCPServerHTTP(url=spec["url"], headers=_expand(spec.get("headers")), **common)
    if kind == "sse":
        return MCPServerSSE(url=spec["url"], headers=_expand(spec.get("headers")), **common)
    if kind == "stdio":
        return MCPServerStdio(
            command=spec["command"], args=list(spec.get("args", [])), env=_expand(spec.get("env")), **common
        )
    raise ValueError(f"unsupported MCP server type '{kind}' (expected http, sse or stdio)")


def mcp_servers_by_agent(specs: list[dict[str, Any]]) -> dict[str, list[MCPServer]]:
    """Map agent key -> MCP servers it may use."""
    out: dict[str, list[MCPServer]] = {name: [] for name in ALL_AGENTS}
    for spec in specs:
        targets = spec.get("agents") or list(SPECIALISTS)
        unknown = set(targets) - set(ALL_AGENTS)
        if unknown:
            raise ValueError(f"MCP spec targets unknown agents: {sorted(unknown)}")
        server = build_server(spec)
        for agent in targets:
            out[agent].append(server)
    return out
