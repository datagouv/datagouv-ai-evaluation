"""
Builds flat run_config dicts from RunConfiguration objects.
Fetches MCP tools once per unique server URL (deduplicates network calls).
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_eval.benchmark.loader import RunConfiguration
from agent_eval.experiment.mcp_tools_getter import get_mcp_tools


async def _fetch_tools_by_url(urls: list[str]) -> dict[str, tuple[str, list[str], list[dict]]]:
    """Fetch MCP tools for all unique server URLs concurrently."""
    unique_urls = list(dict.fromkeys(u for u in urls if u))  # preserve order, dedup

    async def _fetch(url: str):
        return url, await get_mcp_tools(url)

    results = await asyncio.gather(*[_fetch(url) for url in unique_urls], return_exceptions=True)

    tools_by_url: dict[str, tuple[str, list[str], list[dict]]] = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        url, tool_data = res
        tools_by_url[url] = tool_data

    return tools_by_url


async def build_all_run_configs(
    run_configurations: list[RunConfiguration],
) -> list[dict[str, Any]]:
    """
    Convert RunConfiguration objects to flat dicts ready for make_task().
    MCP tools are fetched once per unique server URL.
    """
    urls = [rc.mcp_server_url for rc in run_configurations if rc.mcp_server_url]
    tools_by_url = await _fetch_tools_by_url(urls)

    run_configs: list[dict[str, Any]] = []
    for rc in run_configurations:
        url = rc.mcp_server_url
        if url and url in tools_by_url:
            description, names, schema = tools_by_url[url]
        else:
            description, names, schema = "", [], []

        run_configs.append({
            "evaluation_type": rc.evaluation_type,
            "capabilities": rc.capabilities,
            "mcp_version": rc.mcp_version,
            "mcp_server_url": url,
            "model": rc.model,
            "system_prompt_name": rc.system_prompt_name,
            "system_prompt": rc.system_prompt,
            "metrics": rc.metrics,
            "mcp_tools_description": description,
            "mcp_tool_names": names,
            "mcp_tools_schema": schema,
        })

    return run_configs
