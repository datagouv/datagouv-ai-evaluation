"""
Dispatcher: routes capability names to their toolset factory functions.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityUnavailableError(RuntimeError):
    """Raised when a required agent capability cannot be loaded."""


def build_toolsets(
    capabilities: list[str],
    run_config: dict[str, Any],
    docker_session=None,
) -> list:
    """
    Build pydantic-ai toolsets from capability names.
    Raises CapabilityUnavailableError if a required extra is not installed.

    Supported capabilities:
      mcp        — MCP server tool calls via MCPServerStreamableHTTP
      web_search — DuckDuckGo search + HTTP page fetch (URL blacklist, 50K char cap)
      code           — Docker-based Python execution (universal, all providers)
                       Requires docker_session (DockerSession) — created by the caller.
      datagouv-cli   — extends code: installs datagouv-client in Docker image and
                       exposes the `datagouv` CLI in the tool whitelist/description.
                       Must appear alongside `code` in capabilities.
      skills         — system prompt injection only; no toolset returned
    """
    toolsets = []
    for cap in capabilities:
        if cap == "mcp":
            from agent_eval.experiment.agent.mcp import mcp_toolset
            url = run_config.get("mcp_server_url")
            if url:
                toolsets.append(mcp_toolset(url, timeout=30))
        elif cap == "web_search":
            from agent_eval.experiment.agent.web_search import web_search_toolset
            toolsets.extend(web_search_toolset())
        elif cap == "code":
            from agent_eval.experiment.agent.code import code_toolset
            if docker_session is None:
                raise CapabilityUnavailableError(
                    "code capability requires a DockerSession — "
                    "pass docker_session= to build_toolsets()"
                )
            has_datagouv_cli = "datagouv-cli" in capabilities
            toolsets.extend(code_toolset(docker_session, has_datagouv_cli=has_datagouv_cli))
        elif cap == "datagouv-cli":
            pass  # handled by code capability via has_datagouv_cli flag
        elif cap == "skills":
            pass  # handled via system prompt injection in run_config.py
        else:
            logger.warning("Unknown capability %r — skipped", cap)
    return toolsets
