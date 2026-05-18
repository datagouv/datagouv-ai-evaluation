"""
Deterministic schema compliance check.
Validates each tool call's arguments against the MCP tool's declared JSON inputSchema.
No LLM calls, no Opik imports.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False
    logger.warning("jsonschema not installed; schema compliance checks will always return True")


def _find_schema(tool_name: str, available_tools_schema: list[dict]) -> dict | None:
    for tool in available_tools_schema:
        if tool.get("name") == tool_name:
            return tool.get("inputSchema") or {}
    return None


def check_schema_compliance(
    tool_call: dict[str, Any],
    available_tools_schema: list[dict],
) -> bool:
    """
    Return True if the tool call's arguments are valid against the tool's inputSchema.
    Returns True if jsonschema is not installed or the tool schema is not found
    (to avoid false negatives when schema data is unavailable).
    """
    if not _HAS_JSONSCHEMA:
        return True

    tool_name = tool_call.get("name", "")
    schema = _find_schema(tool_name, available_tools_schema)
    if schema is None:
        return True  # unknown tool — can't validate

    arguments = tool_call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return False

    try:
        jsonschema.validate(instance=arguments, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError as exc:
        logger.warning("Invalid JSON schema for tool %r: %s", tool_name, exc)
        return True  # malformed schema — don't penalise the agent


def compute_schema_compliance(
    actual_tool_calls: list[dict[str, Any]],
    available_tools_schema: list[dict],
) -> tuple[int, int]:
    """
    Returns (schema_compliant_count, total_count).
    schema_compliance_rate = compliant / total (caller derives rate).
    """
    total = len(actual_tool_calls)
    compliant = sum(
        1 for call in actual_tool_calls
        if check_schema_compliance(call, available_tools_schema)
    )
    return compliant, total
