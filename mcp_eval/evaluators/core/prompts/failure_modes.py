"""
Prompt for failure mode detection.

The failure modes definition file lives at mcp_eval/evaluators/failure_modes.yml.
That file will be populated by the user; this module loads it and builds the judge prompt.
Each failure mode gets a score of 1 (present = failure) or 0 (absent = no failure).
Higher total across tasks = more failures = worse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_FAILURE_MODES_PATH = Path(__file__).parent.parent / "config" / "failure_modes.yml"


def load_failure_modes(path: Path | None = None) -> list[dict[str, str]]:
    """
    Load failure mode definitions from YAML.
    Returns list of {"name": ..., "description": ...} dicts.
    Returns empty list if file does not exist yet.
    """
    resolved = path or _DEFAULT_FAILURE_MODES_PATH
    if not resolved.exists():
        return []
    with open(resolved, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("failure_modes") or []


SYSTEM_PROMPT = """\
You are an expert evaluator detecting failure modes in an AI assistant's response.

You will be given:
- The user's original question
- The evaluation criteria that the agent FAILED to meet, with the reason why
- The tool calls the agent made (name, arguments, result for each)
- The assistant's final response
- A list of failure modes, each with a name and description

Your task: identify which failure modes are clearly present.
The failed criteria are the strongest signal — use them as primary evidence.
Also use tool call sequence, arguments, results, and the final answer.
Be willing to flag a mode when evidence supports it; do not default to absent out of caution.

Respond with JSON in exactly this format, with no extra text:
{"present": ["MODE_A", "MODE_B"], "explanations": {"MODE_A": "one sentence citing evidence", "MODE_B": "..."}}

- "present" lists only the failure mode names that are present (use exact names from the input list).
- If no failure mode is present: {"present": [], "explanations": {}}
- Only modes in "present" need an entry in "explanations".
"""


def _fmt_required_tools(tools: list[dict[str, Any]]) -> str:
    lines = []
    for t in tools:
        name = t.get("name", "?")
        args = t.get("args") or []
        arg_parts = []
        for a in args:
            if a.get("strict_value") is not None:
                arg_parts.append(f"{a['name']}={a['strict_value']!r}")
            elif a.get("criteria"):
                arg_parts.append(f"{a['name']}: {a['criteria'][:80]}")
        lines.append(f"  - {name}({', '.join(arg_parts)})")
    return "\n".join(lines) if lines else "  (none)"


def build_user_message(
    user_prompt: str,
    answer: str,
    failed_criteria: list[Any],
    actual_tool_calls: list[dict[str, Any]],
    required_tools_minimal: list[dict[str, Any]],
    required_tools_optimal: list[dict[str, Any]],
    failure_modes: list[dict[str, str]],
) -> str:
    lines = [f"User question:\n{user_prompt}\n"]

    if failed_criteria:
        lines.append("Criteria the agent FAILED to meet:")
        for r in failed_criteria:
            short = r.criterion[:120].strip() + ("…" if len(r.criterion) > 120 else "")
            lines.append(f"  [✗] {short}")
            lines.append(f"      Reason: {r.explanation}")
    else:
        lines.append("Criteria evaluation: all criteria were met.\n")

    if required_tools_minimal or required_tools_optimal:
        lines.append("\nRequired tool calls (minimal level):")
        lines.append(_fmt_required_tools(required_tools_minimal))
        if required_tools_optimal:
            lines.append("Required tool calls (optimal level, superset of minimal):")
            lines.append(_fmt_required_tools(required_tools_optimal))

    if actual_tool_calls:
        lines.append("\nTool calls made by the agent:")
        for i, call in enumerate(actual_tool_calls, 1):
            name = call.get("name", "unknown")
            args = call.get("args") or call.get("arguments") or {}
            result = call.get("result") or call.get("output") or ""
            lines.append(f"  {i}. {name}({args})")
            if result:
                result_str = str(result)[:300] + ("…" if len(str(result)) > 300 else "")
                lines.append(f"     → {result_str}")
    else:
        lines.append("\nTool calls made by the agent: none")

    lines.append(f"\nAssistant final response:\n{answer}\n")

    lines.append("Failure modes to detect:")
    for fm in failure_modes:
        lines.append(f"  {fm['name']}: {fm.get('description', '')}")

    return "\n".join(lines)
