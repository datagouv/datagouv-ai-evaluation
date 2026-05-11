"""
Prompt for evaluating trajectory adherence (sequence alignment).

Trajectory is evaluated independently for the minimal and optimal levels,
since the expected sequence may slightly differ between levels.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are an expert evaluator assessing how well an AI assistant's tool call sequence matches an expected trajectory.

You will be given:
- The user's original question
- The expected tool chain (a human-readable description of the ideal sequence)
- The list of required tools in order (at this level)
- The actual tool calls made by the agent, in order

Assess how closely the agent followed the expected sequence. Consider:
- Whether the key tools were called in the right logical order
- Whether parallel branches were covered (order within a parallel group is flexible)
- Whether the agent took unnecessary detours, repeated calls, or skipped critical steps
- The agent may call additional tools not in the ground truth; penalise only excessive redundancy

Score from 0.0 (completely off trajectory) to 1.0 (perfectly followed):
- 1.0: all required tools called in correct logical order
- 0.7–0.9: all required tools called but minor order deviations or light redundancy
- 0.4–0.6: some required tools called but notable gaps or misordering
- 0.1–0.3: few required tools called or very different sequence
- 0.0: no meaningful alignment with the expected trajectory

Respond with JSON in exactly this format, with no additional text:
{"score": 0.85, "explanation": "Two sentences max describing the alignment and main deviations."}
"""


def build_user_message(
    expected_chain: str,
    required_tools: list[dict[str, Any]],
    actual_tool_calls: list[dict[str, Any]],
    user_prompt: str,
) -> str:
    """
    Build the user message for one trajectory evaluation (minimal OR optimal level).

    Args:
        expected_chain: Human-readable chain string from the task YAML.
        required_tools: List of required tool dicts (name + args) for this level.
        actual_tool_calls: Ordered list of actual calls (name + arguments).
        user_prompt: The original user question.
    """
    lines = [f"User question:\n{user_prompt}\n"]

    lines.append(f"Expected chain:\n{expected_chain}\n")

    lines.append("Required tools (in logical order):")
    for i, tool in enumerate(required_tools):
        lines.append(f"  {i + 1}. {tool.get('name', '?')}")

    lines.append("\nActual tool calls (in order made):")
    if actual_tool_calls:
        for i, call in enumerate(actual_tool_calls):
            args = call.get("arguments") or {}
            if isinstance(args, dict):
                arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
                if len(args) > 3:
                    arg_str += ", ..."
            else:
                arg_str = str(args)[:80]
            lines.append(f"  {i + 1}. {call.get('name', '?')}({arg_str})")
    else:
        lines.append("  (no tool calls made)")

    return "\n".join(lines)
