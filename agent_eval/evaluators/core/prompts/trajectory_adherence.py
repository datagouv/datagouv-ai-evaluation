"""
Prompt for evaluating trajectory adherence (sequence alignment).

Trajectory is evaluated independently for the minimal and optimal levels,
since the expected sequence may slightly differ between levels.

Operates on the mapped semantic action sequence, not literal tool calls: one
literal tool call may expand to several actions (e.g. searches inside a script).
"""

from typing import Any

SYSTEM_PROMPT = """\
You are an expert evaluator assessing how well an AI assistant's sequence of semantic actions matches an expected trajectory.

A "semantic action" is one logical operation against data.gouv.fr (e.g. searching datasets, fetching
dataset metadata, fetching data rows), regardless of whether it was realized via an MCP tool call,
an API call, a CLI command, or a step inside a code-execution block.

You will be given:
- The user's original question
- The expected action chain (a human-readable description of the ideal sequence)
- The list of required actions in order (at this level)
- The actual semantic actions the agent performed, in order

Assess how closely the agent followed the expected sequence. Consider:
- Whether the key actions were performed in the right logical order
- Whether parallel branches were covered (order within a parallel group is flexible)
- Whether the agent took unnecessary detours, repeated actions, or skipped critical steps
- The agent may perform additional actions not in the ground truth; penalise only excessive redundancy

Score from 0.0 (completely off trajectory) to 1.0 (perfectly followed):
- 1.0: all required actions performed in correct logical order
- 0.7–0.9: all required actions performed but minor order deviations or light redundancy
- 0.4–0.6: some required actions performed but notable gaps or misordering
- 0.1–0.3: few required actions performed or very different sequence
- 0.0: no meaningful alignment with the expected trajectory

Respond with JSON in exactly this format, with no additional text:
{"score": 0.85, "explanation": "Two sentences max describing the alignment and main deviations."}
"""


def build_user_message(
    expected_chain: str,
    required_actions: list[dict[str, Any]],
    actual_actions: list[dict[str, Any]],
    user_prompt: str,
) -> str:
    """
    Build the user message for one trajectory evaluation (minimal OR optimal level).

    Args:
        expected_chain: Human-readable chain string from the task YAML.
        required_actions: List of required action dicts (name) for this level.
        actual_actions: Ordered list of actual semantic actions (name + arguments).
        user_prompt: The original user question.
    """
    lines = [f"User question:\n{user_prompt}\n"]

    lines.append(f"Expected chain:\n{expected_chain}\n")

    lines.append("Required actions (in logical order):")
    for i, action in enumerate(required_actions):
        lines.append(f"  {i + 1}. {action.get('name', '?')}")

    lines.append("\nActual actions performed (in order):")
    if actual_actions:
        for i, action in enumerate(actual_actions):
            args = action.get("arguments") or {}
            if isinstance(args, dict):
                arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
                if len(args) > 3:
                    arg_str += ", ..."
            else:
                arg_str = str(args)[:80]
            lines.append(f"  {i + 1}. {action.get('name', '?')}({arg_str})")
    else:
        lines.append("  (no actions performed)")

    return "\n".join(lines)
