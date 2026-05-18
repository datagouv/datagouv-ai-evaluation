"""
Prompt for evaluating tool parameter correctness.

Design notes:
- All ground-truth calls for the same tool name are grouped into one prompt together
  with all actual calls for that tool, so the judge can do one-to-one matching.
- The judge first maps each actual call to the best-matching ground-truth call.
  For strict values: exact match required, only one actual call may satisfy one GT call.
  For criteria-based values: pick the semantically closest call; mark others redundant.
- Schema compliance for unlisted args is checked deterministically after the LLM judge
  (see tool_params.py); if a call fails schema compliance it cannot be validated even if
  its listed parameters matched.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are an expert evaluator assessing tool call correctness for an AI assistant.

You will be given:
- The user's original question
- A list of GROUND TRUTH tool calls (expected): each has a list of parameter constraints
  (either a strict expected value that must match exactly, or a semantic criteria description)
- A list of ACTUAL tool calls made by the agent for the same tool

Your task has two steps:

STEP 1 — Map each actual call to at most one ground truth call:
- For strict-value parameters: the actual value must exactly match the expected value.
- For criteria-based parameters: the actual value must be semantically appropriate given the user question.
- Each ground truth call may be matched by at most ONE actual call (first best match wins).
- If multiple actual calls could match the same ground truth, only the closest one counts; the others are marked redundant.
- An actual call that doesn't match any ground truth is marked unmatched.

STEP 2 — For each matched pair, assess whether ALL listed parameter constraints are satisfied.
- A call is "correct" only if EVERY listed constraint (strict or criteria) is satisfied.

Respond with JSON in exactly this format, with no additional text:
{
  "mappings": [
    {
      "ground_truth_index": 0,
      "actual_call_index": 1,
      "correct": true,
      "explanation": "one sentence"
    }
  ],
  "unmatched_actual_indices": [2],
  "redundant_actual_indices": []
}

Notes:
- ground_truth_index and actual_call_index are 0-based.
- An actual call index can appear in at most one of: mappings, unmatched_actual_indices, redundant_actual_indices.
- If a ground truth call has no matching actual call, omit it from mappings (it counts as a miss).
"""


def build_user_message(
    tool_name: str,
    ground_truth_calls: list[dict[str, Any]],
    actual_calls: list[dict[str, Any]],
    user_prompt: str,
) -> str:
    """
    Build the user message for a single tool's parameter correctness evaluation.

    Args:
        tool_name: The exact tool name being evaluated.
        ground_truth_calls: List of required tool dicts, each with {"args": [{name, strict_value, criteria}]}.
        actual_calls: List of actual call dicts, each with {"name", "arguments": dict}.
        user_prompt: The original user question.
    """
    lines = [f"User question:\n{user_prompt}\n"]

    lines.append(f"Tool: {tool_name}\n")

    lines.append("GROUND TRUTH calls (expected):")
    for i, gt in enumerate(ground_truth_calls):
        args = gt.get("args") or []
        arg_strs = []
        for arg in args:
            if arg.get("strict_value") is not None:
                arg_strs.append(f"  {arg['name']}: EXACT VALUE = {arg['strict_value']!r}")
            elif arg.get("criteria"):
                arg_strs.append(f"  {arg['name']}: CRITERIA = {arg['criteria']}")
            else:
                arg_strs.append(f"  {arg['name']}: (any value accepted)")
        lines.append(f"[{i}] " + (("\n" + "\n".join(arg_strs)) if arg_strs else "(no constraints)"))

    lines.append("\nACTUAL calls made by the agent:")
    for i, call in enumerate(actual_calls):
        args = call.get("arguments") or {}
        arg_strs = [f"  {k}: {v!r}" for k, v in args.items()] if isinstance(args, dict) else [f"  {args}"]
        lines.append(f"[{i}] " + (("\n" + "\n".join(arg_strs)) if arg_strs else "(no arguments)"))

    return "\n".join(lines)
