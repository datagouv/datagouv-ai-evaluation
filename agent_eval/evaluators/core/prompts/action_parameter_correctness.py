"""
Prompt for evaluating semantic-action parameter correctness.

Design notes:
- All ground-truth actions with the same semantic name are grouped into one prompt
  together with all actual action instances of that name, so the judge can do
  one-to-one matching.
- The judge first maps each actual action to the best-matching ground-truth action.
  For strict values: exact match required, only one actual action may satisfy one GT action.
  For criteria-based values: pick the semantically closest action; mark others redundant.
- An action instance may originate from an MCP tool call, a CLI command, or a step inside
  a code-execution block; its `args` carry the parameters extracted for that action.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are an expert evaluator assessing semantic-action correctness for an AI assistant.

A "semantic action" is one logical operation against data.gouv.fr (e.g. searching datasets,
fetching dataset metadata, fetching data rows). The agent may realize it through an MCP tool
call, an API call, a CLI command, or a step inside a code-execution block — only the action
and its arguments matter here, not how it was invoked.

You will be given:
- The user's original question
- A list of GROUND TRUTH actions (expected): each has a list of parameter constraints
  (either a strict expected value that must match exactly, or a semantic criteria description)
- A list of ACTUAL actions performed by the agent for the same semantic action

Your task has two steps:

STEP 1 — Map each actual action to at most one ground truth action:
- For strict-value parameters: the actual value must exactly match the expected value.
- For criteria-based parameters: the actual value must be semantically appropriate given the user question.
- Each ground truth action may be matched by at most ONE actual action (first best match wins).
- If multiple actual actions could match the same ground truth, only the closest one counts; the others are marked redundant.
- An actual action that doesn't match any ground truth is marked unmatched.

STEP 2 — For each matched pair, assess whether ALL listed parameter constraints are satisfied.
- An action is "correct" only if EVERY listed constraint (strict or criteria) is satisfied.

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
- An actual action index can appear in at most one of: mappings, unmatched_actual_indices, redundant_actual_indices.
- If a ground truth action has no matching actual action, omit it from mappings (it counts as a miss).
"""


def build_user_message(
    action_name: str,
    ground_truth_calls: list[dict[str, Any]],
    actual_calls: list[dict[str, Any]],
    user_prompt: str,
) -> str:
    """
    Build the user message for a single semantic action's parameter correctness evaluation.

    Args:
        action_name: The semantic action name being evaluated (e.g. "search.datasets").
        ground_truth_calls: List of required action dicts, each with {"args": [{name, strict_value, criteria}]}.
        actual_calls: List of actual action dicts, each with {"arguments": dict}.
        user_prompt: The original user question.
    """
    lines = [f"User question:\n{user_prompt}\n"]

    lines.append(f"Semantic action: {action_name}\n")

    lines.append("GROUND TRUTH actions (expected):")
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

    lines.append("\nACTUAL actions performed by the agent:")
    for i, call in enumerate(actual_calls):
        args = call.get("arguments") or {}
        arg_strs = [f"  {k}: {v!r}" for k, v in args.items()] if isinstance(args, dict) else [f"  {args}"]
        lines.append(f"[{i}] " + (("\n" + "\n".join(arg_strs)) if arg_strs else "(no arguments)"))

    return "\n".join(lines)
