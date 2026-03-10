import ast
import json
from phoenix.evals import create_evaluator


def _parse_expected_tool_calls(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        try:
            return json.loads(value)
        except Exception:
            return ast.literal_eval(value)
    return []


def _dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


@create_evaluator(name="tool_selection_match", kind="code", direction="maximize")
def tool_selection_match(expected, output):
    expected_calls = _parse_expected_tool_calls(expected["expected_tool_calls"])

    expected_tool_names = _dedupe_keep_order(
        [tool["name"] for tool in expected_calls if "name" in tool]
    )
    actual_tool_names = _dedupe_keep_order(
        [tool["name"] for tool in output.get("actual_tool_calls", []) if "name" in tool]
    )

    if not expected_tool_names:
        return 1.0

    correct_tool_names = [
        name for name in actual_tool_names if name in expected_tool_names
    ]

    return len(correct_tool_names) / len(expected_tool_names)
