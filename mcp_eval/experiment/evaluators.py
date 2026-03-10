import ast
from phoenix.evals import create_evaluator


@create_evaluator(name="tool_selection_accuracy", kind="code", direction="maximize")
def tool_selection_accuracy(expected, output):
    expected_tools = [
        tool["name"] for tool in ast.literal_eval(expected["expected_tool_calls"])
    ]

    correct_tools = [
        tool["name"]
        for tool in output["actual_tool_calls"]
        if tool["name"] in expected_tools
    ]
    return len(correct_tools) / len(expected_tools)
