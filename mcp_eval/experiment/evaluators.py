import ast
from phoenix.evals import create_evaluator


@create_evaluator(name="tool_selection_accuracy", kind="code", direction="maximize")
def tool_selection_accuracy(expected, output):
    expected_tool_names = [
        tool["name"] for tool in ast.literal_eval(expected["expected_tool_calls"])
    ]

    correct_tool_names = [
        tool["name"]
        for tool in output["actual_tool_calls"]
        if tool["name"] in expected_tool_names
    ]
    return len(correct_tool_names) / len(expected_tool_names)


"""
@create_evaluator(name="tool_call_accuracy", kind="code", direction="maximize")
def tool_call_accuracy(expected, output):
    expected_tools = ast.literal_eval(expected["expected_tool_calls"])
    expected_tool_names = [
        tool["name"] for tool in expected_tools
    ]
    for actual_tool in output["actual_tool_calls"]:
        actual_tool_name = actual_tool["name"]
        if actual_tool_name in expected_tool_names:
            for key, value in actual_tool.items():
                if key in expected_tools[actual_tool_name]
                
                
            
    
    return len(correct_tools) / len(expected_tools)"""
