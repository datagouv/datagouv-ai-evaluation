import ast
import json
from typing import Any

from phoenix.evals import LLM, create_evaluator
from phoenix.evals.metrics import ToolInvocationEvaluator
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# 1) Judge model for Phoenix's built-in evaluator
# ----------------------------
judge_llm = LLM(provider="openai", model="gpt-5-mini")
tool_invocation_judge = ToolInvocationEvaluator(llm=judge_llm)


def _parse_maybe_literal(value: Any):
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return value
    return value


def _python_repr(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return json.dumps(v, ensure_ascii=False)


def format_tool_calls_human_readable(tool_calls: Any) -> str:
    tool_calls = _parse_maybe_literal(tool_calls) or []
    lines = []

    for call in tool_calls:
        name = call.get("name", "unknown_tool")
        args = _parse_maybe_literal(call.get("arguments", {}))

        if isinstance(args, dict):
            rendered_args = ", ".join(f"{k}={_python_repr(v)}" for k, v in args.items())
        elif isinstance(args, str):
            # Fallback when args is just a raw string
            rendered_args = _python_repr(args)
        elif args is None:
            rendered_args = ""
        else:
            rendered_args = _python_repr(args)

        lines.append(f"{name}({rendered_args})")

    return "\n".join(lines)


@create_evaluator(
    name="tool_invocation_correctness_judge", kind="llm", direction="maximize"
)
def tool_invocation_correctness_judge(input, output):
    """
    Assumes:
      - input is the original prompt string
      - output contains output["actual_tool_calls"]
    """
    eval_input = {
        "input": input["prompt"],
        "available_tools": output["available_tools"],
        "tool_selection": format_tool_calls_human_readable(output["actual_tool_calls"]),
    }

    # Phoenix returns a list of Score objects; for a single example take the first.
    score = tool_invocation_judge.evaluate(eval_input)[0]

    # Returning a dict is convenient for Phoenix experiments UI.
    return {
        "score": score.score,  # 1.0 or 0.0
        "label": score.label,  # "correct" / "incorrect"
        "explanation": score.explanation,
    }
