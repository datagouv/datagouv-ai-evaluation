import ast
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from opik.evaluation.metrics import base_metric, score_result

load_dotenv(override=True)

_SYSTEM_PROMPT = """\
You are an expert evaluator for AI tool usage. Given a user query, a list of \
available tools, and the tools that were actually called, decide whether the \
tool invocation is correct.

If no tool was needed and :
a. no tool was called, then the score is 1
b. a tool or more were called, even if its invocation is correct, the score is 0 

Respond with JSON in exactly this format:
{"score": <1 or 0>, "label": "<correct|incorrect>", "explanation": "<one sentence>"}
"""


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
            rendered_args = _python_repr(args)
        elif args is None:
            rendered_args = ""
        else:
            rendered_args = _python_repr(args)

        lines.append(f"{name}({rendered_args})")

    return "\n".join(lines)


class ToolInvocationCorrectnessJudge(base_metric.BaseMetric):
    def __init__(self):
        super().__init__(name="tool_invocation_correctness_judge")
        self._client = OpenAI()
        self._model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    def score(
        self,
        input: dict,
        output: dict,
        **kwargs,
    ) -> score_result.ScoreResult:
        user_message = (
            f"User query: {input['prompt']}\n\n"
            f"Available tools:\n{output['available_tools']}\n\n"
            f"Tool calls made:\n{format_tool_calls_human_readable(output['actual_tool_calls'])}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)

        return score_result.ScoreResult(
            name=self.name,
            value=float(raw.get("score", 0)),
            reason=f"{raw.get('label', '')}: {raw.get('explanation', '')}",
        )
