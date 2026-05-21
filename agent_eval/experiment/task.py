import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv, dotenv_values
import opik
from opik import opik_context
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import Tool as PydanticTool
from agent_eval.utils import get_model_config_object
from agent_eval.experiment.agent import build_toolsets
from agent_eval.experiment.agent.builder import CapabilityUnavailableError
from agent_eval.experiment.agent.code import check_and_create_session

load_dotenv(override=True)
config = dotenv_values(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# CapabilityUnavailableError is re-exported from agent.builder for callers that import it here
# build_toolsets is imported from agent_eval.experiment.agent


# ── Tool call extraction ──────────────────────────────────────────────────────


def _normalize_tool_args(part: ToolCallPart) -> Any:
    args = part.args
    if isinstance(args, (dict, str)):
        return args
    if hasattr(part, "args_as_json_str"):
        return part.args_as_json_str()
    return args


def extract_actual_tool_calls(run_result) -> list[dict[str, Any]]:
    messages = run_result.new_messages()
    actual_tool_calls: list[dict[str, Any]] = []
    tool_returns_by_id: dict[str, Any] = {}

    for message in messages:
        parts = getattr(message, "parts", [])
        if isinstance(message, ModelResponse):
            for part in parts:
                if isinstance(part, ToolCallPart):
                    actual_tool_calls.append(
                        {
                            "tool_call_id": part.tool_call_id,
                            "name": part.tool_name,
                            "arguments": _normalize_tool_args(part),
                            "result": None,
                        }
                    )
        for part in parts:
            if isinstance(part, ToolReturnPart):
                tool_returns_by_id[part.tool_call_id] = part.content

    for call in actual_tool_calls:
        if call["tool_call_id"] in tool_returns_by_id:
            call["result"] = tool_returns_by_id[call["tool_call_id"]]

    return actual_tool_calls


# ── Agent result ──────────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    server_version: str | None
    model: str
    answer: str
    actual_tool_calls: list[dict[str, Any]]
    available_tools: str
    available_tool_names: list[str]
    available_tools_schema: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: int = 0


# ── Agent runner ──────────────────────────────────────────────────────────────


async def run_agent(task: dict[str, Any]) -> AgentResult:
    logger = logging.getLogger(__name__)

    capabilities: list[str] = task.get("capabilities") or []

    # code capability requires a persistent Docker container for the task's lifetime
    if "code" in capabilities:
        has_datagouv_cli = "datagouv-cli" in capabilities
        session = check_and_create_session(has_datagouv_cli=has_datagouv_cli)
        with session:
            return await _run_agent_inner(task, capabilities, docker_session=session)
    else:
        return await _run_agent_inner(task, capabilities, docker_session=None)


async def _run_agent_inner(
    task: dict[str, Any],
    capabilities: list[str],
    docker_session,
) -> AgentResult:
    logger = logging.getLogger(__name__)

    # May raise CapabilityUnavailableError — caller handles it
    all_tools = build_toolsets(capabilities, task, docker_session=docker_session)
    # pydantic-ai distinguishes individual Tool objects (tools=) from AbstractToolset
    # objects like MCPServerStreamableHTTP (toolsets=)
    individual_tools = [t for t in all_tools if isinstance(t, PydanticTool)]
    toolsets = [t for t in all_tools if not isinstance(t, PydanticTool)]

    model = get_model_config_object(task["model"])
    agent_name = f"{task['model']['name']}-{task.get('mcp_server_url') or 'no-tools'}"
    system_prompt = task.get("system_prompt", "")
    skills_content = task.get("skills_content", "")
    if skills_content:
        system_prompt = f"{system_prompt}\n\n{skills_content}".strip()
    agent = Agent(
        name=agent_name,
        model=model,
        tools=individual_tools or None,
        toolsets=toolsets or None,
        system_prompt=system_prompt,
        instrument=True,
    )

    logger.info("Running agent: %s (capabilities=%s)", agent_name, capabilities)

    t0 = time.perf_counter()
    async with agent:
        run_result = await agent.run(task["prompt"])
    latency_ms = (time.perf_counter() - t0) * 1000
    net_latency_ms = latency_ms - model.rate_limit_wait_ms

    usage = run_result.usage() if callable(getattr(run_result, "usage", None)) else None
    token_usage = getattr(usage, "total_tokens", 0) or 0

    return AgentResult(
        server_version=task.get("mcp_version"),
        model=task["model"]["name"],
        answer=run_result.output,
        actual_tool_calls=extract_actual_tool_calls(run_result),
        available_tools=task.get("mcp_tools_description", ""),
        available_tool_names=task.get("mcp_tool_names", []),
        available_tools_schema=task.get("mcp_tools_schema", []),
        latency_ms=net_latency_ms,
        token_usage=token_usage,
    )


# ── Opik-native tracing ───────────────────────────────────────────────────────


@opik.track(type="tool", capture_input=False, capture_output=False)
def _log_tool_call(call: dict) -> None:
    """Create one Opik child span per tool call, nested under the agent_run span."""
    opik_context.update_current_span(
        name=call.get("name", "tool_call"),
        input={"arguments": call.get("arguments") or {}},
        output={
            "result": str(call["result"]) if call.get("result") is not None else None
        },
    )


@opik.track(type="llm", capture_input=False, capture_output=False)
async def _run_agent_and_log(task_data: dict[str, Any]) -> AgentResult:
    """
    Run the agent inside an Opik span so the trace nests under the evaluation task.
    Tool calls are logged as child spans. Model/token info is attached to this span.
    """
    result = await run_agent(task_data)

    opik_context.update_current_span(
        name=f"agent:{task_data.get('model', 'unknown')}",
        input={"prompt": task_data.get("prompt", "")},
        output={"answer": result.answer},
        model=task_data.get("model", {})["name"],
        usage={
            "total_tokens": result.token_usage,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
    )
    for call in result.actual_tool_calls:
        _log_tool_call(call)

    return result


# ── Task factory for Opik evaluate() ─────────────────────────────────────────

_MAX_RETRIES = 5
_RETRY_BACKOFF = [
    61,
    70,
    80,
    90,
    120,
]  # Albert API RPM quotas is 10, to avoid 429 errors


def make_task(run_config: dict[str, Any]):
    logger = logging.getLogger(__name__)

    def task(dataset_item: dict) -> dict:
        task_data = dataset_item["input"] | run_config
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                result = asyncio.run(_run_agent_and_log(task_data))
                return {
                    "output": {
                        "answer": result.answer,
                        "actual_tool_calls": result.actual_tool_calls,
                        "available_tools": result.available_tools,
                        "available_tool_names": result.available_tool_names,
                        "available_tools_schema": result.available_tools_schema,
                        "latency_ms": result.latency_ms,
                        "token_usage": result.token_usage,
                        "mcp_version": task_data.get("mcp_version"),
                        "capabilities": task_data.get("capabilities", []),
                    }
                }
            except CapabilityUnavailableError as exc:
                logger.error("Skipping task — capability unavailable: %s", exc)
                return {"output": None, "skipped": True, "skip_reason": str(exc)}
            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning(
                    "Task attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("Task failed after %d attempts: %s", _MAX_RETRIES, last_exc)
        return {
            "output": {
                "answer": "",
                "actual_tool_calls": [],
                "available_tools": task_data.get("mcp_tools_description", ""),
                "available_tool_names": task_data.get("mcp_tool_names", []),
                "available_tools_schema": task_data.get("mcp_tools_schema", []),
                "latency_ms": 0.0,
                "token_usage": 0,
                "mcp_version": task_data.get("mcp_version"),
                "capabilities": task_data.get("capabilities", []),
            }
        }

    return task
