import asyncio
import time
from dataclasses import dataclass
from typing import Any
import logging

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, ToolReturnPart, ToolCallPart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _normalize_tool_args(part: ToolCallPart) -> Any:
    args = part.args
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        return args
    # Fallback for richer arg wrappers
    if hasattr(part, "args_as_json_str"):
        return part.args_as_json_str()
    return args


def extract_actual_tool_calls(run_result) -> list[dict[str, Any]]:
    messages = run_result.new_messages()

    actual_tool_calls: list[dict[str, Any]] = []
    tool_returns_by_id: dict[str, Any] = {}

    for message in messages:
        parts = getattr(message, "parts", [])

        # Collect tool calls from model responses
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

        # Collect tool returns from any remaining message parts
        for part in parts:
            if isinstance(part, ToolReturnPart):
                tool_returns_by_id[part.tool_call_id] = part.content

    # Attach results when available
    for call in actual_tool_calls:
        tool_call_id = call["tool_call_id"]
        if tool_call_id in tool_returns_by_id:
            call["result"] = tool_returns_by_id[tool_call_id]

    return actual_tool_calls


@dataclass
class AgentResult:
    server_version: str
    model: str
    answer: str
    actual_tool_calls: list[dict[str, Any]]
    available_tools: str


async def run_agent(task: dict[str, Any]) -> AgentResult:

    logger = logging.getLogger(__name__)
    logger.info(f"MCP Server URL : {task['mcp_server_url']}")
    mcp_server = MCPServerStreamableHTTP(url=task["mcp_server_url"], timeout=30)

    agent = Agent(
        name=f"{task['model']}-{task['mcp_server_url']}",
        model=task["model"],
        toolsets=[mcp_server],
        system_prompt=task["system_prompt"],
        # instrument=True,
    )

    async with agent:
        run_result = await agent.run(task["prompt"])

    actual_tool_calls = extract_actual_tool_calls(run_result)

    return AgentResult(
        server_version=task["mcp_version"],
        model=task["model"],
        answer=run_result.output,
        actual_tool_calls=actual_tool_calls,
        available_tools=task["mcp_tools_description"],
    )


_MAX_RETRIES = 5
_RETRY_BACKOFF = [10, 30, 60, 90, 120]  # secondes entre tentatives


def make_task(run_config: dict[str, Any]):
    logger = logging.getLogger(__name__)

    def task(dataset_item: dict) -> dict:
        # Opik DatasetItem shape: {"input": {...}, "expected_output": {...}, "metadata": {...}}
        # evaluate() runs tasks in a ThreadPoolExecutor — asyncio.run() creates a fresh loop per thread
        task_data = dataset_item["input"] | run_config
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = asyncio.run(run_agent(task_data))
                return {
                    "output": {
                        "answer": result.answer,
                        "actual_tool_calls": result.actual_tool_calls,
                        "available_tools": result.available_tools,
                    }
                }
            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF[attempt]
                logger.warning(f"Task attempt {attempt + 1}/{_MAX_RETRIES} failed ({exc}), retrying in {wait}s")
                time.sleep(wait)
        raise last_exc

    return task
