"""A simple agent to run the test prompt."""

from dataclasses import dataclass
from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP


@dataclass
class AgentResult:
    task_id: str
    server_version: str
    model: str
    answer: str


async def run_agent(task: Dict[str, Any]) -> AgentResult:
    mcp_server = MCPServerStreamableHTTP(url=task["server_url"])

    agent = Agent(
        model=task["model"],
        toolsets=[mcp_server],
        system_prompt=task["system_prompt"],
    )

    # IMPORTANT: ensures MCP + HTTP clients are opened/closed within the test's event loop
    async with agent:
        run_result = await agent.run(task["prompt"])

    return AgentResult(
        task_id=task["task_id"],
        server_version=task["server_version"],
        model=task["model"],
        answer=run_result.output,
    )
