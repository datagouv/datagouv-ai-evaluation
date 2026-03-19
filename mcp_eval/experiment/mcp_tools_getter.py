import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def get_remote_tools(server_url: str):
    async with streamable_http_client(server_url) as (read_stream, write_stream, *_):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return tools_result.tools


def mcp_tools_to_names(tools) -> list[str]:
    return [tool.name for tool in tools]


def mcp_tools_to_text(tools) -> str:
    lines = []

    for tool in tools:
        name = tool.name
        desc = getattr(tool, "description", "") or ""
        schema = getattr(tool, "inputSchema", {}) or {}

        lines.append(f"{name}: {desc}")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for param_name, meta in properties.items():
            param_type = meta.get("type", "any")
            param_desc = meta.get("description", "")
            req = "required" if param_name in required else "optional"
            lines.append(f"- {param_name} ({req}, type={param_type}): {param_desc}")

        lines.append("")

    return "\n".join(lines).strip()


async def get_mcp_tools(server_url: str) -> tuple[str, list[str]]:
    tools = await get_remote_tools(server_url)
    return mcp_tools_to_text(tools), mcp_tools_to_names(tools)
