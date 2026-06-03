from pydantic_ai.mcp import MCPServerStreamableHTTP


def mcp_toolset(url: str, timeout: int = 30) -> MCPServerStreamableHTTP:
    return MCPServerStreamableHTTP(url=url, timeout=timeout)
