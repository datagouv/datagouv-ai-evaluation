"""
web_search capability: DuckDuckGo search + HTTP page fetch.

URL blacklist prevents fetching data API endpoints — those return large payloads
that would explode the LLM context. Only HTML pages are allowed.
Response size is capped at _MAX_RESPONSE_CHARS to minimize token consumption.

To use the 'code' capability instead for fetching large data files, see code.py.
"""

_BLOCKED_URL_PREFIXES = (
    "https://www.data.gouv.fr/api/",  # REST API paths (bulk data)
    "https://tabular-api.data.gouv.fr/",  # entire tabular API domain
    "https://static.data.gouv.fr/resources/",  # large static resource files
)
_MAX_RESPONSE_CHARS = 50_000


async def http_fetch(url: str) -> str:  # noqa: D401
    """Fetch a web page. Data API URLs are blocked — use the 'code' capability to fetch data."""
    if any(url.startswith(p) for p in _BLOCKED_URL_PREFIXES):
        raise ValueError(
            f"URL blocked for web_search (data API path) — use 'code' capability to fetch data. URL: {url}"
        )
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url, headers={"User-Agent": "datagouv-agent/1.0"})
        response.raise_for_status()
    text = response.text
    if len(text) > _MAX_RESPONSE_CHARS:
        text = (
            text[:_MAX_RESPONSE_CHARS]
            + f"\n[truncated — response exceeded {_MAX_RESPONSE_CHARS} chars]"
        )
    return text


def web_search_toolset() -> list:
    """Return [duckduckgo_search_tool, http_fetch_tool] for the web_search capability."""
    try:
        from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
    except ImportError as exc:
        from agent_eval.experiment.agent.builder import CapabilityUnavailableError

        raise CapabilityUnavailableError(
            "Capability 'web_search' requires the 'ddgs' package. "
            "Install it with: uv add ddgs  (or pip install ddgs)"
        ) from exc

    from pydantic_ai.tools import Tool

    return [duckduckgo_search_tool(), Tool(http_fetch)]
