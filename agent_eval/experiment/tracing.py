import logging
import os
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_INITIALIZED = False


def _check_endpoint(endpoint: str, timeout: float = 5.0) -> None:
    """Raise RuntimeError if the Opik OTLP endpoint is not reachable via TCP."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Opik OTLP endpoint {endpoint!r} is not reachable: {exc}. "
            "Is your local Opik instance running? "
            "Set OTEL_EXPORTER_OTLP_ENDPOINT in your .env "
            "(e.g. http://localhost:5173/api/v1/private/otel)."
        ) from exc


def setup_tracing() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:5173/api/v1/private/otel"
    )
    _check_endpoint(endpoint)

    # Intentionally NOT calling logfire.instrument_pydantic_ai(): it instruments
    # every pydantic-ai agent globally (task agents AND every LLM judge), flooding
    # Opik with hundreds of orphan "agent run" OTLP traces. Agent activity is traced
    # via the manual @opik.track spans in task.py, which the eval engine links to
    # experiment items. This call only pre-flights that the Opik instance is up.
    _INITIALIZED = True
