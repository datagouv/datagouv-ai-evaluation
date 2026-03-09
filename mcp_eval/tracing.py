from phoenix.otel import register
from openinference.instrumentation.mcp import MCPInstrumentor

_INITIALIZED = False


def setup_tracing() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    tracer_provider = register(
        project_name="datagouv_mcp",
        endpoint="http://localhost:4317",
        protocol="grpc",
        auto_instrument=True,
        batch=False,
    )

    MCPInstrumentor().instrument(tracer_provider=tracer_provider)

    _INITIALIZED = True
