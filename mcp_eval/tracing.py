import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.mcp import MCPInstrumentor

_INITIALIZED = False


def setup_tracing() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    project_name = os.getenv("OPIK_PROJECT_NAME", "datagouv_mcp")
    endpoint = os.getenv("OPIK_OTLP_ENDPOINT", "http://localhost:4317")

    tracer_provider = TracerProvider(
        resource=Resource({"service.name": project_name})
    )
    exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

    MCPInstrumentor().instrument(tracer_provider=tracer_provider)

    _INITIALIZED = True
