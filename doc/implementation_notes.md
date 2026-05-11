# Implementation Notes

Decisions, dead ends, and things to remember so we don't re-litigate them.

---

## Evaluation platform: why Opik

Several platforms were benchmarked (see `README_legacy.md` for details). The decisive criterion was **aggregated experiment-level metrics**: being able to compute averages across a full dataset, not just per-task scores. Opik was the only self-hosted option that supported this natively via its `evaluate()` API. Phoenix Arize was tested first and dropped for this reason.

Opik must be run locally via Docker — see the README for installation instructions. The Python SDK (`opik`) handles dataset management, experiment runs, and scoring. All trace data is stored locally.

---

## Benchmark cross-product

The evaluation runs a cross-product of:
- **Evaluation types** (`evaluation_types.yml`): e.g. `data_contamination`, `mcp_versions`, `capabilities_benchmark`
- **Models** (`models.yml`): list of `provider:model-name` strings
- **MCP versions** (`mcp_versions.yml`): versioned server URLs
- **System prompts** (`system_prompts.yml`): named prompt templates

Each YAML entry has an `active: true/false` flag. Only active entries enter the cross-product. This lets you toggle specific models or versions without deleting config.

System prompts can be filtered per evaluation type via the `system_prompts` key in `evaluation_types.yml` — accepts a list of names, `"active"` (default), or `"all"`.

---

## Judge model

The judge model (used for LLM-as-a-Judge metrics) must be set via `JUDGE_MODEL` in `.env` or passed as `--judge-model` on the CLI. There is **no default fallback** — the run will exit with an error if neither is provided.

Early versions had a silent fallback to `openai:gpt-4o-mini` in both argparse and each metric class `__init__`. This caused the wrong (and expensive) model to be used silently when the env var wasn't loaded in time. The fix: resolve the model in `main()` after `load_dotenv()` and pass it explicitly to every metric constructor.

---

## Metric architecture

All metrics are implemented as Opik `BaseMetric` subclasses in `mcp_eval/evaluators/opik/`. The actual LLM judge logic lives in `mcp_eval/evaluators/core/` (no Opik dependency), called via `asyncio.run()` from the metric `score()` method.

### Why failure modes are in `ResultAccuracyMetric`

Opik metrics run independently and cannot share Python objects — only numeric scores flow between them via `kwargs`. Failure mode detection needs context that only `ResultAccuracyMetric` has: the list of failed criteria (with reasons), actual tool calls, and required tool chains. Moving failure mode detection into `ResultAccuracyMetric` was the only way to give the judge enough signal without duplicating LLM calls.

### Why efficiency metrics are in `ResultAccuracyMetric`

Efficiency metrics (latency, token usage, time/token efficiency) were originally in `ToolUsageMetric`. The `data_contamination` evaluation type does not include `tool_usage` in its metrics list, so efficiency was never computed for those experiments. Moving efficiency to `ResultAccuracyMetric` (which always runs) fixed the silent omission.

---

## Tracing: what was tried and what works

We need traces to show the agent's internal execution (LLM calls, MCP tool calls) nested inside the Opik evaluation experiment trace.

### Attempt 1: Raw OTEL gRPC (port 4317) — failed

Initial `tracing.py` used `opentelemetry-exporter-otlp-proto-grpc` pointing to `localhost:4317`. Opik's local Docker setup does not run an OTEL collector on that port. The exporter silently retried on every task, adding latency.

**Fix**: Added a TCP preflight check in `setup_tracing()` that raises `RuntimeError` immediately if the endpoint is unreachable, so failures are loud rather than silent.

### Attempt 2: logfire + OTEL context bridge — failed to nest

Switched to `logfire` (the correct Opik/pydantic-ai integration). The Opik pydantic-ai integration doc specifies:
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5173/api/v1/private/otel
```
(not port 4317 — port 5173 is the Opik frontend which proxies OTLP).

`logfire.configure(send_to_logfire=False)` + `logfire.instrument_pydantic_ai()` sends pydantic-ai spans to Opik via OTLP. However, these appear as **separate, standalone traces** in the Opik UI — not nested under the evaluation experiment trace.

We tried injecting the Opik evaluation span's `trace_id` and `id` as the OTEL parent context (converting UUID → 128-bit int / 64-bit int) before `asyncio.run()`. This did not connect the traces. The Opik backend stores SDK-created spans and OTLP-ingested spans in separate namespaces: even if they share the same trace_id format, they are not joined in the UI.

**Conclusion**: The Opik OTLP endpoint and the Opik SDK are two separate ingestion paths. You cannot mix them in the same trace view.

### Attempt 3: Opik SDK `@opik.track` — works

The solution is to stay entirely within the Opik SDK. Using `@opik.track` on the agent runner function creates spans via the same SDK path as the evaluation framework, so they nest naturally.

`_run_agent_and_log()` (decorated with `@opik.track(type="llm")`) runs inside `asyncio.run()`, which copies the current `contextvars.Context` (where Opik's `task` span is active). The decorator reads that context and creates `agent:model-name` as a child span. Each tool call is then logged as a further child via `_log_tool_call()` (decorated with `@opik.track(type="tool")`).

The resulting trace hierarchy:
```
evaluation_task
  task
    agent:mistral-small-latest   ← @opik.track, includes prompt/answer/tokens
      tool.search_datasets        ← @opik.track, includes args + result[:500]
      tool.get_dataset_info
      …
  metrics_calculation
    result_accuracy
    tool_usage
    trajectory_adherence
```

**Important**: `logfire` is still installed as a dependency (pydantic-ai uses it for `instrument=True`), but `logfire.configure()` / `logfire.instrument_pydantic_ai()` are called in `setup_tracing()` and only affect the logfire OTLP pipeline (which is separate and can be ignored or removed). The actual trace nesting comes from `@opik.track`.

---

## pydantic-ai breaking change: `result_type` → `output_type`

pydantic-ai renamed the `Agent` constructor parameter from `result_type` to `output_type` in a breaking release. All four core evaluator files (`result_accuracy.py`, `tool_params.py`, `trajectory.py`, `failure_modes.py`) were updated.

---

## File layout after reorganisation

After an initial flat layout, files were reorganised into subdirectories:

| Old location | New location |
|---|---|
| `mcp_eval/tracing.py` | `mcp_eval/experiment/tracing.py` |
| `mcp_eval/evaluators/prompts/` | `mcp_eval/evaluators/core/prompts/` |
| `mcp_eval/evaluators/failure_modes.yml` | `mcp_eval/evaluators/core/config/failure_modes.yml` |
| `mcp_eval/benchmark/*.yml` | `mcp_eval/benchmark/config/*.yml` |
| `mcp_eval/tasks/task_*.yml` | `mcp_eval/tasks/config/task_*.yml` |

**Critical**: `mcp_eval/evaluators/core/prompts/__init__.py` must exist (empty file). Without it, all `from mcp_eval.evaluators.core.prompts import ...` imports fail at runtime with `ModuleNotFoundError`.

Path constants updated accordingly:
- `BENCHMARK_DIR` → `mcp_eval/benchmark/config`
- `TASKS_DIR` → `mcp_eval/tasks/config`
- `_DEFAULT_FAILURE_MODES_PATH` → `mcp_eval/evaluators/core/config/failure_modes.yml`

Three legacy evaluator files were deleted (`evaluators/experiment_metrics.py`, `evaluators/tool_invocation_judge.py`, `evaluators/tool_selection_match.py`) along with the now-superseded `evaluators/opik/failure_mode_metric.py`.

---

## Resource validation scoping

`validate_all_tasks()` runs HTTP checks against live data.gouv.fr resources. When `--nb-samples N` is passed (e.g. for a smoke test), only the first N tasks are validated, not the full set. This avoids validating resources for tasks that won't run.

## Other notes

- does not truncate trace tool results because it is hard to debug
- add albert api for model provider
- add good mcp versions number
