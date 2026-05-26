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

All metrics are implemented as Opik `BaseMetric` subclasses in `agent_eval/evaluators/opik/`. The actual LLM judge logic lives in `agent_eval/evaluators/core/` (no Opik dependency), called via `asyncio.run()` from the metric `score()` method.

### Why failure modes are in `ResultAccuracyMetric`

Opik metrics run independently and cannot share Python objects — only numeric scores flow between them via `kwargs`. Failure mode detection needs context that only `ResultAccuracyMetric` has: the list of failed criteria (with reasons), actual tool calls, and required tool chains. Moving failure mode detection into `ResultAccuracyMetric` was the only way to give the judge enough signal without duplicating LLM calls.

### Failure modes scenario scope

`judge_failure_modes` currently receives both `required_tools_minimal` and `required_tools_optimal` but the judge prompt effectively reasons against the **optimal** scenario (the richer context). This is implicit and blurry — the judge has no explicit instruction to treat the two levels differently. **TODO:** decide whether failure modes should be scoped to a single level (optimal is the natural choice) and update the prompt accordingly.

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

### Span tree detail (current implementation)

```
evaluate() trace
└── agent:<model-name>              type="llm" — prompt / final answer / token usage
    ├── llm_turn_1                  type="llm" — output={"text":"…"} or {} (tool-only turns)
    │   ├── <tool_name>             type="tool" — input=args, output=result[:4000]
    │   └── thinking                type="llm" — output={"output":"<reasoning>"} (only if ThinkingPart present)
    ├── llm_turn_2
    │   └── <tool_name>
    └── llm_turn_N                  final answer turn, output={"text":"…"}
```

One `llm_turn_N` span is created per `ModelResponse` in `run_result.new_messages()`. Tool-only turns (model called tools without prose) have `output={}` — this is correct, not a bug.

### Opik SDK gotchas

- **`type="general"` spans have no text area in the UI** — use `type="llm"` for any span where you want the output displayed as readable text. General spans only show a collapsed JSON blob.
- **`capture_output=False` does NOT overwrite `update_current_span(output=…)`** — `SpanData.update()` skips `None` values, so the finalisation step is a no-op when the function returns `None`. The output set manually inside the function is preserved.
- **`tools=individual_tools` not `tools=individual_tools or None`** — pydantic-ai's `_AgentFunctionToolset.__init__` iterates over the argument; passing `None` raises `TypeError: 'NoneType' object is not iterable` for any run where no function-type tools are registered (e.g. MCP-only capability sets).
- **`evaluate()` needs `project_name=`** — without it, all experiment traces land in "Default Project" regardless of `OPIK_PROJECT_NAME` in `.env`.

### Debugging tips

`[msg-diag]` log lines in `_log_messages_as_spans` are at `DEBUG` level. Re-enable with `LOG_LEVEL=DEBUG` (or `basicConfig(level=logging.DEBUG)`) to see per-message part counts and first 80 chars of each TextPart / ThinkingPart.

### Model behaviour — intermediate reasoning text vs. ThinkingPart

Two distinct things appear as "reasoning" in traces:

**1. `TextPart` alongside tool calls** — prose the model emits in the same `ModelResponse` as a `ToolCallPart`. Visible as non-empty `llm_turn_N` output text blob in Opik. Model-dependent:

| Model | Behaviour | Opik appearance |
|---|---|---|
| `mistral-medium-latest` | One tool at a time; emits reasoning text before each subsequent tool call | Text blob inside `llm_turn_N` |
| `gpt-4o` | Batches multiple tool calls in parallel, silently; text only in final answer | `llm_turn_N` output empty for tool turns |
| Claude 3.x | One tool at a time + reasoning text | Text blob inside `llm_turn_N` |

**2. `ThinkingPart`** — a dedicated extended-reasoning block, separate from `TextPart`. Appears as a separate `thinking` child span **nested under** `llm_turn_N` in Opik. Produced by:
- `openai/gpt-oss-120b` (Albert API) — confirmed ✅ reasoning appears in `thinking` child span
- Claude 3.7+ with extended thinking explicitly enabled in the API call
- Magistral models (`magistral-medium-2506` etc.) via `MistralThinkChunk`

`mistral-medium-latest` is **not** a reasoning model and never produces `ThinkingPart`. If Langfuse shows "reasoning" for a standard Mistral run, it is capturing `TextPart` prose between tool calls — the same content that appears in `llm_turn_N` spans in Opik.

Note: if a model returns reasoning in a top-level HTTP field (`reasoning_content`) rather than inside the `content` array, pydantic-ai does **not** map it to `ThinkingPart`. logfire can see it via raw HTTP capture; our Opik spans will not.

---

## pydantic-ai breaking change: `result_type` → `output_type`

pydantic-ai renamed the `Agent` constructor parameter from `result_type` to `output_type` in a breaking release. All four core evaluator files (`result_accuracy.py`, `tool_params.py`, `trajectory.py`, `failure_modes.py`) were updated.

---

## File layout after reorganisation

After an initial flat layout, files were reorganised into subdirectories:

| Old location | New location |
|---|---|
| `agent_eval/tracing.py` | `agent_eval/experiment/tracing.py` |
| `agent_eval/evaluators/prompts/` | `agent_eval/evaluators/core/prompts/` |
| `agent_eval/evaluators/failure_modes.yml` | `agent_eval/evaluators/core/config/failure_modes.yml` |
| `agent_eval/benchmark/*.yml` | `agent_eval/benchmark/config/*.yml` |
| `agent_eval/tasks/task_*.yml` | `agent_eval/tasks/config/task_*.yml` |

**Critical**: `agent_eval/evaluators/core/prompts/__init__.py` must exist (empty file). Without it, all `from agent_eval.evaluators.core.prompts import ...` imports fail at runtime with `ModuleNotFoundError`.

Path constants updated accordingly:
- `BENCHMARK_DIR` → `agent_eval/benchmark/config`
- `TASKS_DIR` → `agent_eval/tasks/config`
- `_DEFAULT_FAILURE_MODES_PATH` → `agent_eval/evaluators/core/config/failure_modes.yml`

Three legacy evaluator files were deleted (`evaluators/experiment_metrics.py`, `evaluators/tool_invocation_judge.py`, `evaluators/tool_selection_match.py`) along with the now-superseded `evaluators/opik/failure_mode_metric.py`.

---

## Resource validation scoping

`validate_all_tasks()` runs HTTP checks against live data.gouv.fr resources. When `--nb-samples N` is passed (e.g. for a smoke test), only the first N tasks are validated, not the full set. This avoids validating resources for tasks that won't run.

## Other notes

- does not truncate trace tool results because it is hard to debug # done
- add albert api for model provider # done
- add good mcp versions number # done
- remove the backoff factor from latency calculation to avoid biaising latency results for some providers : compute net_lantecy_ms for lantency_ms now # done

---

## Semantic action layer

Task YAMLs define ground-truth tool sequences using **semantic action names** (`search.datasets`, `get.dataset.info`, etc.) rather than MCP tool names. The semantic layer (`agent_eval/benchmark/config/semantic_layer.yml`) maps (action, framework, version) → concrete tool names at evaluation time.

`SemanticLayerResolver` (`agent_eval/evaluators/core/semantic_layer.py`) handles this resolution via `packaging.specifiers.SpecifierSet` for version matching. `ToolUsageMetric.score()` calls `_resolve_to_concrete()` before matching actual calls against ground truth.

Semantic action → MCP tool name mapping (version `<=0.2.24`):
- `search.datasets` → `search_datasets`
- `search.dataservices` → `search_dataservices`
- `get.dataset.info` → `get_dataset_info`
- `get.dataset.resources` → `list_dataset_resources`
- `get.resource.info` → `get_resource_info`
- `get.resource.profile` → `query_resource_data` (proxy; no dedicated MCP tool)
- `get.data` → `query_resource_data` (all rows, no filtering)
- `analyze.data` → `query_resource_data` (with filters/aggregations)
- `get.dataservice.info` → `get_dataservice_info`
- `get.dataservice.openapi_spec` → `get_dataservice_openapi_spec`

`get.resource.info` (basic metadata) and `get.resource.profile` (tabular schema) are **distinct actions** even though both hit `query_resource_data` in MCP. The API framework uses different endpoints: `GET /datasets/{id}/resources/{rid}/` vs `GET /resources/{rid}/profile/`.

---

## Capabilities vs. semantic actions

**Capabilities** (`mcp`, `web_search`, `code`, `skills`) are agent execution modes. Semantic actions are framework-agnostic intentions. The key distinction:

- **`mcp` capability**: directly implements semantic actions via named MCP tool calls. Each semantic action maps to a concrete tool name (see mapping above).
- **`web_search` capability**: gives the agent DuckDuckGo search + HTML page fetching for **discovery and browsing**. It does NOT implement semantic actions — the URL blacklist specifically prevents fetching data API endpoints. The agent can browse dataset HTML pages (e.g. `https://www.data.gouv.fr/datasets/xxx`) but cannot call the REST API through `http_fetch`.
- **`code` capability**: gives the agent Docker-based Python execution + CLI, which can fetch data URLs internally (only the result reaches LLM context).
- **`skills` capability**: system prompt injection only — no tool calls.

### `web_search` capability

Two tools: `duckduckgo_search_tool()` (from `pydantic_ai.common_tools.duckduckgo`, requires `ddgs` package) + custom `http_fetch` tool (httpx-based).

**URL prefix blacklist** — prevents large data payloads in LLM context:
- `https://www.data.gouv.fr/api/` (REST API paths)
- `https://tabular-api.data.gouv.fr/` (entire tabular API domain)
- `https://static.data.gouv.fr/resources/` (large static files)

HTML pages at `https://www.data.gouv.fr/` (dataset pages, search result pages) are allowed; only the `/api/` path prefix is blocked. This means `web_search` agents can browse and discover datasets but cannot query structured data directly. Response is truncated at 50,000 chars before returning to the LLM.

### `code` capability — Docker-based local execution

Two tools running in the same Docker image (`datagouv-agent:latest`):
- `execute_python(code: str) → str` — arbitrary Python; only stdout returns to LLM
- `execute_cli(command: str) → str` — whitelisted commands only: `datagouv`, `python`/`python3`, `ls`, `mkdir`, `rm`, `rmdir`, `cp`, `mv`, `cat`, `echo`, `touch`, `curl`, `wget`, `pip`/`pip3`

Docker flags: `--cap-drop ALL --no-new-privileges --memory 512m --cpus 1 --network bridge`

**Why Docker?** Provider-native `CodeExecutionTool` doesn't work with Albert API and has no internet access. Docker is universal (all providers) and the bridge network lets code fetch data APIs directly — only the result (stdout) reaches the LLM context, avoiding the token explosion from piping large files through `web_search`.

**Security model:** `--cap-drop ALL` removes all Linux capabilities (no raw sockets, no privilege escalation, no port binding). No host volume mounts. Bridge network isolates the container from host loopback but allows outbound internet. CLI whitelist prevents arbitrary shell commands at application level; Docker handles OS-level isolation.

**Docker image build:** `docker build -t datagouv-agent:latest agent_eval/experiment/agent/`

**Data flow comparison:**
- `web_search → http_fetch → large JSON` → **blocked**: API URLs are blacklisted; even if not blocked, the 50K char cap would truncate large responses
- `code → execute_python("import requests; data = requests.get(url).json(); print(len(data['data']))")` → **good**: only the count/result reaches LLM context; full data stays in container

### `skills` capability

System prompt injection only — no tool calls, no action chain entries. Skills content is loaded from `benchmark/config/skills/` via `load_skills_prompt()` and appended to the agent's system prompt by `run_config.py` before the experiment runs.

---

challenges:
- many platforms, open source x all criteria
- unlike traditional ML, way more metrics, and a metric name across papers and platforms has not always the same definition/formula
- balance quality/reproducibility (often required more determinism and/or manual labeling) vs scalability/statistical power/versatility (more llm-as-a-judge) of evaluations
