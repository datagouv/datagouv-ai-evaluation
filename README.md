# datagouv-ai-evaluation

Evaluation framework for AI applications at [data.gouv.fr](https://data.gouv.fr), starting with the MCP server.

The framework runs agents against a benchmark of real user tasks, scores their responses with LLM-as-a-judge metrics, and tracks results in Opik.

---

## Why evaluate the MCP server?

An MCP server is not only an infrastructure component — it shapes how an agent understands and queries the data catalog. The names, descriptions, and argument structures of MCP tools directly influence whether an agent finds the right dataset, calls the right tool, or wastes tokens on redundant calls.

This framework evaluates:
- whether agents can find the right datasets and resources using the MCP tools
- how efficiently they do it (token usage, latency, number of tool calls)
- how different MCP versions, models, and system prompts compare
- whether a new MCP version introduces regressions (CI/CD integration)

---

## Prerequisites: local Opik instance

This framework requires a **self-hosted Opik instance running in Docker**. Opik handles dataset management, experiment tracking, trace visualization, and metric aggregation.

To install Opik locally:
1. Clone the Opik repository: `git clone https://github.com/comet-ml/opik.git`
2. Start it: `cd opik && ./opik.sh`
3. Open the UI at [http://localhost:5173](http://localhost:5173)

Full instructions: [Opik local deployment guide](https://www.comet.com/docs/opik/self-host/local_deployment)

The Opik OTLP tracing endpoint used by this framework is `http://localhost:5173/api/v1/private/otel`.

---

## Repository structure

```
mcp_eval/
├── benchmark/
│   └── config/                  # Experiment cross-product configuration
│       ├── evaluation_types.yml # Evaluation types (data_contamination, mcp_versions, …)
│       ├── models.yml           # Models to evaluate (active: true/false per entry)
│       ├── mcp_versions.yml     # MCP server versions and URLs
│       └── system_prompts.yml   # Named system prompts
│
├── tasks/
│   └── config/                  # Task YAML files (task_0001.yml, …)
│
├── evaluators/
│   ├── core/                    # Pure evaluation logic (no Opik dependency)
│   │   ├── prompts/             # LLM judge prompt builders
│   │   ├── config/
│   │   │   └── failure_modes.yml
│   │   ├── result_accuracy.py
│   │   ├── tool_usage.py
│   │   ├── tool_params.py
│   │   ├── trajectory.py
│   │   ├── failure_modes.py
│   │   ├── schema_compliance.py
│   │   └── efficiency.py
│   └── opik/                    # Opik BaseMetric wrappers
│       ├── result_accuracy_metric.py   # Also runs efficiency + failure modes
│       ├── tool_usage_metric.py
│       ├── trajectory_metric.py
│       └── experiment_metrics.py       # Experiment-level aggregates
│
└── experiment/
    ├── run_experiments.py       # CLI entry point
    ├── run_config.py            # Fetches MCP tool schemas per version
    ├── task.py                  # Agent runner + Opik trace hooks
    ├── mcp_tools_getter.py      # MCP tool discovery
    └── tracing.py               # Opik/logfire tracing setup

doc/
├── README_legacy.md             # Original planning document
├── metrics.md                   # Metric definitions and formulas
└── implementation_notes.md      # Architecture decisions and dead ends
```

---

## Evaluation types

| Type | Description |
|---|---|
| `data_contamination` | Baseline: agent with no MCP tools. Measures what the model knows from training data alone. |
| `mcp_versions` | Agent with MCP enabled. Compares across MCP server versions. |
| `capabilities_benchmark` | Compares different capability sets (MCP only, MCP + web search, MCP + code execution). |

Each type defines which metrics to compute and which system prompts to apply.

---

## Tasks

Each task is a YAML file in `mcp_eval/tasks/config/`. A task defines:
- **prompt**: the user question sent to the agent
- **evaluation_criteria**: what constitutes a correct answer, at minimal and optimal levels
- **tool_chain**: expected MCP tool calls (name, arguments) at minimal and optimal levels
- **resources**: data.gouv.fr datasets/resources referenced, with validation checks

Only tasks with `meta.status: active` are included in runs. The `example_task_0000.yml` shows the format.

---

## Metrics

See [doc/metrics.md](doc/metrics.md) for full definitions and formulas.

Key groups:
- **Result accuracy**: are the evaluation criteria met? (minimal / optimal levels)
- **Tool usage**: did the agent call the right tools with correct parameters?
- **Trajectory adherence**: did the agent follow the expected tool call sequence?
- **Efficiency**: token usage, latency, and their ratio to result accuracy
- **Failure modes**: binary flags for named failure patterns (hallucination, early stop, …)

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy `.env.example` to `.env` (or create `.env`) and fill in:

```bash
# Model API keys
MISTRAL_API_KEY=...
OPENAI_API_KEY=...       # only needed if using OpenAI models

# Judge model — REQUIRED, no default fallback
JUDGE_MODEL=mistral:mistral-medium-latest

# Opik tracing
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5173/api/v1/private/otel
OTEL_METRICS_EXPORTER=none
```

### 3. Configure Opik SDK

```bash
opik configure  # point to http://localhost:5173
```

---

## Running evaluations

```bash
# Run all active evaluation types
python -m mcp_eval.experiment.run_experiments

# Run a single evaluation type
python -m mcp_eval.experiment.run_experiments --evaluation-type mcp_versions

# Smoke test (1 task, no resource validation)
python -m mcp_eval.experiment.run_experiments --nb-samples 1 --no-validate --dry-run

# Override the judge model
python -m mcp_eval.experiment.run_experiments --judge-model openai:gpt-4o
```

The run validates all referenced data.gouv.fr resources before executing (can be skipped with `--no-validate`). Results appear in the Opik UI at [http://localhost:5173](http://localhost:5173) under the configured project name.

---

## Trace structure in Opik

Each task execution produces a nested trace:

```
evaluation_task
  task
    agent:<model-name>          ← LLM call: prompt, answer, token usage
      tool.<tool-name>          ← each MCP tool call: arguments + result
      …
  metrics_calculation
    result_accuracy
    tool_usage
    trajectory_adherence
```

---

## Adding tasks

Create `mcp_eval/tasks/config/task_XXXX.yml` following the format in `example_task_0000.yml`. Set `meta.status: active` when the task is ready. Run with `--no-validate --dry-run` first to check the YAML parses correctly.

## Adding models or MCP versions

Edit `mcp_eval/benchmark/config/models.yml` or `mcp_versions.yml`. Set `active: true` for entries to include in the next run. No code changes needed.
