# datagouv-ai-evaluation

An evaluation framework for AI agent's data.gouv.fr toolsets — measuring response quality, cost efficiency, and viability across different models and capability configurations.

## At a glance

A configurable cross-product of **evaluation type × model × capabilities (MCP, cli, skills.md...) × system prompt** runs against a curated set of real data.gouv.fr questions. Each agent run is scored with LLM-as-a-judge and deterministic metrics, aggregated at experiment level, and stored in self-hosted [Opik](https://www.comet.com/docs/opik/) so you can compare configurations side by side.

**Quick start:** start Opik (`./opik.sh`), `uv sync`, copy `.env.example` to `.env`, then `uv run python -m agent_eval.experiment.run_experiments`. Full walkthrough in [§3 Setup](#setup--running-evaluations).

**Navigate this README:** [1. Purpose & Goals](#1-purpose--goals) · [2. Core Concepts](#2-core-concepts) · [3. Technical Documentation](#3-technical-documentation) · [4. Limitations](#4-limitations)

For development decisions, dead ends, and gotchas accumulated along the way, see [`doc/implementation_notes.md`](doc/implementation_notes.md).

---

## 1. Purpose & Goals

### Why this framework?

data.gouv.fr offers several ways for AI agents to access the French open data catalog: an MCP server, a REST API, a Python SDK and a CLI. A skills.md has also been developed to improve AI agents usage fo these methods. The quality of these solutions — their tool/endpoint/class/command descriptions, argument design, and data coverage — directly shapes whether an agent finds the right dataset or dataservices and how to leverage it, calls the right solutions the right way, or wastes tokens on redundant queries.

This framework evaluates agents across three axes:

- **Response quality** — does the agent find the right resource and answer the user's question correctly?
- **Cost efficiency** — how many tokens and how much time does it take? Can accuracy be maintained while reducing consumption?
- **Smaller model accessibility** — do open-source - often smaller - models perform adequately, or does quality collapse below a certain model size?

Practically, this means the framework can be used to:
- **Compare toolset and skills.md versions** (e.g. non-regression testing as the MCP server evolves)
- **Benchmark capability combinations** (MCP only vs. MCP + web search vs. MCP + code execution)
- **Measure the impact of other environment parameters** like system prompts or models changes
- **Identify failure patterns** (hallucinations, missing caveats, early stops) to guide toolset improvements

### Capabilities in scope

| Capability | Status* |
|---|---|
| MCP server (`mcp.data.gouv.fr`) | Active |
| data.gouv.fr REST API + Tabular API (via code execution) | Active |
| `datagouv-client` CLI (via code execution) | Active |
| Skills.md | Active |
| Python SDK (via code execution) | Roadmap |

*Status means whether the capability is available as a config parameter for evaluations

---
## 2. Core Concepts

### The Benchmark

The benchmark is a **cross-product** of what you are measuring (evaluation type) and what can vary (modalities).

#### Evaluation types

Each evaluation type defines a specific question to answer, e.g. :

| Type | Question |
|---|---|
| `data_contamination` | What does the model know from training data alone, without any tools? |
| `mcp_versions` | Does a new MCP server version improve or regress against the previous one? |
| `capabilities_benchmark` | General comparison across active capability configurations |

Each type specifies which capabilities, system prompts, and metrics apply. Configuration lives in `agent_eval/benchmark/config/evaluation_types.yml`.

#### Modalities

Within an evaluation type, the following dimensions can vary:

- **Models** — Mistral, OpenAI, Albert (Etalab). All use the OpenAI-compatible API. Configured in `benchmark/config/models.yml` with `active: true/false`.
- **MCP versions** — version-pinned server URLs. Configured in `benchmark/config/mcp_versions.yml`.
- **Skills.md versions** — Configured in `benchmark/config/skills_versions.yml` (referenced version to be added to `benchmark/config/skills/`). Skills.md is currently injected into the system prompt.
- **System prompts** — named variants from `benchmark/config/system_prompts.yml`
- **Capabilities** — the toolset combination given to the agent: `[mcp]`, `[mcp, web_search]`, `[mcp, code]`, `[code, datagouv-cli]`, etc.

Each run is one combination of `(evaluation_type × model × capabilities [× MCP version][× skills.md version] × system prompt)`. The cross-product is built automatically from the active entries in each YAML file.

---

### The Semantic Layer

#### The problem

The same user intent — "find a dataset about sports facilities" — can be realised as:
- An MCP tool call: `search_datasets(query="équipements sportifs")`
- A REST API call: `GET /api/1/datasets/?q=équipements+sportifs`
- A CLI command: `datagouv dataset list --query "équipements sportifs"`
- A code block hitting the same API endpoint

Evaluating "did the agent do the right thing?" - and not only "did it answer right ?" - across all these implementations requires abstracting over the concrete tool names and frameworks.

#### How it works

The semantic layer maps each **semantic action name** to its concrete implementations per framework and version:

- `semantic_layer/config/actions.yml` — defines which MCP tool, CLI command, or API endpoint corresponds to each semantic action, with version specifiers (PEP 440 syntax)
- `semantic_layer/config/action_args.yml` — maps semantic argument names to framework-specific argument names
- `semantic_layer/resolver.py` — version-aware resolver used at evaluation time

At evaluation time, the action mapper (`evaluators/core/action_mapper.py`) classifies each literal tool call into one or more semantic action instances. Classification is:
- **Deterministic** for MCP tool calls and known CLI commands
- **LLM-judged** for code execution (Python scripts, curl calls, CLI one-liners) and unrecognised tools

This allows tasks to be written once and evaluated against any toolset implementation.

#### Defined semantic actions

| Action | What it means |
|---|---|
| `search.datasets` | Keyword or semantic search for datasets in the catalog |
| `search.dataservices` | Search for API dataservices in the catalog |
| `get.dataset.info` | Retrieve a specific dataset's metadata |
| `get.dataset.resources` | List the files and resources attached to a dataset |
| `get.resource.info` | Retrieve a specific resource's metadata |
| `get.resource.profile` | Get the tabular schema and column statistics of a resource |
| `get.data` | Fetch actual data rows from a resource |
| `get.dataservice.info` | Retrieve a dataservice's metadata |
| `get.dataservice.openapi_spec` | Fetch the OpenAPI specification of a dataservice |

See `doc/vocabulary.md` for the naming convention (`action.object_type[.facet]`).

---

### Tasks

#### Source

Task prompts are inspired from real questions extracted from the data.gouv.fr community forum (`forum.data.gouv.fr`) and support channels (messages are anonymized and rephrased). This grounds the evaluation in actual user difficulties: ambiguous requests, multi-source answers, resources that aren't queryable via the tabular API, datasets that require caveats about size or access.

#### Human-reviewed labeling

Each task is manually annotated with expected outcomes. Rather than writing a too restrictive reference answer, tasks define more lasting criteria :

- **Evaluation criteria** — what properties the response must have ("agent identifies the FINESS dataset", "agent explains that the CSV is not queryable via the tabular API")
- **Action chain** — which semantic actions the agent should perform, in which order, including parallel branches

This labeling approach is more scalable: new tool calls or catalog entries don't invalidate the criteria, and criteria can be evaluated automatically via LLM-as-a-judge across any number of runs.

#### The minimal / optimal split

Every task defines two performance tiers:

- **Minimal** — what a basic user would consider satisfactory: the agent found the right resource and gave usable information
- **Optimal** — what a more technically experienced user would expect or what a very good agent (performing the last mile) : the agent also explains caveats, access constraints, data quality issues, differences between sources, or how to use the data programmatically

All metrics are reported separately for minimal and optimal levels. This lets you distinguish "it works" from "it works very well", and identify which configurations close the gap between the two.

#### Production data drift

Even though the criteria system is made to last, some breaking changes can still happen : datasets removed from data.gouv.fr catalogue, unavailability of a resource on Tabular API... To prevent this from failing silently, before launching experiments, a script runs to validate that required resources haven't changed (see `agent_eval/tasks/resource_validator.py`, stored and git versioned data `agent_eval/tasks/data/` and the key `resources` in `agent_eval/tasks/config/example_task_0000.yml`)

#### Current set

10 tasks, covering dataset discovery, dataservice identification, tabular queries, and multi-source scenarios. Task prompts are in French, reflecting the actual user population. At 10 tasks, the evaluation is primarily **qualitative** — suitable for catching regressions and understanding failure patterns, but not statistically significant for fine-grained comparisons. See [Limitations](#4-limitations).

Task files: `agent_eval/tasks/config/task_XXXX.yml`. Format documented in `example_task_0000.yml`.

---

### Metrics

Four groups of metrics, all computed per task and aggregated to experiment level (averaged across the dataset for a given model × configuration combination).

The metric design is grounded in existing agent-eval frameworks — **DeepEval** (Task Completion, Tool Correctness, Step Efficiency, Plan Adherence), **RAGAS** (Tool Call F1), **Arize Phoenix** (Tool Selection / Invocation / Response Handling) — and recent tool-use evaluation papers. The exact score-name → source mapping lives in [`doc/metrics.md`](doc/metrics.md#sources-of-inspiration).

#### 1. Result Accuracy (LLM-as-a-judge)

Did the agent's response satisfy each evaluation criterion?

- Reported for **minimal** and **optimal** criterion sets separately
- Score per level: `validated_criteria / total_criteria` ∈ [0, 1]
- Each criterion is judged independently and concurrently

#### 2. Action Metrics (semantic space)

Three sub-dimensions, all operating on the semantic action instances produced by the action mapper:

**Action usage** (deterministic)
Measures whether the agent used the right *types* of actions:
- Precision, recall, F1 at the **action-type level** (unique action names: did the agent use `search.datasets` at all?)
- Precision, recall, F1 at the **action-instance level** (per call: did the agent looked up a specific dataset `id` resources through `get.dataset.resources` ?)
- Both computed for minimal and optimal action chains

**Action parameter correctness** (LLM-as-a-judge)
Measures whether the arguments passed to actions were correct:
- Actual action instances are matched against ground-truth required actions
- Judge evaluates `strict_value` parameters (exact match) and `criteria`-based parameters (semantic match)
- Score: `matched_actions / required_actions`

**Trajectory adherence** (LLM-as-a-judge)
Measures whether the agent followed the expected action sequence:
- Accounts for sequential steps (`A > B`) and parallel branches (`(A > B) + (C > D)`)
- Score ∈ [0, 1]; evaluated for minimal and optimal chains independently

#### 3. Efficiency (deterministic)

Raw measures:
- `latency_ms` — end-to-end wall time, excluding rate-limit wait time
- `token_usage` — total tokens consumed

Derived scores (higher = more result per unit cost):
- `token_efficiency` = `result_accuracy / (token_usage / 1000)`
- `time_efficiency` = `result_accuracy / (latency_ms / 60_000)`

Both derived scores are computed for minimal and optimal accuracy levels.

#### 4. Failure Modes (LLM-as-a-judge, binary flags)

Nine named failure patterns, each scored 0 (absent) or 1 (present):

| Mode | Description |
|---|---|
| `HALLUCINATION` | Invented dataset IDs, resource names, endpoints, or facts not grounded in tool results |
| `MISINTERPRETATION` | Correct tool results but wrong conclusions drawn from them |
| `WRONG_RESOURCE` | Selected the wrong type of dataset or dataservice for the user's need |
| `PARAMETER_ERROR` | Wrong, malformed, or missing tool call arguments |
| `TOOL_OMISSION` | Answered without tools when tools were clearly needed |
| `EARLY_STOP` | Incomplete tool chain — searched but never fetched details |
| `REDUNDANT_LOOP` | Repeated the same tool call without progress |
| `MISSING_CAVEAT` | Omitted a critical limitation (access restrictions, license, file size, deprecation) |
| `NO_FALLBACK` | No alternative offered when the primary option was unavailable |

For full formulas, every emitted score name, and the cross-platform / paper attribution table, see [`doc/metrics.md`](doc/metrics.md).

---

## 3. Technical Documentation

### Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Agent framework | [pydantic-ai](https://ai.pydantic.dev/) |
| Experiment tracking | [Opik](https://www.comet.com/docs/opik/) (self-hosted) |
| Models | MistralAI · OpenAI · Albert (Etalab) — all OpenAI-compatible |
| Code execution | Docker — isolated Python/CLI sandbox |
| Web search | DuckDuckGo (`ddgs`) + HTTP fetch |
| MCP client | `mcp` SDK (`MCPServerStreamableHTTP`) |

---

### Setup & Running Evaluations

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker (required only for `code` or `datagouv-cli` capabilities)

#### Step 1 — Start Opik (self-hosted)

```bash
git clone https://github.com/comet-ml/opik.git
cd opik && ./opik.sh
```

The Opik UI will be available at [http://localhost:5173](http://localhost:5173).
Full instructions: [Opik local deployment guide](https://www.comet.com/docs/opik/self-host/local_deployment)

#### Step 2 — Clone & install

```bash
git clone <repo-url>
cd datagouv-ai-evaluation
uv sync
```

#### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in the environment variables values.

At least one model API key is required. The active model(s) are controlled by `benchmark/config/models.yml`.

#### Step 4 — Configure Opik SDK

```bash
uv run opik configure   # enter http://localhost:5173 when prompted
```

#### Running evaluations

```bash
# Run all active evaluation types
uv run python -m agent_eval.experiment.run_experiments

# Run a single evaluation type
uv run python -m agent_eval.experiment.run_experiments --evaluation-type mcp_versions

# Limit to N tasks (for smoke-testing)
uv run python -m agent_eval.experiment.run_experiments --nb-samples 1

# Skip data.gouv.fr resource pre-flight checks
uv run python -m agent_eval.experiment.run_experiments --no-validate

# Build and validate configs without running any agents
uv run python -m agent_eval.experiment.run_experiments --dry-run
```

**Where to find results in Opik:**
- Experiment scores → **Datasets → `datagouv_tasks_v2` → Experiments tab**
- LLM traces → **Projects → `<OPIK_PROJECT_NAME>`**

---

### Repository Structure

```
agent_eval/
├── _env.py                          # Loads .env relative to project root
├── utils.py                         # OpenAI-compatible model wrapper
│                                    #   (rate-limit retry, Mistral quirk patches)
│
├── benchmark/
│   └── config/
│       ├── evaluation_types.yml     # Evaluation types: metrics, capabilities, prompts
│       ├── models.yml               # Models to evaluate (active: true/false)
│       ├── mcp_versions.yml         # MCP server version URLs (active: true/false)
│       ├── system_prompts.yml       # Named system prompt variants
│       └── skills/                  # Skills documents for system prompt injection
│
├── tasks/
│   ├── config/
│   │   ├── example_task_0000.yml    # Task format template — read before adding tasks
│   │   └── task_0001.yml … task_0010.yml
│   ├── data/                        # Snapshot cache for resource validation
│   ├── loader.py                    # Parses task YAMLs into Task dataclasses
│   └── resource_validator.py        # Pre-flight validation against data.gouv.fr API
│
├── semantic_layer/
│   ├── resolver.py                  # Version-aware action ↔ tool name resolver
│   └── config/
│       ├── actions.yml              # Semantic action → MCP / CLI / API implementations
│       └── action_args.yml          # Semantic arg name → framework-specific arg name
│
├── evaluators/
│   ├── core/                        # Pure evaluation logic — no Opik dependency
│   │   ├── _math.py                 # Shared safe_div, f1_score helpers
│   │   ├── action_mapper.py         # Literal tool calls → semantic action instances
│   │   ├── action_params.py         # LLM judge: action parameter correctness
│   │   ├── action_usage.py          # Deterministic action usage (precision/recall/F1)
│   │   ├── trajectory.py            # LLM judge: action sequence adherence
│   │   ├── result_accuracy.py       # LLM judge: evaluation criteria validation
│   │   ├── efficiency.py            # Latency + token efficiency scores
│   │   ├── failure_modes.py         # LLM judge: failure mode detection
│   │   ├── judge_model.py           # Judge model loader
│   │   ├── config/
│   │   │   ├── judge_model.yml      # Which model acts as the LLM judge
│   │   │   └── failure_modes.yml    # Failure mode names and descriptions
│   │   └── prompts/                 # Prompt builders, one per evaluator
│   └── opik/                        # Opik BaseMetric wrappers
│       ├── result_accuracy_metric.py    # result_accuracy + efficiency + failure_modes
│       ├── action_metric.py             # action_mapper + usage + params + trajectory
│       ├── tool_call_stats_metric.py    # Literal tool call counts (no LLM)
│       └── experiment_metrics.py        # Experiment-level aggregated scores
│
└── experiment/
    ├── run_experiments.py           # CLI entry point
    ├── run_config.py                # Builds run configs, fetches MCP tool schemas
    ├── task.py                      # Agent runner + Opik trace hooks
    ├── tracing.py                   # Opik endpoint reachability check
    ├── mcp_tools_getter.py          # MCP tool schema discovery
    └── agent/
        ├── builder.py               # Assembles toolsets from capability list
        ├── mcp.py                   # MCP server toolset
        ├── code.py                  # Docker-based Python/CLI execution
        ├── web_search.py            # DuckDuckGo search + HTTP fetch
        ├── skills.py                # Skills document injection
        └── Dockerfile               # Base image + datagouv-cli variant

doc/
├── metrics.md                       # Full metric formulas and definitions
├── vocabulary.md                    # Semantic layer naming conventions
└── implementation_notes.md          # Architecture decisions and known issues

tests/
└── test_action_metrics.py           # Unit tests for action mapper and usage metrics
```

---

## Code Linting and Formatting

This project follows PEP 8 style guidelines using [Ruff](https://astral.sh/ruff/) for linting and formatting.

**Either running these commands manually or [installing the pre-commit hook](#-pre-commit-hooks) is required before submitting contributions.**

```shell
# Lint and format code
uv run ruff check --fix && uv run ruff format
```

---

## Pre-commit Hooks

This repository uses a [pre-commit](https://pre-commit.com/) hook which lints and formats code before each commit. Installing the pre-commit hook is strongly recommended so the checks run automatically.

**Install pre-commit hooks:**

```shell
uv run pre-commit install
```

The pre-commit hook automatically:

- Check YAML syntax
- Fix end-of-file issues
- Remove trailing whitespace
- Check for large files
- Run Ruff linting and formatting

---

## Contributing

**Adding a task**

1. Create `agent_eval/tasks/config/task_XXXX.yml` following the format in `example_task_0000.yml`
2. Set `meta.status: draft` while iterating, `active` when ready
3. Bump `DATASET_VERSION` in `run_experiments.py` so the Opik dataset is refreshed
4. Validate with `--dry-run --no-validate` before running agents

**Adding a model**

Edit `benchmark/config/models.yml` and set `active: true` on the entry. No code changes required — the model must use an OpenAI-compatible API.

**Adding an MCP version**

Edit `benchmark/config/mcp_versions.yml`. If the new version renames tools, add entries to `semantic_layer/config/actions.yml` with the appropriate version specifiers.

**Adding an evaluation type**

Edit `benchmark/config/evaluation_types.yml`. Define which capabilities, system prompts, and metrics apply.

### Contributing rules

We welcome contributions! To keep the project stable and reviews manageable, please observe these rules before submitting:

- **Human review and accountability:** **Issues and pull requests** must not be raw, unreviewed AI output. You must have read, fully understood, and (for code) tested what you submit. **By opening an issue or a pull request, you certify you could explain and defend it in review without relying on an AI assistant.**
- **Keep it small:** We strictly follow a **1 feature = 1 PR** workflow.
- **Conventional commits:** Use the [Conventional Commits](https://www.conventionalcommits.org/) format for **git commit messages** and **PR titles** (e.g. `feat: add dataset search`, `fix: handle empty API response`). See the specification for allowed types, scopes, and breaking-change markers.

---

## Limitations

This framework is an early-stage MVP. The following limitations should be understood before interpreting results.

**Single human labeler.** All task ground truths — evaluation criteria and action chains — were defined by one person with no cross-labeling or inter-rater agreement measurement. Defining what constitutes a "correct" or "optimal" response to a data discovery question is not always straightforward, and the current labels reflect one person's judgment. Disagreement between labelers would be expected on several tasks.

**Small evaluation set.** 10 tasks is enough for qualitative assessment and catching obvious regressions, but too few for statistically significant comparisons between configurations. Metric differences of a few percentage points between two models or MCP versions should be treated as directional signals, not conclusions. Expanding the task set via semi-automated generation with human review is planned.

**Metrics under development.** The metric formulas — particularly trajectory adherence scoring bands, the minimal/optimal split thresholds, and the weighting of action-instance vs. action-type F1 — have not been validated against human agreement scores. Numeric scores are useful for relative comparisons within a run, but their absolute values are not calibrated.

**LLM-as-a-judge variance.** Result accuracy, trajectory adherence, parameter correctness, and failure mode detection all rely on a judge LLM. Judge outputs vary across runs, and the judge's own capabilities and biases affect every score. The judge has not been calibrated against human annotations.

**Implementation bias in code and CLI capabilities.** The `code` and `datagouv-cli` capabilities run in a hardened Docker container with restricted network access and a controlled Python environment. This setup does not reflect how these capabilities would be used in real coding agents (Claude Code, Cursor, Copilot Workspace), which have different execution environments, file system access, and permission models. Results for these configurations should be interpreted as performance within the evaluation harness, not as predictions of real-world agentic coding performance.

**Skills injection method.** When the `skills` capability is enabled, the full skills document is injected as a block into the system prompt. In production agent deployments, skills or tool documentation are often injected differently: as part of tool descriptions, retrieved dynamically from memory, or chunked to fit context constraints. The current injection method may over- or under-represent the practical benefit of having skills available, and doesn't reflect the context window pressure realistic deployments face.

**Static resource snapshots.** Resource pre-flight validation compares live data.gouv.fr API responses against pre-computed snapshots in `tasks/data/`. If a dataset is restructured, renamed, or removed (beyond what the snapshots track), a task may become unanswerable without the snapshot being refreshed.

**Single-turn only.** All tasks are single-turn: one user prompt, one agent response. Multi-turn conversations — where users refine their questions, ask for clarifications, or iterate on partial results — are not evaluated. Real user interactions frequently involve multiple turns.

**No cross-dataset or temporal evaluation.** Tasks are fixed to specific datasets and resources as they existed when the tasks were authored. The catalog evolves; a task that was well-defined when authored may become ambiguous or unanswerable if the referenced resource is updated, deprecated, or replaced.

---

## Releases and versioning

The release process uses the [`tag_version.sh`](tag_version.sh) script to create git tags, GitHub releases and update [CHANGELOG.md](CHANGELOG.md) automatically. Package version numbers are automatically derived from git tags using [setuptools_scm](https://github.com/pypa/setuptools_scm), so no manual version updates are needed in `pyproject.toml`.

**Prerequisites**: [GitHub CLI](https://cli.github.com/) must be installed and authenticated, and you must be on the main branch with a clean working directory.

```shell
# Create a new release
./tag_version.sh <version>

# Example
./tag_version.sh 2.5.0

# Dry run to see what would happen
./tag_version.sh 2.5.0 --dry-run
```

The script automatically:
- Extracts commits since the last tag and formats them for CHANGELOG.md
- Identifies breaking changes (commits with `!:` in the subject)
- Creates a git tag and pushes it to the remote repository
- Creates a GitHub release with the changelog content

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
