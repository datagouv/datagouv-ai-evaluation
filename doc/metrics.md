# Metrics

Score names listed here match exactly what `agent_eval/evaluators/opik/` emits and what shows up in Opik. All scores are in `[0, 1]` unless otherwise stated; higher is better unless flagged.

**Metric type:**

- **Computed** — deterministic from the trace / API response (counts, durations, tokens, exact matching).
- **LLM-judge** — an LLM-as-a-Judge call. The judge model is configured globally via `JUDGE_MODEL` and is independent of the model being evaluated.

For now, every task is a single-turn evaluation.

---

## Task-level metrics

Emitted by three metric classes:

| Class | File | What it produces |
|---|---|---|
| `ResultAccuracyMetric` | `evaluators/opik/result_accuracy_metric.py` | Answer correctness, efficiency, failure modes |
| `ToolCallStatsMetric` | `evaluators/opik/tool_call_stats_metric.py` | Literal tool-call counts |
| `ActionMetric` | `evaluators/opik/action_metric.py` | Semantic action usage, parameter correctness, trajectory adherence |

### Result accuracy (`ResultAccuracyMetric`)

LLM-judged correctness against per-task `evaluation_criteria` (minimal + optimal levels — see `agent_eval/tasks/config/example_task_0000.yml`).

| Score name | Type | Description |
|---|---|---|
| `result_accuracy_minimal` | LLM-judge | `total_validated_criteria_minimal / total_criteria_minimal` |
| `result_accuracy_optimal` | LLM-judge | `total_validated_criteria_optimal / total_criteria_optimal` |
| `total_validated_criteria_minimal` | LLM-judge | criteria the judge confirmed (minimal level, count) |
| `total_validated_criteria_optimal` | LLM-judge | criteria the judge confirmed (optimal level, count) |
| `total_criteria_minimal` | Computed | number of criteria defined for the task (minimal) |
| `total_criteria_optimal` | Computed | number of criteria defined for the task (optimal) |

### Efficiency (`ResultAccuracyMetric`)

| Score name | Type | Description |
|---|---|---|
| `latency_ms` | Computed | net inference time, ms — backoff/quota wait is subtracted via `CompatibleOpenAIChatModel.rate_limit_wait_ms` |
| `token_usage` | Computed | total tokens from the model usage object |
| `token_efficiency_minimal` / `_optimal` | Computed | `result_accuracy / token_usage` — quality per token |
| `time_efficiency_minimal` / `_optimal` | Computed | `result_accuracy / latency_ms` — quality per ms |

### Failure modes (`ResultAccuracyMetric`)

Binary flags (0/1) per failure mode, detected by an LLM judge. Not a quality score on its own — they enable aggregate failure analysis at experiment level.

The judge receives both minimal and optimal tool chains but implicitly reasons against the **optimal** scenario. Scoping is currently blurry — a future improvement should make this explicit in the prompt.

Defined failure modes: `HALLUCINATION`, `MISINTERPRETATION`, `WRONG_RESOURCE`, `PARAMETER_ERROR`, `TOOL_OMISSION`, `EARLY_STOP`, `REDUNDANT_LOOP`, `MISSING_CAVEAT`, `NO_FALLBACK`.

### Literal tool-call stats (`ToolCallStatsMetric`)

| Score name | Type | Description |
|---|---|---|
| `total_tool_calls` | Computed | number of pydantic-ai `ToolCallPart`s in the trace |

This is the tool-call efficiency signal: same task done in fewer literal calls is cheaper. Distinct from `total_actions_made` (semantic actions), which can be smaller (multiple literal calls collapsed into one semantic action) or larger (one literal call implementing several semantic actions, e.g. via `code`).

### Action usage (`ActionMetric`)

Built on the semantic action layer (`agent_eval/semantic_layer/`). The action mapper classifies every literal tool call into zero, one, or more semantic actions; everything below operates on those mapped actions.

**Counts:**

| Score name | Type | Description |
|---|---|---|
| `total_actions_made` | Computed | total semantic actions emitted by the agent |
| `unique_action_names` | Computed | distinct action names called (e.g. `search.datasets` counted once) |
| `action_success_rate` | Computed | actions that returned a non-error result / total |
| `action_mapped_fraction` | Computed | fraction of literal calls the mapper could classify (the rest go through the LLM judge) |
| `required_action_types_minimal` / `_optimal` | Computed | distinct required action names in the task's ground truth |
| `required_actions_minimal` / `_optimal` | Computed | total required action calls in the task's ground truth |
| `matched_action_types_minimal` / `_optimal` | Computed | distinct ground-truth action names the agent did call |
| `matched_actions_minimal` / `_optimal` | LLM-judge | required action calls the agent fulfilled (with judge-checked parameters) |

**Rates and scores** (precision/recall/F1 — classic IR formulas):

| Score name | Description |
|---|---|
| `precision_action_type_minimal` / `_optimal` | over action **names** — "did the agent pick the right tools?" |
| `recall_action_type_minimal` / `_optimal` | over action **names** |
| `f1_action_type_minimal` / `_optimal` | F1 over action names |
| `precision_action_minimal` / `_optimal` | over action **instances** (name + args) — "did the agent invoke them correctly?" |
| `recall_action_minimal` / `_optimal` | over action instances |
| `f1_action_minimal` / `_optimal` | F1 over action instances |

**Trajectory:**

| Score name | Type | Description |
|---|---|---|
| `trajectory_adherence_minimal` / `_optimal` | LLM-judge | how well the agent's ordered sequence of mapped actions matches the expected trajectory |

**Capability distribution counters** (computed from the literal-call source channel — useful at experiment level for "which capability did the agent actually use?"):

`calls_mcp_tool`, `calls_datagouv_cli`, `calls_datagouv_api_http`, `calls_python_local_analysis`, `calls_web_search`, `calls_web_page_fetch`.

---

## Experiment-level aggregates

Computed by `experiment_metrics.compute_experiment_metrics` and emitted as Opik experiment-level scores:

- **Totals** (`_TOTAL_METRICS`) — summed across all tasks: `total_tool_calls`, `total_actions_made`, all the `required_*` / `matched_*` counts, the `calls_*` capability counters, `total_validated_criteria_*`, `total_criteria_*`, `token_usage`.
- **Averages** (`_AVG_METRICS`) — mean across tasks: `result_accuracy_*`, `action_success_rate`, `action_mapped_fraction`, all the `precision/recall/f1_action*` rates, `trajectory_adherence_*`, `latency_ms`, `token_efficiency_*`, `time_efficiency_*`.
- **Failure modes** — summed (each is 0/1 per task), so the aggregate value = number of tasks exhibiting that failure mode.

---

## Sources of inspiration

The metric design draws from existing eval frameworks and recent papers. Mapping is informal — we re-implemented and adapted, not copied.

| Framework / paper | Concept borrowed from | Where it shows up here |
|---|---|---|
| **DeepEval** — Task Completion | LLM-judged alignment between task and outcome | `result_accuracy_*` — grounded on per-task `evaluation_criteria` instead of free-form task extraction |
| **DeepEval** — Argument Correctness | LLM-judged parameter correctness per tool call | parameter judging inside `f1_action_*` (instance-level matching uses arg judging) |
| **DeepEval** — Tool Correctness | Deterministic "did the expected tool sequence happen?" | `matched_action_types_*` / `matched_actions_*` |
| **DeepEval** — Step Efficiency | "Was every action strictly required?" | `precision_action_*` — wrong/redundant calls drop precision |
| **DeepEval** — Plan Adherence | Sequence alignment with the expected plan | `trajectory_adherence_*` |
| **RAGAS** — Tool Call Accuracy (sequence × args) | All-or-nothing alignment scoring | informed the choice to also expose the **soft** F1 versions (see below) |
| **RAGAS** — Tool Call F1 | Unordered precision/recall/F1 on tool calls — softer than strict sequence match | `precision_action_*` / `recall_action_*` / `f1_action_*` (and `_type` siblings) |
| **Arize Phoenix** — Tool Selection | "Did the LLM pick the right tool?" — orthogonal to argument correctness | `f1_action_type_*` (over names only — the *what*) |
| **Arize Phoenix** — Tool Invocation | "Was the call well-formed with correct args?" | `f1_action_*` (over name + args — the *how*) |
| **Arize Phoenix** — Tool Response Handling | Did the agent use the result correctly? | implicit in `result_accuracy_*` (the criteria capture whether the answer used the results correctly) |
| **arxiv 2507.12806** | Tool-call evaluation framework | `precision/recall/f1_action_*`, `trajectory_adherence_*` |
| **arxiv 2510.04550** | Tool-use benchmark methodology | `recall_action_type_*`, `trajectory_adherence_*` |
| **arxiv 2512.24565** | Cost-vs-quality composite scoring | `token_efficiency_*`, `time_efficiency_*` |
| **arxiv 2508.20453** | Tool-call schema/structural validation | informed the original "schema compliance" idea — now subsumed by `action_success_rate` + parameter judging |
| Custom — domain-specific | Open-data agent failure taxonomy | failure modes (`HALLUCINATION`, `MISSING_CAVEAT`, `WRONG_RESOURCE`, …) |
| Custom — capability profiling | "Which tool surface did the agent actually use?" | `calls_*` counters |

**On terminology — "accuracy" vs precision/F1.** RAGAS's `ToolCallAccuracy` label is misleading in the classical-ML sense: if "no tool call" is also a valid prediction, computing accuracy requires labelled negatives. Example: if an agent makes 100 decisions (90 correct "no call", 10 tool calls of which 8 are right), accuracy = 98% looks great while precision = 80% reveals that 1 in 5 actual tool picks is wrong. We expose **precision / recall / F1** instead (both at action-type and action-instance granularity), which is unambiguous and matches DeepEval's Tool Correctness framing.
