# Metrics

## Task-level metrics

*For now, all metrics are for single-turn evaluation tasks only.*

To start, we need basic metrics that will serve as basis to compute more elaborate evaluation metrics like rates and scores.

**Metric type:**

- **Computed**: when the metric can be obtained through computation, ground truth annotation, or from the LLM provider (like token count)
- **LLM-as-a-Judge**: when the metric requires an LLM-as-a-Judge to be computed

---

## Basic metrics

### Tool usage

| Metric | Type | Description |
|---|---|---|
| Total tool calls | Computed | Total number of tool calls made by the agent (whether correct or not, errored or not) |
| Min. required tool calls | Computed | Minimum number of tool calls required to complete the task (ground truth) — derived for both minimal and optimal |
| Schema compliant tool calls | Computed | Number of tool calls with correctly structured parameters matching the expected input schema |
| Correct parameters tool calls | LLM-as-a-Judge | Tool calls with correct schema AND parameter values (strict match for IDs, semantic match for queries) — counts each call individually |
| Called tool matching names | Computed | Number of unique ground truth tools used, by exact name match |
| Ground truth tool calls | LLM-as-a-Judge | Number of required tool calls (from ground truth) correctly executed with the right parameters |
| Successful tool calls | Computed | Number of tool calls that did not return an error (4xx, 5xx) |

### Efficiency

| Metric | Type | Description |
|---|---|---|
| Latency | Computed | End-to-end time to complete the task (prompt sent → full response received), in ms |
| Token usage | Computed | Total tokens consumed to complete the task |

### Result accuracy

| Metric | Type | Description |
|---|---|---|
| Total validated criteria | LLM-as-a-Judge | Number of evaluation criteria validated — derived for both minimal and optimal levels |
| Total criteria | Computed | Total number of existing criteria for the task — derived for both minimal and optimal levels |

### Failure modes

Binary flags (0/1) per failure mode, detected by an LLM judge. Not a score in itself but enables aggregate failure analysis at experiment level.

The judge receives both minimal and optimal tool chains but implicitly reasons against the **optimal** scenario. Scoping is currently blurry — a future improvement should make this explicit in the prompt.

Defined failure modes: `HALLUCINATION`, `MISINTERPRETATION`, `WRONG_RESOURCE`, `PARAMETER_ERROR`, `TOOL_OMISSION`, `EARLY_STOP`, `REDUNDANT_LOOP`, `MISSING_CAVEAT`, `NO_FALLBACK`.

---

## Rates & Scores

*Unless otherwise noted, all scores range from [0, 1] — higher is always better.*

| Metric | Formula | Notes |
|---|---|---|
| Schema compliance rate | Compliant tool calls / Total tool calls | Proportion of structurally valid calls. *(arxiv 2508.20453)* |
| Correct parameters rate | Correct parameters tool calls / Total tool calls | *(arxiv 2507.12806)* |
| Tool call success rate | Successful tool calls / Total tool calls | |
| Recall tool usage (minimal / optimal) | Ground truth tool calls / Min. required tool calls | Proportion of required tools correctly invoked. *(arxiv 2510.04550, 2512.24565)* |
| Tool call efficiency | Ground truth tool calls / Total tool calls | Lower = more redundant calls |
| Result accuracy (minimal / optimal) | Total validated criteria / Total criteria | Core quality score |
| Trajectory adherence | LLM-as-a-Judge | Sequence alignment between predicted and expected tool call trajectory. *(arxiv 2507.12806, 2510.04550)* |
| Token efficiency (minimal / optimal) | Result accuracy / Token usage | Score per token consumed. *(arxiv 2512.24565)* |
| Time efficiency (minimal / optimal) | Result accuracy / Latency | Score per ms of execution. *(arxiv 2512.24565)* |

---

## Experiment-level aggregates

All task-level metrics are averaged across the dataset at experiment level to give global scores per (model, MCP version, system prompt, capabilities) combination.
