"""
Offline tests for the action metrics refactor (no LLM judge required).

Covers:
- map_action_calls deterministic pass: cross-capability mapping into semantic space
  (MCP tool + datagouv CLI), error attribution, counts.
- compute_action_usage_basics / rates: the one-call → many-actions case (e.g. several
  searches inside one execute_python), verified via hand-built ActionInstances.
"""

import asyncio
from pathlib import Path

from agent_eval.evaluators.core.action_mapper import (
    CATEGORY_DATAGOUV_CLI,
    CATEGORY_MCP_TOOL,
    ActionInstance,
    map_action_calls,
)
from agent_eval.evaluators.core.action_usage import (
    compute_action_usage_basics,
    compute_action_usage_rates,
)
from agent_eval.tasks.loader import RequiredAction, RequiredActionArg

_SEMANTIC_LAYER_DIR = (
    Path(__file__).parents[1] / "agent_eval" / "semantic_layer" / "config"
)


def _inst(action: str, *, errored: bool = False, src: str = "c1") -> ActionInstance:
    return ActionInstance(
        action=action,
        args={"query": "x"},
        source_tool_call_id=src,
        source_tool_name="execute_python",
        capability_category="datagouv_api_http",
        errored=errored,
    )


def test_deterministic_cross_capability_mapping():
    """MCP search_datasets + datagouv CLI both land as semantic actions; errors attributed."""
    calls = [
        {
            "tool_call_id": "c1",
            "name": "search_datasets",
            "arguments": {"query": "neurologie"},
            "result": "ok",
        },
        {
            "tool_call_id": "c2",
            "name": "execute_cli",
            "arguments": {
                "command": "datagouv dataset display 536c47d9a3a72933d8d1b3b2"
            },
            "result": "Error: not found",
        },
    ]
    out = asyncio.run(
        map_action_calls(
            actual_tool_calls=calls,
            available_tool_names=["search_datasets"],
            mcp_version="0.2.24",
            semantic_layer_dir=_SEMANTIC_LAYER_DIR,
            judge_model=None,  # deterministic pass only
        )
    )

    by_id = {cm.tool_call_id: cm for cm in out.call_mappings}
    # MCP call → confirmed semantic action, args preserved
    assert by_id["c1"].capability_category == CATEGORY_MCP_TOOL
    assert [a.action for a in by_id["c1"].actions] == ["search.datasets"]
    assert by_id["c1"].actions[0].args == {"query": "neurologie"}
    assert by_id["c1"].actions[0].errored is False
    # datagouv CLI categorised; its result was an error
    assert by_id["c2"].capability_category == CATEGORY_DATAGOUV_CLI
    assert by_id["c2"].errored is True

    assert out.counts.get("mcp_tool") == 1
    assert out.counts.get("datagouv_cli") == 1


def test_usage_basics_one_call_many_actions():
    """One execute_python performing two searches + one metadata fetch → 3 action instances."""
    instances = [
        _inst("search.datasets"),
        _inst("search.datasets"),
        _inst("get.dataset.info"),
    ]
    required_minimal = [
        RequiredAction(name="search.datasets", args=[RequiredActionArg(name="query")]),
        RequiredAction(
            name="get.dataset.info", args=[RequiredActionArg(name="dataset_id")]
        ),
    ]

    basics = compute_action_usage_basics(instances, required_minimal, required_minimal)

    assert basics.total_actions_made == 3
    assert basics.unique_action_names == 2
    assert basics.required_actions_minimal == 2
    assert basics.required_action_types_minimal == 2
    assert basics.matched_action_types_minimal == 2
    assert basics.action_success_rate == 1.0

    # Suppose the params judge validated both required actions (matched_actions == 2)
    rates = compute_action_usage_rates(
        basics, matched_actions_minimal=2, matched_actions_optimal=2
    )
    assert (
        rates.precision_action_type_minimal == 1.0
    )  # 2 matched types / 2 unique names
    assert rates.recall_action_type_minimal == 1.0  # 2 / 2 required types
    assert rates.precision_action_minimal == round(2 / 3, 6)  # 2 matched / 3 instances
    assert rates.recall_action_minimal == 1.0  # 2 / 2 required actions
    assert rates.f1_action_minimal == round(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0), 6)


def test_action_success_rate_with_error():
    instances = [_inst("search.datasets"), _inst("get.data", errored=True)]
    basics = compute_action_usage_basics(instances, [], [])
    assert basics.total_actions_made == 2
    assert basics.action_success_rate == 0.5
