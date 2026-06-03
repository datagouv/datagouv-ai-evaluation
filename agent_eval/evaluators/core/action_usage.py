"""
Deterministic action-usage metrics (semantic space).

Compares the mapped semantic action instances (from action_mapper) against the
semantic baseline (RequiredAction lists). Two granularities:
  - action-type level: distinct semantic action names (set-based)
  - action-instance level: per action instance, params judged by action_params

Pure Python — no LLM calls, no Opik imports.
"""

from dataclasses import dataclass

from agent_eval.evaluators.core._math import safe_div, f1_score
from agent_eval.evaluators.core.action_mapper import ActionInstance
from agent_eval.tasks.loader import RequiredAction


@dataclass
class ActionUsageBasics:
    # actual action counts (precision denominators)
    total_actions_made: int  # len(mapped instances)
    unique_action_names: int  # |set(mapped action names)|
    action_success_rate: float  # instances whose source call did not error
    # GT requirement sizes (recall denominators)
    required_action_types_minimal: int  # |set(GT names)|, minimal level
    required_action_types_optimal: int
    required_actions_minimal: int  # len(GT list), minimal level
    required_actions_optimal: int
    # matched type counts (TP numerators, set-based)
    matched_action_types_minimal: int  # |set(GT names) ∩ set(actual names)|
    matched_action_types_optimal: int


@dataclass
class ActionUsageRates:
    # action-type level (unique name sets)
    precision_action_type_minimal: float
    precision_action_type_optimal: float
    recall_action_type_minimal: float
    recall_action_type_optimal: float
    f1_action_type_minimal: float
    f1_action_type_optimal: float
    # action-instance level (multiset, name + params)
    precision_action_minimal: float
    precision_action_optimal: float
    recall_action_minimal: float
    recall_action_optimal: float
    f1_action_minimal: float
    f1_action_optimal: float


def compute_action_usage_basics(
    instances: list[ActionInstance],
    required_minimal: list[RequiredAction],
    required_optimal: list[RequiredAction],
) -> ActionUsageBasics:
    """Compute deterministic action-usage counts in semantic space."""
    actual_names = {i.action for i in instances}
    gt_names_minimal = {a.name for a in required_minimal}
    gt_names_optimal = {a.name for a in required_optimal}

    n = len(instances)
    errored = sum(1 for i in instances if i.errored)
    success_rate = round((n - errored) / n, 6) if n else 0.0

    return ActionUsageBasics(
        total_actions_made=n,
        unique_action_names=len(actual_names),
        action_success_rate=success_rate,
        required_action_types_minimal=len(gt_names_minimal),
        required_action_types_optimal=len(gt_names_optimal),
        required_actions_minimal=len(required_minimal),
        required_actions_optimal=len(required_optimal),
        matched_action_types_minimal=len(gt_names_minimal & actual_names),
        matched_action_types_optimal=len(gt_names_optimal & actual_names),
    )


def compute_action_usage_rates(
    basics: ActionUsageBasics,
    matched_actions_minimal: int,
    matched_actions_optimal: int,
) -> ActionUsageRates:
    """
    Derive rate metrics from basics + LLM-judged matched_actions counts.
    All rates are [0, 1] and guard against division by zero (return 0.0).
    """
    p_type_min = safe_div(
        basics.matched_action_types_minimal, basics.unique_action_names
    )
    p_type_opt = safe_div(
        basics.matched_action_types_optimal, basics.unique_action_names
    )
    r_type_min = safe_div(
        basics.matched_action_types_minimal, basics.required_action_types_minimal
    )
    r_type_opt = safe_div(
        basics.matched_action_types_optimal, basics.required_action_types_optimal
    )

    p_act_min = safe_div(matched_actions_minimal, basics.total_actions_made)
    p_act_opt = safe_div(matched_actions_optimal, basics.total_actions_made)
    r_act_min = safe_div(matched_actions_minimal, basics.required_actions_minimal)
    r_act_opt = safe_div(matched_actions_optimal, basics.required_actions_optimal)

    return ActionUsageRates(
        precision_action_type_minimal=p_type_min,
        precision_action_type_optimal=p_type_opt,
        recall_action_type_minimal=r_type_min,
        recall_action_type_optimal=r_type_opt,
        f1_action_type_minimal=f1_score(p_type_min, r_type_min),
        f1_action_type_optimal=f1_score(p_type_opt, r_type_opt),
        precision_action_minimal=p_act_min,
        precision_action_optimal=p_act_opt,
        recall_action_minimal=r_act_min,
        recall_action_optimal=r_act_opt,
        f1_action_minimal=f1_score(p_act_min, r_act_min),
        f1_action_optimal=f1_score(p_act_opt, r_act_opt),
    )
