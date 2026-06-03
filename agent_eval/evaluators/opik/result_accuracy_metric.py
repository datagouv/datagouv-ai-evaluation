"""
Opik wrapper for result accuracy evaluation.
Calls core/result_accuracy.py via asyncio.run().
"""

import asyncio
from pathlib import Path
from opik.evaluation.metrics import base_metric, score_result

from agent_eval.evaluators.core.efficiency import compute_efficiency
from agent_eval.evaluators.core.failure_modes import (
    FailureModeOutput,
    judge_failure_modes,
)
from agent_eval.evaluators.core.result_accuracy import (
    CriterionResult,
    compute_result_accuracy,
)
from agent_eval.evaluators.core.prompts.failure_modes import load_failure_modes
from agent_eval.evaluators.core.judge_model import JudgeModel
from agent_eval.tasks.loader import EvaluationCriteria


def _criteria_reason(results: list[CriterionResult]) -> str:
    lines = []
    for r in results:
        marker = "✓" if r.validated else "✗"
        short = r.criterion[:100].strip() + ("…" if len(r.criterion) > 100 else "")
        lines.append(f"[{marker}] {short}: {r.explanation}")
    return "\n".join(lines)


class ResultAccuracyMetric(base_metric.BaseMetric):
    def __init__(self, judge_model_path: Path):
        # track=False: avoid wrapping score() in @opik.track. The per-metric span it
        # creates makes the dataset-compare view duplicate each item once per scorer
        # span ("Avg of N trials"). Scores still attach to the trace as feedback scores.
        super().__init__(name="result_accuracy", track=False)
        self._judge_model = JudgeModel(judge_model_path)

    def score(
        self,
        input: dict,
        output: dict,
        expected_output: dict,
        **kwargs,
    ) -> list[score_result.ScoreResult]:
        out = output or {}
        user_prompt = (input or {}).get("prompt", "")
        agent_answer = out.get("answer", "")
        actual_tool_calls: list[dict] = out.get("actual_tool_calls") or []
        latency_ms: float = float(out.get("latency_ms") or 0.0)
        token_usage: int = int(out.get("token_usage") or 0)

        exp = expected_output or {}
        raw_criteria = exp.get("evaluation_criteria", {})
        evaluation_criteria = EvaluationCriteria(
            minimal=raw_criteria.get("minimal") or [],
            optimal=raw_criteria.get("optimal") or [],
        )
        raw_chain = exp.get("action_chain") or {}
        required_actions_minimal: list[dict] = (raw_chain.get("minimal") or {}).get(
            "required_actions"
        ) or []
        required_actions_optimal: list[dict] = (raw_chain.get("optimal") or {}).get(
            "required_actions"
        ) or []

        failure_modes_active = bool(load_failure_modes())

        async def _run_all():
            result = await compute_result_accuracy(
                model=self._judge_model,
                user_prompt=user_prompt,
                agent_answer=agent_answer,
                evaluation_criteria=evaluation_criteria,
            )
            seen: set[str] = set()
            failed: list[CriterionResult] = []
            for r in result.results_minimal + result.results_optimal:
                if not r.validated and r.criterion not in seen:
                    seen.add(r.criterion)
                    failed.append(r)

            fm_output: FailureModeOutput = FailureModeOutput()
            if failure_modes_active:
                fm_output = await judge_failure_modes(
                    model=self._judge_model,
                    user_prompt=user_prompt,
                    agent_answer=agent_answer,
                    failed_criteria=failed,
                    actual_tool_calls=actual_tool_calls,
                    required_actions_minimal=required_actions_minimal,
                    required_actions_optimal=required_actions_optimal,
                )
            return result, fm_output

        result, fm_output = asyncio.run(_run_all())

        eff = compute_efficiency(
            latency_ms=latency_ms,
            token_usage=token_usage,
            result_accuracy_minimal=result.result_accuracy_minimal,
            result_accuracy_optimal=result.result_accuracy_optimal,
        )

        scores = [
            score_result.ScoreResult(
                name="result_accuracy_minimal",
                value=result.result_accuracy_minimal,
                reason=_criteria_reason(result.results_minimal),
            ),
            score_result.ScoreResult(
                name="result_accuracy_optimal",
                value=result.result_accuracy_optimal,
                reason=_criteria_reason(result.results_optimal),
            ),
            score_result.ScoreResult(
                name="total_validated_criteria_minimal",
                value=float(result.total_validated_minimal),
            ),
            score_result.ScoreResult(
                name="total_validated_criteria_optimal",
                value=float(result.total_validated_optimal),
            ),
            score_result.ScoreResult(
                name="total_criteria_minimal",
                value=float(result.total_criteria_minimal),
            ),
            score_result.ScoreResult(
                name="total_criteria_optimal",
                value=float(result.total_criteria_optimal),
            ),
            score_result.ScoreResult(name="latency_ms", value=eff.latency_ms),
            score_result.ScoreResult(name="token_usage", value=float(eff.token_usage)),
            score_result.ScoreResult(
                name="token_efficiency_minimal", value=eff.token_efficiency_minimal
            ),
            score_result.ScoreResult(
                name="token_efficiency_optimal", value=eff.token_efficiency_optimal
            ),
            score_result.ScoreResult(
                name="time_efficiency_minimal", value=eff.time_efficiency_minimal
            ),
            score_result.ScoreResult(
                name="time_efficiency_optimal", value=eff.time_efficiency_optimal
            ),
        ]

        for mode_name, mode_score in fm_output.scores.items():
            scores.append(
                score_result.ScoreResult(
                    name=mode_name,
                    value=float(mode_score),
                    reason=fm_output.explanations.get(mode_name) or None,
                )
            )

        return scores
