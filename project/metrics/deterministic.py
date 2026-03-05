from dataclasses import dataclass
from project.agent.agent import AgentResult


@dataclass
class DeterministicScores:
    answer_present: bool
    answer_length: int


def score_deterministic(result: AgentResult) -> DeterministicScores:
    answer = result.answer or ""

    return DeterministicScores(
        answer_present=len(answer.strip()) > 0,
        answer_length=len(answer),
    )
