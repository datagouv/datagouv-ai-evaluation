from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

# Example judge backend: OpenAI (swap provider later if you want)
from openai import AsyncOpenAI

from project.agent.agent import AgentResult


class JudgeScore(BaseModel):
    # Strict, typed schema for the judge response
    score: float = Field(
        ge=0.0, le=1.0, description="Overall success score between 0 and 1."
    )
    rationale: str = Field(
        min_length=1, description="Short explanation justifying the score."
    )


class JudgeFailure(BaseModel):
    # Useful for debugging when judge output is invalid
    raw_output: str
    error: str


async def score_with_llm_judge(
    prompt: str, result: AgentResult
) -> Optional[JudgeScore]:
    """
    Returns a validated JudgeScore if OPENAI_API_KEY is set, otherwise None.
    The judge MUST output JSON matching JudgeScore.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None

    client = AsyncOpenAI()

    judge_instructions = f"""
You are an evaluator grading an MCP-agent output.

User request:
{prompt}

Agent answer:
{result.answer}

Return ONLY a JSON object matching this schema:
{{
  "score": number between 0 and 1,
  "rationale": string
}}

Scoring rubric:
- 1.0: fully answers the request accurately and completely.
- 0.5: partially correct or missing key elements.
- 0.0: incorrect, irrelevant, or failed.
""".strip()

    resp = await client.responses.create(
        model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        input=judge_instructions,
        max_output_tokens=300,
    )

    raw = (resp.output_text or "").strip()

    # Strict Pydantic parsing (v2)
    try:
        return JudgeScore.model_validate_json(raw)
    except ValidationError as ve:
        # Optional: try to salvage JSON if the model wrapped it in text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return JudgeScore.model_validate_json(candidate)
            except ValidationError:
                pass

        # If you prefer: return None instead of raising
        raise RuntimeError(
            f"LLM judge returned invalid JSON.\nError: {ve}\nRaw output: {raw[:2000]}"
        )
