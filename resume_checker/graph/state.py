from __future__ import annotations

from typing import Any, TypedDict

from resume_checker.schemas import (
    ATSScore,
    ExtractionResult,
    GuardrailFinding,
    SemanticScore,
    StructuredCritique,
)


class GraphState(TypedDict, total=False):
    candidate_id: str
    job_description: str
    resume_path: str | None
    resume_text: str | None
    recipient_email: str | None
    send_email: bool
    include_specialists: bool
    extraction: ExtractionResult | None
    ats: ATSScore | None
    semantic_panel: list[SemanticScore]
    critique: StructuredCritique | None
    composite_score: float | None
    groundedness: float
    guardrails: list[GuardrailFinding]
    blocked: bool
    email_status: str | None
    llm_backend: str
    html_report: str | None
    result: dict[str, Any]
