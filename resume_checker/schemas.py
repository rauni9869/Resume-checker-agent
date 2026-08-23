from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


class ExtractionMethod(str, Enum):
    NATIVE = "native_pdf"
    OCR = "ocr"
    TEXT = "plain_text"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class GuardrailFinding(BaseModel):
    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceSpan(BaseModel):
    claim: str
    quote: str
    found_in_source: bool = False


class SkillMatch(BaseModel):
    skill: str
    in_resume: bool
    in_job: bool


class ATSScore(BaseModel):
    keyword_similarity: float = Field(ge=0, le=100)
    required_skill_coverage: float = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    extra_skills: list[str] = Field(default_factory=list)
    skill_matrix: list[SkillMatch] = Field(default_factory=list)


class SemanticScore(BaseModel):
    model_id: str
    kind: str
    score: float = Field(ge=0, le=100)
    notes: str = ""


class RequirementAlignment(BaseModel):
    requirement: str
    resume_span: str
    score: float = Field(ge=0, le=100)


class SemanticMatch(BaseModel):
    backend: str
    document_score: float = Field(ge=0, le=100)
    requirement_coverage: float = Field(ge=0, le=100)
    composite: float = Field(ge=0, le=100)
    alignments: list[RequirementAlignment] = Field(default_factory=list)


class DimensionScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    rationale: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class StructuredCritique(BaseModel):
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rewrite_suggestions: list[str] = Field(default_factory=list)
    grammar: DimensionScore
    formatting: DimensionScore
    relevance: DimensionScore
    impact: DimensionScore
    overall_score: float = Field(ge=0, le=100)
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @field_validator("strengths", "gaps", "rewrite_suggestions")
    @classmethod
    def _cap_lists(cls, value: list[str]) -> list[str]:
        return value[:8]


class ExtractionResult(BaseModel):
    text: str
    method: ExtractionMethod
    page_count: int = 1
    char_count: int = 0
    quality_score: float = Field(ge=0, le=100, default=0)


class AnalysisRequest(BaseModel):
    job_description: str = Field(min_length=20)
    resume_text: str | None = None
    recipient_email: EmailStr | None = None
    send_email: bool = False
    candidate_id: str = "anonymous"


class AnalysisResult(BaseModel):
    ok: bool
    candidate_id: str
    extraction: ExtractionResult | None = None
    ats: ATSScore | None = None
    semantic_match: SemanticMatch | None = None
    semantic_panel: list[SemanticScore] = Field(default_factory=list)
    critique: StructuredCritique | None = None
    composite_score: float | None = Field(default=None, ge=0, le=100)
    guardrails: list[GuardrailFinding] = Field(default_factory=list)
    blocked: bool = False
    email_status: str | None = None
    llm_backend: str = "template"
    html_report: str | None = None


class EvalCase(BaseModel):
    id: str
    resume_text: str
    job_description: str
    expected_skills_present: list[str] = Field(default_factory=list)
    expected_skills_missing: list[str] = Field(default_factory=list)
    expected_score_min: float = 0
    expected_score_max: float = 100
    should_block: bool = False
    notes: str = ""
    label: str | None = None
    source: str | None = None
    source_index: int | None = None


class JudgeScore(BaseModel):
    model_id: str
    task: str
    score: float = Field(ge=0, le=100)
    available: bool = True
    error: str | None = None


class CaseEvalReport(BaseModel):
    case_id: str
    passed: bool
    skill_precision: float
    skill_recall: float
    skill_f1: float
    score_in_band: bool
    groundedness: float
    schema_valid: bool
    blocked_as_expected: bool
    composite_score: float = 0.0
    predicted_label: str | None = None
    judge_scores: list[JudgeScore] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


class EvalSuiteReport(BaseModel):
    dataset: str
    cases: list[CaseEvalReport]
    n: int
    pass_rate: float
    mean_skill_f1: float
    mean_groundedness: float
    label_accuracy: float | None = None
    pairwise_ranking_accuracy: float | None = None
    spearman_vs_labels: float | None = None
    judge_means: dict[str, float] = Field(default_factory=dict)
    spearman_vs_judges: dict[str, float] = Field(default_factory=dict)
    gate_passed: bool
    thresholds: dict[str, float]
