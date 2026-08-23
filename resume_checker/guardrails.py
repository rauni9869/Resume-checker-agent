"""Input, injection, PII, groundedness, and output schema guardrails."""

from __future__ import annotations

import re
from pathlib import Path

from resume_checker.config import Settings
from resume_checker.schemas import (
    EvidenceSpan,
    GuardrailFinding,
    Severity,
    StructuredCritique,
)

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"disregard (the )?(job|resume)",
    r"reveal (your )?(hidden|secret)",
    r"<\|im_start\|>",
    r"\[INST\]",
]

PII_PATTERNS = {
    "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

RESUME_HINTS = (
    "experience",
    "education",
    "skills",
    "projects",
    "work",
    "engineer",
    "developer",
    "bachelor",
    "university",
    "responsibilities",
)


def redact_pii(text: str) -> str:
    redacted = text
    for name, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{name.upper()}]", redacted, flags=re.I)
    return redacted


def scan_injection(text: str) -> list[GuardrailFinding]:
    blob = text.lower()
    hits = [p for p in INJECTION_PATTERNS if re.search(p, blob, flags=re.I)]
    if not hits:
        return []
    severity = Severity.BLOCKING if len(hits) >= 2 else Severity.WARNING
    return [
        GuardrailFinding(
            code="prompt_injection",
            severity=severity,
            message="Possible prompt-injection language detected in user-supplied text.",
            details={"patterns": hits},
        )
    ]


def validate_upload(path: Path, settings: Settings) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        findings.append(
            GuardrailFinding(
                code="file_too_large",
                severity=Severity.BLOCKING,
                message=f"Upload exceeds {settings.max_upload_mb} MB.",
                details={"size_mb": round(size_mb, 2)},
            )
        )
    header = path.read_bytes()[:8]
    if path.suffix.lower() != ".pdf" or not header.startswith(b"%PDF"):
        findings.append(
            GuardrailFinding(
                code="invalid_pdf",
                severity=Severity.BLOCKING,
                message="Only valid PDF resumes are accepted.",
            )
        )
    return findings


def validate_extracted_text(text: str, settings: Settings) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    if len(text.strip()) < settings.min_extract_chars:
        findings.append(
            GuardrailFinding(
                code="weak_extraction",
                severity=Severity.BLOCKING,
                message="Extracted text is too short to score reliably.",
                details={"chars": len(text.strip())},
            )
        )
        return findings
    lowered = text.lower()
    if sum(1 for hint in RESUME_HINTS if hint in lowered) < 2:
        findings.append(
            GuardrailFinding(
                code="not_a_resume",
                severity=Severity.WARNING,
                message="Document does not look like a resume; scores may be unreliable.",
            )
        )
    findings.extend(scan_injection(text))
    return findings


def quote_in_source(quote: str, source: str) -> bool:
    q = re.sub(r"\s+", " ", quote.lower()).strip()
    s = re.sub(r"\s+", " ", source.lower())
    if len(q) < 8:
        return False
    if q in s:
        return True
    tokens = [t for t in q.split() if len(t) > 3]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in s)
    return hits / len(tokens) >= 0.7


def apply_groundedness(critique: StructuredCritique, resume_text: str) -> tuple[StructuredCritique, float]:
    spans: list[EvidenceSpan] = []
    for item in list(critique.evidence):
        found = quote_in_source(item.quote, resume_text)
        spans.append(item.model_copy(update={"found_in_source": found}))
    for dim in (critique.grammar, critique.formatting, critique.relevance, critique.impact):
        updated = []
        for item in dim.evidence:
            found = quote_in_source(item.quote, resume_text)
            updated.append(item.model_copy(update={"found_in_source": found}))
        dim.evidence = updated
        spans.extend(updated)
    critique.evidence = spans
    ratio = (sum(1 for s in spans if s.found_in_source) / len(spans)) if spans else 1.0
    return critique, round(ratio, 3)


def clamp_critique_scores(critique: StructuredCritique, composite: float) -> StructuredCritique:
    dims = [critique.grammar, critique.formatting, critique.relevance, critique.impact]
    mean = sum(d.score for d in dims) / 4
    # Keep overall within 12 points of the dimension mean and composite blend.
    target = 0.5 * mean + 0.5 * composite
    critique.overall_score = round(min(100.0, max(0.0, target)), 2)
    return critique


def output_schema_findings(critique: StructuredCritique, groundedness: float, min_ratio: float) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    if groundedness < min_ratio:
        findings.append(
            GuardrailFinding(
                code="ungrounded_claims",
                severity=Severity.WARNING,
                message="Some critique claims were not backed by resume quotes.",
                details={"groundedness": groundedness},
            )
        )
    if not critique.rewrite_suggestions:
        findings.append(
            GuardrailFinding(
                code="empty_recommendations",
                severity=Severity.WARNING,
                message="Critique is missing actionable rewrite suggestions.",
            )
        )
    return findings
