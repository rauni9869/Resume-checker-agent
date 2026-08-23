from __future__ import annotations

from pathlib import Path

from resume_checker.config import Settings, get_settings
from resume_checker.extraction.pdf import extract_from_pdf, extract_from_text
from resume_checker.graph.state import GraphState
from resume_checker.guardrails import (
    apply_groundedness,
    clamp_critique_scores,
    output_schema_findings,
    redact_pii,
    scan_injection,
    validate_extracted_text,
    validate_upload,
)
from resume_checker.llm import build_generator
from resume_checker.reporting import render_html_report
from resume_checker.schemas import AnalysisResult, GuardrailFinding, Severity
from resume_checker.scoring.ats import score_ats
from resume_checker.scoring.matcher import semantic_match
from resume_checker.scoring.semantic import score_semantic_panel


def _settings() -> Settings:
    return get_settings()


def _append(state: GraphState, finding: GuardrailFinding) -> list[GuardrailFinding]:
    return list(state.get("guardrails") or []) + [finding]


def validate_inputs_node(state: GraphState) -> GraphState:
    settings = _settings()
    findings: list[GuardrailFinding] = []
    if state.get("resume_path"):
        findings.extend(validate_upload(Path(state["resume_path"]), settings))
    findings.extend(scan_injection(state.get("job_description") or ""))
    if state.get("resume_text"):
        findings.extend(scan_injection(state["resume_text"]))
    blocked = any(f.severity == Severity.BLOCKING for f in findings)
    return {"guardrails": findings, "blocked": blocked}


def extract_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return {}
    settings = _settings()
    if state.get("resume_path"):
        extraction = extract_from_pdf(state["resume_path"], settings)
    else:
        extraction = extract_from_text(state.get("resume_text") or "")
    if settings.redact_pii:
        extraction.text = redact_pii(extraction.text)
    findings = list(state.get("guardrails") or []) + validate_extracted_text(extraction.text, settings)
    blocked = any(f.severity == Severity.BLOCKING for f in findings)
    return {"extraction": extraction, "guardrails": findings, "blocked": blocked}


def score_node(state: GraphState) -> GraphState:
    if state.get("blocked") or not state.get("extraction"):
        return {}
    ats = score_ats(state["extraction"].text, state["job_description"])
    match = semantic_match(state["extraction"].text, state["job_description"])
    # Chips, critique, and CLI follow retrieval phrases — not leftover keyword tokens.
    ats = ats.model_copy(
        update={
            "matched_skills": match.matched_requirements,
            "missing_skills": match.gap_requirements,
            "extra_skills": [],
        }
    )
    panel = score_semantic_panel(
        state["extraction"].text,
        state["job_description"],
        include_specialists=bool(state.get("include_specialists")),
    )
    return {
        "ats": ats,
        "semantic_match": match,
        "semantic_panel": panel,
        "composite_score": match.composite,
    }


def critique_node(state: GraphState) -> GraphState:
    if state.get("blocked") or not state.get("extraction") or not state.get("ats"):
        return {}
    settings = _settings()
    generator = build_generator(settings)
    critique = generator.critique(
        state["extraction"],
        state["job_description"],
        state["ats"],
        semantic_score=state.get("composite_score"),
        semantic=state.get("semantic_match"),
    )
    critique, groundedness = apply_groundedness(critique, state["extraction"].text)
    critique = clamp_critique_scores(critique, state.get("composite_score") or critique.overall_score)
    findings = list(state.get("guardrails") or []) + output_schema_findings(
        critique, groundedness, settings.groundedness_min_ratio
    )
    return {
        "critique": critique,
        "groundedness": groundedness,
        "guardrails": findings,
        "llm_backend": generator.backend,
    }


def report_node(state: GraphState) -> GraphState:
    result = AnalysisResult(
        ok=not bool(state.get("blocked")),
        candidate_id=state.get("candidate_id") or "anonymous",
        extraction=state.get("extraction"),
        ats=state.get("ats"),
        semantic_match=state.get("semantic_match"),
        semantic_panel=list(state.get("semantic_panel") or []),
        critique=state.get("critique"),
        composite_score=state.get("composite_score"),
        guardrails=list(state.get("guardrails") or []),
        blocked=bool(state.get("blocked")),
        email_status=state.get("email_status"),
        llm_backend=state.get("llm_backend") or "template",
    )
    html = render_html_report(result)
    result.html_report = html
    email_status = None
    if state.get("send_email") and state.get("recipient_email") and result.ok:
        from resume_checker.notifications import send_html_email

        email_status = send_html_email(html, state["recipient_email"])
        result.email_status = email_status
    return {"html_report": html, "email_status": email_status, "result": result.model_dump()}
