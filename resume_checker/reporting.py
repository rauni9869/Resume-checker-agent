from __future__ import annotations

from html import escape

from resume_checker.schemas import AnalysisResult


def render_html_report(result: AnalysisResult) -> str:
    ats = result.ats
    critique = result.critique
    findings = "".join(
        f"<li><strong>{escape(f.code)}</strong> ({escape(f.severity.value)}): {escape(f.message)}</li>"
        for f in result.guardrails
    ) or "<li>No guardrail findings.</li>"
    matched = ", ".join(ats.matched_skills) if ats else "—"
    missing = ", ".join(ats.missing_skills) if ats else "—"
    panel = "".join(
        f"<li>{escape(s.model_id)} ({escape(s.kind)}): {s.score:.1f}"
        + (f" — {escape(s.notes)}" if s.notes else "")
        + "</li>"
        for s in result.semantic_panel
    ) or "<li>Semantic panel not run.</li>"
    suggestions = ""
    if critique:
        suggestions = "".join(f"<li>{escape(item)}</li>" for item in critique.rewrite_suggestions)
    score = result.composite_score if result.composite_score is not None else 0
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Resume analysis</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 760px; margin: 2rem auto; color: #222; }}
    .score {{ font-size: 2.4rem; font-weight: 700; }}
    .muted {{ color: #555; }}
    section {{ margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <h1>Resume analysis report</h1>
  <p class="muted">Candidate: {escape(result.candidate_id)} · backend: {escape(result.llm_backend)}</p>
  <p class="score">{score:.1f}<span class="muted"> / 100</span></p>
  <section>
    <h2>ATS skill coverage</h2>
    <p>Coverage {ats.required_skill_coverage if ats else 0:.1f}% · TF-IDF {ats.keyword_similarity if ats else 0:.1f}%</p>
    <p><strong>Matched:</strong> {escape(matched)}</p>
    <p><strong>Missing:</strong> {escape(missing)}</p>
  </section>
  <section>
    <h2>Open-source model panel</h2>
    <ul>{panel}</ul>
  </section>
  <section>
    <h2>Critique</h2>
    <p>{escape(critique.summary) if critique else "Blocked before critique."}</p>
    <h3>Rewrite suggestions</h3>
    <ul>{suggestions or "<li>None</li>"}</ul>
  </section>
  <section>
    <h2>Guardrails</h2>
    <ul>{findings}</ul>
  </section>
</body>
</html>
"""
