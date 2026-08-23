"""Specialist open-source judges used only at evaluation time.

1. shankerram3/resumator — sentence-transformer fine-tuned on resume–JD pairs
2. cross-encoder/ms-marco-MiniLM-L-6-v2 — passage-relevance cross encoder
3. TechWolf/JobBERT-v3 — job-title / skill embedding model
"""

from __future__ import annotations

from resume_checker.config import get_settings
from resume_checker.schemas import JudgeScore
from resume_checker.scoring.semantic import score_with_model

SPECIALISTS = (
    ("shankerram3/resumator", "resume_matcher"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross_encoder"),
    ("TechWolf/JobBERT-v3", "job_title"),
)


def run_specialist_judges(resume_text: str, job_description: str) -> list[JudgeScore]:
    settings = get_settings()
    panel = [
        (settings.resume_matcher_model, "resume_matcher"),
        (settings.cross_encoder_model, "cross_encoder"),
        (settings.job_title_model, "job_title"),
    ]
    scores: list[JudgeScore] = []
    for model_id, kind in panel:
        result = score_with_model(model_id, kind, resume_text, job_description)
        scores.append(
            JudgeScore(
                model_id=model_id,
                task=kind,
                score=result.score,
                available=not bool(result.notes),
                error=result.notes or None,
            )
        )
    return scores
