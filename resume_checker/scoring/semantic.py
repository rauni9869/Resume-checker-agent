"""Open-source embedding and cross-encoder scoring panel."""

from __future__ import annotations

import logging
from functools import lru_cache

from resume_checker.config import Settings, get_settings
from resume_checker.schemas import SemanticScore
from resume_checker.scoring.ats import extract_skills

logger = logging.getLogger(__name__)
_UNAVAILABLE: set[str] = set()

_MAX_CHARS = 4000


def _clip(text: str) -> str:
    return (text or "").strip()[:_MAX_CHARS]


@lru_cache(maxsize=8)
def _sentence_model(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@lru_cache(maxsize=4)
def _cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


def _cosine_percent(model_id: str, resume_text: str, job_description: str) -> float:
    model = _sentence_model(model_id)
    embeddings = model.encode(
        [_clip(resume_text), _clip(job_description)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    score = float((embeddings[0] * embeddings[1]).sum())
    return round(max(0.0, min(score, 1.0) * 100.0), 2)


def _cross_encoder_percent(model_id: str, resume_text: str, job_description: str) -> float:
    model = _cross_encoder(model_id)
    raw = float(model.predict([(_clip(job_description), _clip(resume_text))], show_progress_bar=False)[0])
    # ms-marco MiniLM scores are unbounded logits; squash with a sigmoid-ish map.
    import math

    prob = 1 / (1 + math.exp(-raw))
    return round(prob * 100, 2)


def _jobbert_skill_alignment(model_id: str, resume_text: str, job_description: str) -> float:
    """JobBERT-v3 is trained on short job titles/skills (64 tokens). Compare skill phrases."""
    model = _sentence_model(model_id)
    resume_skills = sorted(extract_skills(resume_text)) or ["general professional"]
    job_skills = sorted(extract_skills(job_description)) or ["general professional"]
    resume_vec = model.encode(resume_skills, normalize_embeddings=True, show_progress_bar=False)
    job_vec = model.encode(job_skills, normalize_embeddings=True, show_progress_bar=False)
    sims = resume_vec @ job_vec.T
    best = float(sims.max()) if sims.size else 0.0
    return round(max(0.0, min(best * 100, 100.0)), 2)


def score_with_model(
    model_id: str,
    kind: str,
    resume_text: str,
    job_description: str,
) -> SemanticScore:
    try:
        if kind == "cross_encoder":
            score = _cross_encoder_percent(model_id, resume_text, job_description)
        elif kind == "job_title":
            score = _jobbert_skill_alignment(model_id, resume_text, job_description)
        else:
            score = _cosine_percent(model_id, resume_text, job_description)
        return SemanticScore(model_id=model_id, kind=kind, score=score)
    except Exception as exc:  # noqa: BLE001 - model download/runtime should never crash the API
        if model_id not in _UNAVAILABLE:
            logger.warning("Semantic model %s unavailable: %s", model_id, exc)
            _UNAVAILABLE.add(model_id)
        return SemanticScore(
            model_id=model_id,
            kind=kind,
            score=0.0,
            notes=f"unavailable: {exc.__class__.__name__}",
        )


def score_semantic_panel(
    resume_text: str,
    job_description: str,
    settings: Settings | None = None,
    include_specialists: bool = False,
) -> list[SemanticScore]:
    settings = settings or get_settings()
    panel = [
        score_with_model(settings.semantic_model, "bi_encoder", resume_text, job_description),
    ]
    if include_specialists or settings.download_eval_models:
        panel.extend(
            [
                score_with_model(
                    settings.resume_matcher_model, "resume_matcher", resume_text, job_description
                ),
                score_with_model(
                    settings.cross_encoder_model, "cross_encoder", resume_text, job_description
                ),
                score_with_model(settings.job_title_model, "job_title", resume_text, job_description),
            ]
        )
    return panel
