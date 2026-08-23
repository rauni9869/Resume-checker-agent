"""Context-vector matching: embed resume and job, then score by cosine.

Primary score is semantic (document cosine + max-sim of each JD requirement
chunk against resume chunks). Keyword overlap is not used for the score.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from resume_checker.config import Settings, get_settings
from resume_checker.schemas import RequirementAlignment, SemanticMatch
from resume_checker.scoring.ats import job_focus_text

logger = logging.getLogger(__name__)

_MAX_CHUNKS = 28
_MIN_CHUNK = 24


def split_chunks(text: str) -> list[str]:
    raw = re.split(r"\n+|•|(?<=[.!?])\s+", text or "")
    chunks = [re.sub(r"\s+", " ", part).strip() for part in raw]
    chunks = [part for part in chunks if len(part) >= _MIN_CHUNK]
    if not chunks:
        blob = (text or "").strip()
        return [blob[:1200]] if blob else ["n/a"]
    return chunks[:_MAX_CHUNKS]


class TfidfEncoder:
    """Shared vector space over the texts being compared (CI / no-GPU fallback)."""

    name = "tfidf-context-vectors"

    def encode(self, texts: list[str]) -> np.ndarray:
        cleaned = [re.sub(r"\s+", " ", (text or "").lower().replace("-", " ")) for text in texts]
        word = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)(?:c\+\+\d*|cpp|ci/cd|[a-zA-Z][a-zA-Z0-9+#]{1,})",
        )
        chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=1)
        word_mat = word.fit_transform(cleaned)
        char_mat = chars.fit_transform(cleaned)
        return normalize(hstack([word_mat, char_mat])).toarray()


class MiniLMEncoder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_id
        self._model = SentenceTransformer(model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)


@lru_cache
def get_encoder(backend: str = "auto", model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if backend in {"tfidf", "hash"}:
        return TfidfEncoder()
    if backend in {"auto", "minilm", "sentence-transformers"}:
        try:
            return MiniLMEncoder(model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to TF-IDF context vectors: %s", exc)
            return TfidfEncoder()
    return TfidfEncoder()


def _to_percent(cosine: float) -> float:
    return round(max(0.0, min(float(cosine), 1.0) * 100.0), 2)


def semantic_match(
    resume_text: str,
    job_description: str,
    settings: Settings | None = None,
) -> SemanticMatch:
    settings = settings or get_settings()
    encoder = get_encoder(settings.embedding_backend, settings.semantic_model)
    job_focus = job_focus_text(job_description)
    resume_chunks = split_chunks(resume_text)
    job_chunks = split_chunks(job_focus)
    corpus = [
        (resume_text or " ")[:7000],
        (job_focus or job_description or " ")[:7000],
        *resume_chunks,
        *job_chunks,
    ]
    matrix = encoder.encode(corpus)
    document_cosine = float(matrix[0] @ matrix[1])
    resume_mat = matrix[2 : 2 + len(resume_chunks)]
    job_mat = matrix[2 + len(resume_chunks) :]
    alignments: list[RequirementAlignment] = []
    if len(job_mat) and len(resume_mat):
        sims = job_mat @ resume_mat.T
        best_idx = sims.argmax(axis=1)
        best_vals = sims.max(axis=1)
        chunk_cosine = float(best_vals.mean())
        for req, idx, value in zip(job_chunks, best_idx, best_vals, strict=True):
            alignments.append(
                RequirementAlignment(
                    requirement=req[:240],
                    resume_span=resume_chunks[int(idx)][:240],
                    score=_to_percent(float(value)),
                )
            )
        alignments.sort(key=lambda item: item.score, reverse=True)
    else:
        chunk_cosine = document_cosine

    document_score = _to_percent(document_cosine)
    requirement_coverage = _to_percent(chunk_cosine)
    composite = round(0.5 * document_score + 0.5 * requirement_coverage, 2)
    return SemanticMatch(
        backend=encoder.name,
        document_score=document_score,
        requirement_coverage=requirement_coverage,
        composite=composite,
        alignments=alignments[:8],
    )
