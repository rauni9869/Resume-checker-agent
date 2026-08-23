"""Semantic scoring as retrieval: JD requirement phrases vs resume bullets.

Fragments and vectors are taken from the two documents. There is no skill
dictionary. Short JD phrases are encoded as themselves (not glued to the
whole qualifications sentence, which made every query look the same).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from resume_checker.config import Settings, get_settings
from resume_checker.schemas import RequirementAlignment, SemanticMatch
from resume_checker.scoring.ats import job_focus_text

logger = logging.getLogger(__name__)

_MAX_PASSAGES = 40
_STOP = {word.lower() for word in ENGLISH_STOP_WORDS}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def resume_passages(text: str) -> list[str]:
    parts = re.split(r"\n+|•|(?<=[.!?])\s+", text or "")
    passages = [_squash(part) for part in parts]
    passages = [part for part in passages if len(part) >= 20]
    if not passages:
        blob = _squash(text)
        return [blob[:1500]] if blob else ["n/a"]
    return passages[:_MAX_PASSAGES]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\n])\s+", text or "")
    return [_squash(part) for part in parts if _squash(part)]


def _clean_phrase(phrase: str) -> str | None:
    phrase = re.sub(r"\([^)]*\)", " ", phrase)
    phrase = _squash(phrase).strip(" -:;.")
    tokens = [_squash(token) for token in phrase.split()]
    tokens = [token for token in tokens if token]
    while tokens and re.sub(r"[^a-z0-9]+", "", tokens[0].lower()) in _STOP:
        tokens.pop(0)
    while tokens and re.sub(r"[^a-z0-9]+", "", tokens[-1].lower()) in _STOP:
        tokens.pop()
    if not tokens:
        return None
    content = [re.sub(r"[^a-z0-9+#.+]+", "", token.lower()) for token in tokens]
    content = [token for token in content if token and token not in _STOP]
    if not content:
        return None
    if len(content) == 1 and len(content[0]) < 2:
        return None
    return " ".join(tokens)


def requirement_queries(job_description: str) -> list[str]:
    """Requirement phrases copied from the JD only."""
    focus = job_focus_text(job_description)
    phrases: list[str] = []
    seen: set[str] = set()
    for sentence in _sentences(focus):
        fragments = [_squash(part) for part in re.split(r"[,;/|•]|(?:\s+and\s+)", sentence)]
        for fragment in fragments:
            phrase = _clean_phrase(fragment)
            if not phrase:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)
    if not phrases:
        blob = _squash(focus or job_description)
        if blob:
            phrases.append(blob[:160])
    return phrases[:_MAX_PASSAGES]


class TfidfEncoder:
    name = "tfidf-context-vectors"

    def encode(self, texts: list[str]) -> np.ndarray:
        cleaned = [re.sub(r"\s+", " ", (text or "").lower().replace("-", " ")) for text in texts]
        word = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)(?:c\+\+\d*|cpp|ci/cd|[a-zA-Z][a-zA-Z0-9+#]{1,})",
        )
        chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        return normalize(hstack([word.fit_transform(cleaned), chars.fit_transform(cleaned)])).toarray()


class TransformerEncoder:
    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_id
        self._model = SentenceTransformer(model_id)
        lowered = model_id.lower()
        self._query_prefix = "query: " if "e5" in lowered else ""
        self._passage_prefix = "passage: " if "e5" in lowered else ""

    def encode(self, texts: list[str], *, as_query: bool = False) -> np.ndarray:
        prefix = self._query_prefix if as_query else self._passage_prefix
        payload = [prefix + (text or "") for text in texts]
        vectors = self._model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=float)


@lru_cache
def get_encoder(backend: str = "auto", model_id: str = "intfloat/e5-small-v2"):
    if backend in {"tfidf", "hash"}:
        return TfidfEncoder()
    if backend in {"auto", "minilm", "sentence-transformers", "e5"}:
        try:
            return TransformerEncoder(model_id)
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
    passages = resume_passages(resume_text)
    queries = requirement_queries(job_description)

    if isinstance(encoder, TransformerEncoder):
        resume_doc = encoder.encode([(resume_text or " ")[:7000]], as_query=False)[0]
        job_doc = encoder.encode([(job_focus or job_description or " ")[:7000]], as_query=True)[0]
        passage_mat = encoder.encode(passages, as_query=False)
        query_mat = encoder.encode(queries, as_query=True)
    else:
        corpus = [
            (resume_text or " ")[:7000],
            (job_focus or job_description or " ")[:7000],
            *passages,
            *queries,
        ]
        matrix = encoder.encode(corpus)
        resume_doc, job_doc = matrix[0], matrix[1]
        passage_mat = matrix[2 : 2 + len(passages)]
        query_mat = matrix[2 + len(passages) :]

    document_cosine = float(resume_doc @ job_doc)
    alignments: list[RequirementAlignment] = []
    if len(query_mat) and len(passage_mat):
        sims = query_mat @ passage_mat.T
        best_idx = sims.argmax(axis=1)
        best_vals = sims.max(axis=1)
        coverage = float(best_vals.mean())
        for label, idx, value in zip(queries, best_idx, best_vals, strict=True):
            alignments.append(
                RequirementAlignment(
                    requirement=label[:240],
                    resume_span=passages[int(idx)][:240],
                    score=_to_percent(float(value)),
                )
            )
        alignments.sort(key=lambda item: item.score, reverse=True)
    else:
        coverage = document_cosine

    scores = [item.score for item in alignments]
    cut = float(np.median(scores)) if scores else 0.0
    matched = [item.requirement for item in alignments if item.score >= cut]
    gaps = [item.requirement for item in alignments if item.score < cut]

    document_score = _to_percent(document_cosine)
    requirement_coverage = _to_percent(coverage)
    composite = round(0.25 * document_score + 0.75 * requirement_coverage, 2)
    return SemanticMatch(
        backend=encoder.name,
        document_score=document_score,
        requirement_coverage=requirement_coverage,
        composite=composite,
        alignments=_diversify_alignments(alignments, 12),
        matched_requirements=matched[:12],
        gap_requirements=gaps[:12],
    )


def _diversify_alignments(alignments: list[RequirementAlignment], max_n: int) -> list[RequirementAlignment]:
    """Prefer unique resume spans so the UI does not repeat one bullet."""
    seen: set[str] = set()
    unique: list[RequirementAlignment] = []
    rest: list[RequirementAlignment] = []
    for row in alignments:
        key = row.resume_span[:100]
        if key not in seen:
            seen.add(key)
            unique.append(row)
        else:
            rest.append(row)
    return (unique + rest)[:max_n]
