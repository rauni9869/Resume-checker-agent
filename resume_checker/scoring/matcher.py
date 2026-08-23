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

_MAX_PASSAGES = 160
_MAX_QUERIES = 40
_STOP = {word.lower() for word in ENGLISH_STOP_WORDS}
_CHIP_MIN = 40.0


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def resume_passages(text: str) -> list[str]:
    """Keep wrapped PDF bullets intact and do not drop later projects."""
    lines = [ln.strip() for ln in (text or "").replace("•", "\n•").splitlines()]
    merged: list[str] = []
    buf = ""
    for line in lines:
        if not line:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        is_bullet = line.startswith("•") or line.startswith("- ")
        payload = line[2:].strip() if line.startswith("- ") else line.lstrip("•").strip()
        if not payload:
            continue
        if is_bullet:
            if buf:
                merged.append(buf)
            buf = payload
            continue
        if buf and re.match(r"[a-z]", payload) and not re.search(r"[.!?]$", buf) and len(buf) < 160:
            buf = f"{buf} {payload}"
        else:
            if buf:
                merged.append(buf)
            buf = payload
    if buf:
        merged.append(buf)
    passages = [_squash(part) for part in merged]
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
    phrase = re.sub(
        r"^(qualifications|requirements|responsibilities|must have)\s*:\s*",
        "",
        phrase,
        flags=re.I,
    )
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
    if not _keep_query(phrase, content):
        return None
    return " ".join(tokens)


def _keep_query(phrase: str, content: list[str]) -> bool:
    if len(content) >= 2:
        return True
    token = content[0]
    if re.search(r"[0-9+#]", token) or len(token) >= 8:
        return True
    if re.search(r"b\.?\s*tech|m\.?\s*tech", phrase, flags=re.I):
        return True
    if re.fullmatch(r"[A-Z]{2,6}", phrase.strip()):
        return True
    return False


def _fold_tokens(text: str) -> set[str]:
    blob = (text or "").lower().replace("c++", "cpp")
    blob = re.sub(r"b\.?\s*tech", "btech", blob)
    blob = re.sub(r"m\.?\s*tech", "mtech", blob)
    blob = re.sub(r"[^a-z0-9+#]+", " ", blob)
    tokens: set[str] = set()
    for token in blob.split():
        if token in _STOP or len(token) < 2:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.add(token)
    return tokens


def _lexically_supported(query: str, passage: str) -> bool:
    query_tokens = _fold_tokens(query)
    passage_tokens = _fold_tokens(passage)
    if not query_tokens:
        return False
    must = {token for token in query_tokens if token in {"cpp", "btech", "mtech"} or any(ch.isdigit() for ch in token)}
    if must:
        return must <= passage_tokens
    if query_tokens <= passage_tokens:
        return True
    core = {token for token in query_tokens if len(token) >= 5}
    return bool(core) and core <= passage_tokens


def _select_passage(sims_row: np.ndarray, passages: list[str], query: str) -> tuple[int, float]:
    order = np.argsort(-sims_row)
    supported = [int(idx) for idx in order[:16] if _lexically_supported(query, passages[int(idx)])]
    if supported:
        idx = max(supported, key=lambda i: float(sims_row[i]))
        return idx, float(sims_row[idx])
    idx = int(order[0])
    return idx, float(sims_row[idx])


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
    return phrases[:_MAX_QUERIES]


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


# Bi-encoders (E5) put most English professional text in a high cosine band.
# Reporting cosine*100 made 0.79 look like a 79% match. Hits below the floor
# contribute 0; 100 means cosine at or above the "full credit" end of the band.
_NEURAL_FLOOR = 0.78
_NEURAL_FULL = 0.92
_SPARSE_FLOOR = 0.08
_SPARSE_FULL = 0.38


def calibrate_cosine(cosine: float, *, neural: bool) -> float:
    lo, hi = (_NEURAL_FLOOR, _NEURAL_FULL) if neural else (_SPARSE_FLOOR, _SPARSE_FULL)
    span = hi - lo
    if span <= 0:
        return 0.0
    mapped = (float(cosine) - lo) / span
    return round(100.0 * max(0.0, min(mapped, 1.0)), 2)


def semantic_match(
    resume_text: str,
    job_description: str,
    settings: Settings | None = None,
) -> SemanticMatch:
    settings = settings or get_settings()
    encoder = get_encoder(settings.embedding_backend, settings.semantic_model)
    neural = isinstance(encoder, TransformerEncoder)
    job_focus = job_focus_text(job_description)
    passages = resume_passages(resume_text)
    queries = requirement_queries(job_description)

    if neural:
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
        coverage_hits: list[float] = []
        for q_i, label in enumerate(queries):
            idx, raw = _select_passage(sims[q_i], passages, label)
            calibrated = calibrate_cosine(raw, neural=neural)
            if _lexically_supported(label, passages[idx]):
                calibrated = max(calibrated, 55.0)
            coverage_hits.append(calibrated)
            alignments.append(
                RequirementAlignment(
                    requirement=label[:240],
                    resume_span=passages[idx][:240],
                    score=calibrated,
                )
            )
        alignments.sort(key=lambda item: item.score, reverse=True)
        coverage = float(np.mean(coverage_hits)) if coverage_hits else 0.0
    else:
        coverage = calibrate_cosine(document_cosine, neural=neural)

    matched = [item.requirement for item in alignments if item.score >= _CHIP_MIN]
    gaps = [item.requirement for item in alignments if item.score < _CHIP_MIN]

    document_score = calibrate_cosine(document_cosine, neural=neural)
    requirement_coverage = round(float(coverage), 2)
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
