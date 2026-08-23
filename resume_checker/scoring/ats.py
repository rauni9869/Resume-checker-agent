"""ATS scoring by extracting terms from the job and resume — no skill ontology."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_checker.schemas import ATSScore, SkillMatch

_SECTION_CUT = (
    "qualifications",
    "requirements",
    "responsibilities",
    "required skills",
    "what you'll do",
    "must have",
    "minimum qualifications",
)
_SKILL_HEADERS = re.compile(
    r"(?im)^(skills|technical skills|languages|utilities|tools|tech stack|software)[:\s]+(.+)$"
)
_FLUFF = frozenset(ENGLISH_STOP_WORDS) | {
    "about",
    "us",
    "role",
    "job",
    "position",
    "team",
    "company",
    "employees",
    "world",
    "join",
    "work",
    "working",
    "opportunity",
    "looking",
    "seeking",
    "including",
    "using",
    "used",
    "strong",
    "plus",
    "huge",
    "added",
    "advantage",
    "experience",
    "years",
    "year",
    "equivalent",
}


def _norm(text: str) -> str:
    text = (text or "").lower().replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("c/c++", "c++ c")
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def job_focus_text(job_description: str) -> str:
    blob = job_description or ""
    lower = blob.lower()
    hits = []
    for marker in _SECTION_CUT:
        idx = lower.find(marker)
        if idx != -1:
            hits.append(idx)
    return blob[min(hits) :] if hits else blob


def _split_list(raw: str) -> list[str]:
    parts = re.split(r"[,;|/•]", raw)
    out: list[str] = []
    for part in parts:
        item = re.sub(r"\s+", " ", part).strip(" .:-")
        if item.count(" ") >= 3:
            item = " ".join(item.split()[-2:])
        if 1 < len(item) <= 48 and "/" not in item:
            out.append(item.lower())
    return out


def _header_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in _SKILL_HEADERS.finditer(text):
        terms.extend(_split_list(match.group(2)))
    return terms


def _slash_clusters(text: str) -> list[str]:
    terms: list[str] = []
    for cluster in re.findall(r"\b[A-Za-z][A-Za-z0-9+#]*(?:/[A-Za-z][A-Za-z0-9+#.+]*){1,6}\b", text):
        terms.extend(_split_list(cluster.replace("/", ",")))
    return terms


def _comma_line_terms(text: str) -> list[str]:
    terms: list[str] = []
    for line in text.splitlines():
        if "," not in line:
            continue
        if not re.search(r"(?i)skill|qualif|requir|tech|language|stack|must", line):
            if line.count(",") < 2:
                continue
        terms.extend(_split_list(line.split(":", 1)[-1]))
    return terms


def _capitalized_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-zA-Z0-9+#]{1,}(?:[./+][A-Za-z0-9+#]+)*)\b", text):
        token = match.group(1)
        start = match.start()
        before = text[:start]
        sentence_start = (start == 0) or bool(re.search(r"[\n.!?][ \t]*$", before))
        is_acronym = token.isupper() and 2 <= len(token) <= 8
        in_list = bool(re.search(r"[,/:(]\s*$", before))
        if token.lower() in _FLUFF:
            continue
        if sentence_start and not is_acronym and not in_list:
            continue
        if token in {"The", "Our", "We", "This", "Strong", "Come", "Join", "About", "Founded"}:
            continue
        terms.append(token.lower())
    return terms


def _tfidf_terms(text: str, limit: int = 16, ngram_range: tuple[int, int] = (2, 3)) -> list[str]:
    blob = _norm(text)
    if len(blob) < 20:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=1,
        token_pattern=r"(?u)(?:c\+\+\d*|ci/cd|[a-zA-Z][a-zA-Z0-9+#]{1,})",
    )
    try:
        matrix = vectorizer.fit_transform([blob])
    except ValueError:
        return []
    vocab = vectorizer.get_feature_names_out()
    weights = matrix.toarray()[0]
    ranked = sorted(zip(weights, vocab, strict=True), reverse=True)
    terms: list[str] = []
    for weight, term in ranked:
        if weight <= 0 or term in _FLUFF:
            continue
        parts = term.split()
        if any(part in _FLUFF for part in parts):
            continue
        if term.isdigit() or len(term) < 3:
            continue
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def extract_terms(text: str, *, from_job: bool = False, limit: int = 28) -> list[str]:
    source = job_focus_text(text) if from_job else text
    ordered: list[str] = []
    seen: set[str] = set()

    def add(items: list[str], allow_unigram: bool = True) -> None:
        for item in items:
            key = _norm(item)
            if not key or key in seen or key in _FLUFF or "/" in key:
                continue
            if key.isdigit() or len(key) < 2:
                continue
            words = key.split()
            if len(words) > 3:
                key = " ".join(words[-2:])
                words = key.split()
            if len(words) == 1 and not allow_unigram and not re.search(r"[+#]|c\+\+", key):
                continue
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)

    add(_header_terms(source), allow_unigram=True)
    add(_slash_clusters(source), allow_unigram=True)
    add(_comma_line_terms(source), allow_unigram=True)
    add(_capitalized_terms(source), allow_unigram=True)
    if not from_job:
        add(_tfidf_terms(source, limit=12, ngram_range=(1, 2)), allow_unigram=True)
    cap = 12 if from_job else limit
    return ordered[:cap]


def _variants(term: str) -> set[str]:
    base = _norm(term)
    out = {base, base.replace(" ", ""), term.lower().strip()}
    if base.endswith("s") and len(base) > 4:
        out.add(base[:-1])
    words = [w for w in base.split() if w]
    if len(words) >= 2 and all(len(w) >= 5 for w in words):
        out.add("".join(w[0] for w in words))
    if re.fullmatch(r"c\+\+\d+", base):
        out.add("c++")
    return {item for item in out if item}


def term_evidenced(term: str, document: str) -> bool:
    hay = _norm(document)
    for variant in _variants(term):
        if len(variant) <= 2 and variant.isalpha() and variant not in {"c", "r", "go", "os", "ml", "ai"}:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", hay):
            return True
        if " " in variant and all(
            re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", hay)
            for word in variant.split()
            if len(word) > 2
        ):
            return True
    if len(term) >= 8:
        for window in re.findall(r"[a-z0-9+# ]{4,40}", hay):
            if SequenceMatcher(None, _norm(term), window).ratio() >= 0.88:
                return True
    return False


def extract_skills(text: str) -> set[str]:
    return set(extract_terms(text, from_job=False))


def tfidf_similarity(resume_text: str, job_description: str) -> float:
    docs = [_norm(resume_text) or " ", _norm(job_focus_text(job_description)) or " "]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)(?:c\+\+\d*|ci/cd|[a-zA-Z][a-zA-Z0-9+#]{1,})",
    )
    try:
        matrix = vectorizer.fit_transform(docs)
        score = float(cosine_similarity(matrix[0], matrix[1])[0][0] * 100)
    except ValueError:
        score = 0.0
    return round(max(0.0, min(score, 100.0)), 2)


def score_ats(resume_text: str, job_description: str) -> ATSScore:
    job_terms = extract_terms(job_description, from_job=True)
    resume_terms = extract_terms(resume_text, from_job=False)
    matched = [term for term in job_terms if term_evidenced(term, resume_text)]
    missing = [term for term in job_terms if term not in matched]
    extra = [term for term in resume_terms if not term_evidenced(term, job_description)]
    coverage = (len(matched) / len(job_terms) * 100) if job_terms else 100.0
    matrix = [
        SkillMatch(skill=term, in_resume=term_evidenced(term, resume_text), in_job=True)
        for term in job_terms
    ]
    return ATSScore(
        keyword_similarity=tfidf_similarity(resume_text, job_description),
        required_skill_coverage=round(coverage, 2),
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra[:16],
        skill_matrix=matrix,
    )
