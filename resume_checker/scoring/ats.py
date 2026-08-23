"""Deterministic ATS / skill scoring used as a non-LLM baseline."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_checker.schemas import ATSScore, SkillMatch

SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("python", "pytorch", "pandas", "numpy"),
    "java": ("java",),
    "javascript": ("javascript", "js", "node.js", "nodejs", "typescript"),
    "typescript": ("typescript", "ts"),
    "sql": ("sql", "postgres", "postgresql", "mysql", "snowflake"),
    "aws": ("aws", "amazon web services", "s3", "ec2", "lambda"),
    "gcp": ("gcp", "google cloud"),
    "azure": ("azure",),
    "docker": ("docker", "container"),
    "kubernetes": ("kubernetes", "k8s"),
    "machine learning": ("machine learning", "ml", "scikit-learn", "sklearn"),
    "deep learning": ("deep learning", "neural network", "tensorflow", "pytorch"),
    "nlp": ("nlp", "natural language", "llm", "langgraph", "langchain"),
    "langgraph": ("langgraph",),
    "langchain": ("langchain",),
    "fastapi": ("fastapi",),
    "django": ("django",),
    "react": ("react", "react.js"),
    "git": ("git", "github"),
    "ci/cd": ("ci/cd", "github actions", "jenkins", "gitlab ci"),
    "leadership": ("leadership", "led a team", "managed a team", "mentored"),
    "communication": ("communication", "presented", "stakeholder"),
    "rest apis": ("rest", "api", "fastapi", "flask"),
    "evaluation": ("evaluation", "evals", "llm-as-judge", "ragas", "guardrail"),
    "excel": ("excel", "spreadsheet", "vlookup"),
    "accounting": ("accounting", "accountant", "gaap", "audit", "bookkeep"),
    "sales": ("sales", "quota", "crm", "pipeline"),
    "marketing": ("marketing", "seo", "campaign", "brand"),
    "customer service": ("customer service", "customer support", "call center"),
    "project management": ("project management", "pmp", "agile", "scrum", "jira"),
    "data analysis": ("data analysis", "tableau", "power bi", "analytics"),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _mentions(text: str, aliases: tuple[str, ...]) -> bool:
    blob = _normalize(text)
    for alias in aliases:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", blob):
            return True
    return False


def extract_skills(text: str) -> set[str]:
    return {name for name, aliases in SKILL_ALIASES.items() if _mentions(text, aliases)}


def tfidf_similarity(resume_text: str, job_description: str) -> float:
    docs = [resume_text or " ", job_description or " "]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        matrix = vectorizer.fit_transform(docs)
        score = float(cosine_similarity(matrix[0], matrix[1])[0][0] * 100)
    except ValueError:
        score = 0.0
    return round(max(0.0, min(score, 100.0)), 2)


def score_ats(resume_text: str, job_description: str) -> ATSScore:
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    extra = sorted(resume_skills - job_skills)
    coverage = (len(matched) / len(job_skills) * 100) if job_skills else 100.0
    matrix = [
        SkillMatch(skill=skill, in_resume=skill in resume_skills, in_job=skill in job_skills)
        for skill in sorted(resume_skills | job_skills)
    ]
    return ATSScore(
        keyword_similarity=tfidf_similarity(resume_text, job_description),
        required_skill_coverage=round(coverage, 2),
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        skill_matrix=matrix,
    )
