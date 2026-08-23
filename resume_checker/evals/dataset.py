from __future__ import annotations

import json
from pathlib import Path

from resume_checker.guardrails import redact_pii
from resume_checker.schemas import EvalCase

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_SAMPLE = Path(__file__).resolve().parents[2] / "evals" / "data" / "hf_resume_job_fit_sample.json"
HF_DATASET = "cnamuangtoun/resume-job-description-fit"
HF_URL = "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit"

LABEL_BAND = {
    "No Fit": (0.0, 45.0),
    "Potential Fit": (35.0, 75.0),
    "Good Fit": (55.0, 100.0),
}


def load_golden_cases() -> list[EvalCase]:
    payload = json.loads((PACKAGE_DIR / "golden.json").read_text())
    return [EvalCase.model_validate(row) for row in payload]


def load_hf_sample(path: Path | None = None) -> list[EvalCase]:
    sample_path = path or REPO_SAMPLE
    payload = json.loads(sample_path.read_text())
    return [EvalCase.model_validate(row) for row in payload["cases"]]


def download_hf_fit_dataset(
    split: str = "test",
    limit: int | None = 100,
    per_label: int | None = None,
) -> list[EvalCase]:
    """Download resume/JD pairs from Hugging Face (no Kaggle account required)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=split)
    counts: dict[str, int] = {label: 0 for label in LABEL_BAND}
    cases: list[EvalCase] = []
    for index, row in enumerate(ds):
        label = row.get("label") or row.get("original_label")
        resume = redact_pii((row.get("resume_text") or row.get("resume") or "")[:4000])
        job = redact_pii(
            (row.get("job_description_text") or row.get("job_description") or row.get("jd") or "")[:4000]
        )
        if label not in LABEL_BAND or len(resume) < 300 or len(job) < 120:
            continue
        if per_label is not None and counts[label] >= per_label:
            if all(counts[k] >= per_label for k in LABEL_BAND):
                break
            continue
        lo, hi = LABEL_BAND[label]
        counts[label] = counts.get(label, 0) + 1
        cases.append(
            EvalCase(
                id=f"hf-{split}-{index}",
                resume_text=resume,
                job_description=job,
                expected_score_min=lo,
                expected_score_max=hi,
                label=label,
                source=HF_DATASET,
                source_index=index,
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def load_cases(name: str, limit: int | None = None) -> tuple[str, list[EvalCase]]:
    key = name.lower()
    if key in {"golden", "guardrails"}:
        cases = load_golden_cases()
        dataset = "golden-guardrails"
    elif key in {"hf-sample", "sample", "hf"}:
        cases = load_hf_sample()
        dataset = HF_DATASET + "#sample"
    elif key in {"hf-full", "huggingface"}:
        cases = download_hf_fit_dataset(limit=limit or 150, per_label=50)
        dataset = HF_DATASET
    else:
        raise ValueError(f"Unknown dataset '{name}'. Use golden, hf-sample, or hf-full.")
    if limit is not None:
        cases = cases[:limit]
    return dataset, cases
