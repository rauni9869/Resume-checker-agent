from __future__ import annotations

from resume_checker.evals.dataset import load_cases
from resume_checker.evals.judges import run_specialist_judges
from resume_checker.evals.metrics import (
    LABEL_ORDINAL,
    bin_label,
    label_accuracy,
    pairwise_ranking_accuracy,
    precision_recall_f1,
    spearman,
)
from resume_checker.graph.pipeline import analyze_resume
from resume_checker.schemas import AnalysisRequest, CaseEvalReport, EvalCase, EvalSuiteReport

DEFAULT_THRESHOLDS = {
    "min_pass_rate": 0.75,
    "min_groundedness": 0.6,
    "min_pairwise_ranking": 0.65,
}


def evaluate_case(case: EvalCase, include_specialists: bool) -> CaseEvalReport:
    result = analyze_resume(
        AnalysisRequest(
            job_description=case.job_description,
            resume_text=case.resume_text,
            candidate_id=case.id,
        )
    )
    failures: list[str] = []
    predicted = set(result.ats.matched_skills) if result.ats else set()
    expected_present = set(case.expected_skills_present)
    if expected_present:
        precision, recall, f1 = precision_recall_f1(predicted, expected_present)
    else:
        precision, recall, f1 = 1.0, 1.0, 1.0
    if expected_present and f1 < 0.5:
        failures.append(f"skill_f1={f1:.2f}")
    if case.expected_skills_missing and result.ats:
        _, miss_recall, _ = precision_recall_f1(
            set(result.ats.missing_skills), set(case.expected_skills_missing)
        )
        if miss_recall < 0.5:
            failures.append(f"missing_skill_recall={miss_recall:.2f}")

    score = result.composite_score if result.composite_score is not None else 0.0
    in_band = case.expected_score_min <= score <= case.expected_score_max
    if result.blocked and case.should_block:
        in_band = True
    if not case.should_block and not in_band and case.expected_skills_present:
        failures.append(f"score {score:.1f} outside {case.expected_score_min}-{case.expected_score_max}")

    blocked_ok = result.blocked is case.should_block
    if not blocked_ok:
        failures.append(f"blocked={result.blocked} expected={case.should_block}")

    schema_valid = bool(result.blocked or result.critique)
    groundedness = 1.0
    if result.critique and result.critique.evidence:
        groundedness = sum(1 for item in result.critique.evidence if item.found_in_source) / len(
            result.critique.evidence
        )

    judges = (
        run_specialist_judges(case.resume_text, case.job_description) if include_specialists else []
    )
    return CaseEvalReport(
        case_id=case.id,
        passed=not failures,
        skill_precision=round(precision, 3),
        skill_recall=round(recall, 3),
        skill_f1=round(f1, 3),
        score_in_band=in_band,
        groundedness=round(groundedness, 3),
        schema_valid=schema_valid,
        blocked_as_expected=blocked_ok,
        composite_score=round(score, 2),
        predicted_label=None if case.should_block else bin_label(score),
        judge_scores=judges,
        failures=failures,
    )


def run_eval_suite(
    dataset: str = "golden",
    include_specialists: bool = False,
    limit: int | None = None,
    thresholds: dict[str, float] | None = None,
) -> EvalSuiteReport:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if dataset.lower() in {"hf", "hf-sample", "sample", "hf-full", "huggingface"}:
        thresholds = {
            **thresholds,
            "min_pass_rate": 0.0,
            "min_pairwise_ranking": 0.55,
        }
    dataset_name, cases = load_cases(dataset, limit=limit)
    reports = [evaluate_case(case, include_specialists=include_specialists) for case in cases]
    n = len(reports) or 1
    pass_rate = sum(1 for report in reports if report.passed) / n
    mean_f1 = sum(report.skill_f1 for report in reports) / n
    mean_g = sum(report.groundedness for report in reports) / n

    ranked_pairs = [
        (case, report)
        for case, report in zip(cases, reports, strict=True)
        if not case.should_block
    ]
    scores = [report.composite_score for _, report in ranked_pairs]
    labels = [case.label or "" for case, _ in ranked_pairs]
    ranking = pairwise_ranking_accuracy(scores, labels) if scores else None
    acc = label_accuracy(scores, labels) if scores else None
    if labels and all(label in LABEL_ORDINAL for label in labels):
        spearman_labels = spearman(scores, [LABEL_ORDINAL[label] for label in labels])
    else:
        spearman_labels = None

    judge_means: dict[str, float] = {}
    spearman_judges: dict[str, float] = {}
    if include_specialists:
        by_model: dict[str, list[float]] = {}
        for _, report in ranked_pairs:
            for judge in report.judge_scores:
                if judge.available:
                    by_model.setdefault(judge.model_id, []).append(judge.score)
        for model_id, values in by_model.items():
            judge_means[model_id] = round(sum(values) / len(values), 2)
            if len(values) == len(scores):
                spearman_judges[model_id] = spearman(scores, values)

    gate = pass_rate >= thresholds["min_pass_rate"] and mean_g >= thresholds["min_groundedness"]
    if ranking is not None:
        gate = gate and ranking >= thresholds["min_pairwise_ranking"]

    return EvalSuiteReport(
        dataset=dataset_name,
        cases=reports,
        n=len(reports),
        pass_rate=round(pass_rate, 3),
        mean_skill_f1=round(mean_f1, 3),
        mean_groundedness=round(mean_g, 3),
        label_accuracy=acc,
        pairwise_ranking_accuracy=ranking,
        spearman_vs_labels=spearman_labels,
        judge_means=judge_means,
        spearman_vs_judges=spearman_judges,
        gate_passed=gate,
        thresholds=thresholds,
    )
