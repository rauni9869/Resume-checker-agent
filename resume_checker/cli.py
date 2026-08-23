from __future__ import annotations

import argparse
import json
from pathlib import Path

from resume_checker.evals.dataset import download_hf_fit_dataset
from resume_checker.evals.runner import run_eval_suite
from resume_checker.graph.pipeline import analyze_resume
from resume_checker.schemas import AnalysisRequest


def _job_text(value: str) -> str:
    path = Path(value)
    return path.read_text(encoding="utf-8") if path.exists() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume Checker Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Score a resume against a job description")
    analyze.add_argument("--resume", help="Path to a PDF resume")
    analyze.add_argument("--resume-text", help="Raw resume text")
    analyze.add_argument("--job", required=True, help="Job description text or path to a .txt file")
    analyze.add_argument("--specialists", action="store_true")
    analyze.add_argument("--json", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Run evaluation on golden or Hugging Face datasets")
    evaluate.add_argument(
        "--dataset",
        default="golden",
        help="golden | hf-sample | hf-full",
    )
    evaluate.add_argument("--specialists", action="store_true", help="Score with Resumator, MiniLM cross-encoder, JobBERT-v3")
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--output", default="artifacts/eval_report.json")

    download = sub.add_parser("download-dataset", help="Fetch Hugging Face resume-job-fit pairs")
    download.add_argument("--limit", type=int, default=150)
    download.add_argument("--output", default="evals/data/hf_resume_job_fit_full.json")

    args = parser.parse_args()
    if args.command == "analyze":
        request = AnalysisRequest(job_description=_job_text(args.job), resume_text=args.resume_text)
        result = analyze_resume(
            request, resume_path=args.resume, include_specialists=args.specialists
        )
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(f"composite={result.composite_score} blocked={result.blocked} backend={result.llm_backend}")
            if result.ats:
                print("matched:", ", ".join(result.ats.matched_skills) or "—")
                print("missing:", ", ".join(result.ats.missing_skills) or "—")
            for finding in result.guardrails:
                print(f"[{finding.severity.value}] {finding.code}: {finding.message}")
        return

    if args.command == "download-dataset":
        cases = download_hf_fit_dataset(limit=args.limit, per_label=max(1, args.limit // 3))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "cnamuangtoun/resume-job-description-fit",
            "n": len(cases),
            "cases": [case.model_dump() for case in cases],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output} ({len(cases)} pairs)")
        return

    report = run_eval_suite(
        dataset=args.dataset, include_specialists=args.specialists, limit=args.limit
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report.model_dump(), indent=2))
    summary = {
        "dataset": report.dataset,
        "n": report.n,
        "pass_rate": report.pass_rate,
        "pairwise_ranking_accuracy": report.pairwise_ranking_accuracy,
        "spearman_vs_labels": report.spearman_vs_labels,
        "label_accuracy": report.label_accuracy,
        "mean_groundedness": report.mean_groundedness,
        "judge_means": report.judge_means,
        "spearman_vs_judges": report.spearman_vs_judges,
        "gate_passed": report.gate_passed,
    }
    print(json.dumps(summary, indent=2))
    if not report.gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
