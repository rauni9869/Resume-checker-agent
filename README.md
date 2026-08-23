# Resume Checker Agent

Production-style resume analysis service: a typed **LangGraph** DAG, deterministic ATS scoring, **open-source** generation/matching models (no OpenAI key), input/output **guardrails**, and an evaluation harness on a public Hugging Face resume–job dataset.

This is a rewrite of the original Colab notebook (`notebooks/original_colab.ipynb`) into something you can run, test, and talk about in interviews.

## Why this is stronger than a notebook demo

| Notebook (before) | This repo |
| --- | --- |
| Unbounded ReAct tool-calling with GPT-4o | Deterministic graph: validate → extract → score → critique → report |
| OCR + TF-IDF only | Hybrid ATS (skill taxonomy + TF-IDF) plus optional specialist embedders |
| No evals | Golden guardrail set + Hugging Face resume-job-fit ranking evals |
| Colab secrets | FastAPI + Docker + CI quality gates |
| SendGrid as the “result” | Structured JSON/HTML report; email is optional |

## Architecture

```
PDF / text
    → input guardrails (type, size, injection, PII)
    → native PDF text (OCR optional)
    → ATS skill coverage + TF-IDF
    → optional specialist models (Resumator, MiniLM cross-encoder, JobBERT-v3)
    → open-source critique (Ollama / Hugging Face) or offline template
    → groundedness + score-consistency guardrails
    → HTML / JSON report
```

Critique generation does **not** require a paid API:

- `template` (default) — citation-backed, deterministic reviewer used in CI
- `ollama` — local Llama / Qwen / Mistral
- `huggingface` — hosted open instruct models (e.g. Qwen2.5)

## Evaluation dataset

Primary public dataset:

- **[cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)** — 8k labeled resume / job-description pairs (`Good Fit`, `Potential Fit`, `No Fit`). Same family of data used by several ATS matching papers and models.

Vendored for CI (PII-redacted, truncated, 24 stratified test pairs):

- `evals/data/hf_resume_job_fit_sample.json`

Download a larger slice (no Kaggle account):

```bash
resume-checker download-dataset --limit 150 --output evals/data/hf_resume_job_fit_full.json
resume-checker evaluate --dataset hf-full --limit 90
```

Related datasets you can swap in later:

- [med2425/resume-job-fit-merged-v1](https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1) (~80k merged pairs)
- [0xnbk/resume-ats-score-v1-en](https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en) (continuous ATS scores)

## Specialist judge models

`resume-checker evaluate --specialists` scores each pair with three **domain** models, then reports Spearman correlation against the agent’s composite score:

1. **[shankerram3/resumator](https://huggingface.co/shankerram3/resumator)** — sentence-transformer fine-tuned on resume–JD pairs  
2. **[cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)** — relevance cross-encoder  
3. **[TechWolf/JobBERT-v3](https://huggingface.co/TechWolf/JobBERT-v3)** — job-title / skill embeddings (used on extracted skill phrases because the model is short-context)

Install extras first (downloads weights):

```bash
pip install -e ".[eval]"
resume-checker evaluate --dataset hf-sample --specialists
```

Metrics:

- **Pairwise ranking accuracy** — fraction of (Good Fit, No Fit) pairs where the agent scores the good match higher  
- **Spearman vs labels** — rank correlation with the dataset’s fit ordinal  
- **Groundedness** — share of critique quotes that actually appear in the resume  
- **Skill F1** — on the golden software-engineering cases  
- **Quality gate** — CI fails if golden guardrails or ranking thresholds regress

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
resume-checker evaluate --dataset golden
resume-checker evaluate --dataset hf-sample
```

## Dashboard

```bash
uvicorn resume_checker.api:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload a PDF (or paste resume text), paste a job description, and get a composite fit score with matched/missing skills, dimension bars, rewrite suggestions, and guardrail flags.

Analyze a PDF from the CLI:

```bash
resume-checker analyze --resume path/to/resume.pdf --job "Senior Python engineer, AWS, Docker..."
```

API: `POST /analyze` (multipart: `job_description`, `resume` PDF or `resume_text`).

Docker:

```bash
docker build -t resume-checker .
docker run -p 8000:8000 resume-checker
```

Local open-source LLM:

```bash
ollama pull llama3.1:8b
export RESUME_CHECKER_LLM_BACKEND=ollama
export OLLAMA_MODEL=llama3.1:8b
```

## Guardrails

- File type / size / page limits  
- Prompt-injection scan on resume and job text  
- PII redaction before LLM/template critique  
- “Does this look like a resume?” check  
- Structured critique schema (Pydantic)  
- Quote-level groundedness  
- Overall score clamped to dimension scores so the model cannot invent a 99 when evidence is weak  

## Resume bullets (drop-in)

**Resume Checker Agent** — Production LLMOps project (open source)

- Built a typed LangGraph pipeline (validate → extract → ATS score → critique → report) with FastAPI/Docker, replacing a Colab ReAct notebook.  
- Hybrid scoring: deterministic skill-taxonomy + TF-IDF ATS features, plus optional JobBERT-v3 / Resumator / MiniLM cross-encoder panel — no OpenAI dependency.  
- Evaluation harness on Hugging Face `resume-job-description-fit` (pairwise ranking + Spearman vs fit labels) and a golden set for injection, PII, and groundedness gates in CI.  
- Production guardrails: schema-validated output, citation checks, prompt-injection blocking, and PII redaction before generation.  
- Shipped a FastAPI scoring dashboard for resume PDF / JD upload with skill coverage, fit band, and grounded critique.

## License

MIT. Hugging Face eval text remains subject to the upstream dataset terms; the vendored sample is redacted for local testing.
