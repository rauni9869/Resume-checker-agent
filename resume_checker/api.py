from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from resume_checker.graph.pipeline import analyze_resume
from resume_checker.schemas import AnalysisRequest, AnalysisResult

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Resume Checker Agent",
    version="1.0.0",
    description="Production resume analysis with ATS scoring, open-source models, and eval guardrails.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _write_pdf(data: bytes) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    handle.write(data)
    handle.close()
    return handle.name


def run_analysis(
    job_description: str,
    candidate_id: str,
    include_specialists: bool,
    filename: str | None,
    file_bytes: bytes | None,
    resume_text: str | None,
) -> AnalysisResult:
    tmp_path = None
    text = (resume_text or "").strip() or None
    if file_bytes:
        name = (filename or "").lower()
        is_pdf = file_bytes.startswith(b"%PDF") or name.endswith(".pdf")
        if is_pdf:
            tmp_path = _write_pdf(file_bytes)
            text = None
        else:
            text = file_bytes.decode("utf-8", errors="replace")
    if not tmp_path and not text:
        raise HTTPException(status_code=400, detail="Provide a PDF/TXT upload or paste resume text.")
    request = AnalysisRequest(
        job_description=job_description,
        resume_text=text,
        candidate_id=candidate_id or "anonymous",
    )
    return analyze_resume(request, resume_path=tmp_path, include_specialists=include_specialists)


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    job_description: str = Form(...),
    candidate_id: str = Form("anonymous"),
    include_specialists: bool = Form(False),
    resume: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
) -> JSONResponse:
    data = await resume.read() if resume is not None else None
    filename = resume.filename if resume is not None else None
    result = run_analysis(
        job_description=job_description,
        candidate_id=candidate_id,
        include_specialists=include_specialists,
        filename=filename,
        file_bytes=data if data else None,
        resume_text=resume_text,
    )
    return JSONResponse(result.model_dump(mode="json"))
