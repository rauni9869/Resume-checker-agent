from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from resume_checker.schemas import AnalysisRequest, AnalysisResult
from resume_checker.graph.pipeline import analyze_resume

app = FastAPI(
    title="Resume Checker Agent",
    version="1.0.0",
    description="Production resume analysis with ATS scoring, open-source models, and eval guardrails.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    job_description: str = Form(...),
    candidate_id: str = Form("anonymous"),
    include_specialists: bool = Form(False),
    resume: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
) -> JSONResponse:
    if resume is None and not resume_text:
        raise HTTPException(status_code=400, detail="Provide a PDF upload or resume_text.")
    tmp_path = None
    if resume is not None:
        data = await resume.read()
        tmp_path = Path("/tmp") / (resume.filename or "resume.pdf")
        tmp_path.write_bytes(data)
    request = AnalysisRequest(
        job_description=job_description,
        resume_text=resume_text,
        candidate_id=candidate_id,
    )
    result = analyze_resume(request, resume_path=str(tmp_path) if tmp_path else None, include_specialists=include_specialists)
    return JSONResponse(result.model_dump())


@app.post("/analyze/html", response_class=HTMLResponse)
async def analyze_html(
    job_description: str = Form(...),
    candidate_id: str = Form("anonymous"),
    resume: UploadFile | None = File(None),
    resume_text: str | None = Form(None),
) -> HTMLResponse:
    request = AnalysisRequest(
        job_description=job_description,
        resume_text=resume_text,
        candidate_id=candidate_id,
    )
    tmp_path = None
    if resume is not None:
        data = await resume.read()
        tmp_path = Path("/tmp") / (resume.filename or "resume.pdf")
        tmp_path.write_bytes(data)
    result = analyze_resume(request, resume_path=str(tmp_path) if tmp_path else None)
    return HTMLResponse(result.html_report or "<p>No report.</p>")
