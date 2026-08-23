from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from resume_checker.config import Settings, get_settings
from resume_checker.schemas import ExtractionMethod, ExtractionResult

PDF_MAGIC = b"%PDF"


def _quality_score(text: str, page_count: int) -> float:
    chars = len(text.strip())
    alpha = sum(ch.isalpha() for ch in text)
    ratio = alpha / max(len(text), 1)
    length_score = min(chars / 1200, 1.0) * 70
    alpha_score = min(ratio / 0.7, 1.0) * 20
    page_penalty = 10 if page_count > 3 else 10
    return round(min(100.0, length_score + alpha_score + page_penalty), 1)


def _ocr_fallback(pdf_path: str) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR extras are not installed. Install with resume-checker-agent[ocr].") from exc

    images = convert_from_path(pdf_path)
    chunks = [pytesseract.image_to_string(image, lang="eng") for image in images]
    text = "\n".join(chunks)
    text = re.sub(r"\n+", "\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def extract_from_pdf(pdf_path: str | Path, settings: Settings | None = None) -> ExtractionResult:
    settings = settings or get_settings()
    path = Path(pdf_path)
    data = path.read_bytes()
    if not data.startswith(PDF_MAGIC):
        raise ValueError("File is not a PDF.")

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        page_count = doc.page_count
        native = "\n".join(page.get_text() for page in doc).strip()

    if len(native) >= settings.min_extract_chars:
        return ExtractionResult(
            text=native,
            method=ExtractionMethod.NATIVE,
            page_count=page_count,
            char_count=len(native),
            quality_score=_quality_score(native, page_count),
        )

    if settings.enable_ocr:
        ocr_text = _ocr_fallback(str(path))
        return ExtractionResult(
            text=ocr_text,
            method=ExtractionMethod.OCR,
            page_count=page_count,
            char_count=len(ocr_text),
            quality_score=_quality_score(ocr_text, page_count),
        )

    return ExtractionResult(
        text=native,
        method=ExtractionMethod.NATIVE,
        page_count=page_count,
        char_count=len(native),
        quality_score=_quality_score(native, page_count),
    )


def extract_from_text(text: str) -> ExtractionResult:
    cleaned = text.strip()
    return ExtractionResult(
        text=cleaned,
        method=ExtractionMethod.TEXT,
        page_count=1,
        char_count=len(cleaned),
        quality_score=_quality_score(cleaned, 1),
    )
