import os

import pytest

from resume_checker.config import get_settings
from resume_checker.scoring.matcher import get_encoder


@pytest.fixture(autouse=True)
def _vector_backend(monkeypatch):
    monkeypatch.setenv("RESUME_CHECKER_EMBEDDING_BACKEND", "tfidf")
    get_settings.cache_clear()
    get_encoder.cache_clear()
    yield
    get_settings.cache_clear()
    get_encoder.cache_clear()
    os.environ.pop("RESUME_CHECKER_EMBEDDING_BACKEND", None)
