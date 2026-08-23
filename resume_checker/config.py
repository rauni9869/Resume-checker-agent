from functools import lru_cache
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config. Open-source models only — no OpenAI key required."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="RESUME_CHECKER_"
    )

    llm_backend: str = "template"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_token: str = ""

    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    resume_matcher_model: str = "shankerram3/resumator"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    job_title_model: str = "TechWolf/JobBERT-v3"
    download_eval_models: bool = False

    sendgrid_api_key: str = ""
    from_email: str = "noreply@example.com"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    max_upload_mb: int = 8
    max_pages: int = 6
    enable_ocr: bool = False
    redact_pii: bool = True
    min_extract_chars: int = 80
    groundedness_min_ratio: float = 0.6
    injection_block_threshold: int = 2

    def model_post_init(self, __context) -> None:
        mapping = {
            "ollama_model": "OLLAMA_MODEL",
            "ollama_base_url": "OLLAMA_BASE_URL",
            "hf_model": "HF_MODEL",
            "hf_token": "HF_TOKEN",
            "sendgrid_api_key": "SENDGRID_API_KEY",
            "langfuse_public_key": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key": "LANGFUSE_SECRET_KEY",
            "langfuse_host": "LANGFUSE_HOST",
        }
        for field, env_name in mapping.items():
            if os.getenv(env_name):
                object.__setattr__(self, field, os.environ[env_name])


@lru_cache
def get_settings() -> Settings:
    return Settings()
