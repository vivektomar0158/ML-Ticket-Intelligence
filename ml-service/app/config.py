from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    gemini_api_key: str = ""
    # comma-separated, tried round-robin; models over quota are skipped (free tier has tiny per-model quotas)
    gemini_models: str = "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-3.6-flash,gemini-flash-lite-latest,gemini-flash-latest,gemini-2.5-flash-lite,gemini-2.5-flash"
    llm_provider: str = "gemini"          # gemini | mock  (mock = deterministic offline stub for tests/demos)
    llm_concurrency: int = 4
    llm_timeout_s: float = 45.0
    llm_cache_dir: str = str(ROOT / "ml-service" / ".llm_cache")
    # approximate USD per 1M tokens (configurable; recorded with every call so cost reports are reproducible)
    price_in_per_m: float = 0.10
    price_out_per_m: float = 0.40
    artifacts_dir: str = str(ROOT / "ml-service" / "artifacts")
    confidence_threshold: float = 0.80    # cascade: below this, category goes to the LLM


settings = Settings()
