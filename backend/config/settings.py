"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Centralized application settings."""

    app_name: str = "SolvixAI"
    app_env: str = "development"
    debug: bool = True

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "solvixai"

    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    checkpoint_dir: Path = data_dir / "checkpoints"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()


settings = get_settings()
