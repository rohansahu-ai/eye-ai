"""
Configuration management using Pydantic BaseSettings.
All values can be overridden via environment variables or .env file.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # AWS
    AWS_REGION: str = "us-east-1"
    AWS_BEDROCK_MODEL_ID: str = "us.anthropic.claude-sonnet-4-6"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    BEDROCK_BEARER_TOKEN: Optional[str] = None  # Set via .env or environment variable

    # Paths
    DATA_DIR: str = "data/synthetic"
    MODEL_DIR: str = "models/artifacts"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Forecasting
    FORECAST_HORIZON_DAYS: int = 90

    # Similarity
    SIMILARITY_TOP_N: int = 5

    # Supply chain
    REORDER_SAFETY_STOCK_DAYS: int = 30

    # Feature flags
    MOCK_BEDROCK: bool = False  # Default True so app works without AWS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    s = get_settings()
    print(s.model_dump())
