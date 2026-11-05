"""Settings, loaded from environment / .env."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: SecretStr
    gemini_model: str = "gemini-3.5-flash-lite"

    mcp_url: str = "http://127.0.0.1:8001/mcp"

    max_llm_calls: int = Field(10, ge=1)

    agent_host: str = "127.0.0.1"
    agent_port: int = Field(8002, ge=1, le=65535)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
