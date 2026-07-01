from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    artifact_dir: Path = Field(default=Path("artifacts"), validation_alias="COMEBACK_ARTIFACT_DIR")
    knowledge_dir: Path = Field(
        default=Path("knowledge"), validation_alias="COMEBACK_KNOWLEDGE_DIR"
    )
    groq_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "COMEBACK_GROQ_API_KEY")
    )
    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias=AliasChoices("GROQ_MODEL", "COMEBACK_GROQ_MODEL"),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
