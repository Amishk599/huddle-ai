"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_STORES_DIR: Path = DATA_DIR / "vector_stores"
    SAMPLE_TRANSCRIPTS_DIR: Path = DATA_DIR / "sample_transcripts"
    EMAILS_DIR: Path = BASE_DIR / "emails_sent"

    # API Keys
    OPENAI_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "huddle-ai"

    # Models
    DEFAULT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
