from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    semantic_scholar_api_key: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    azure_openai_endpoint: str | None = os.getenv("AZURE_OPENAI_ENDPOINT") or None
    azure_openai_api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY") or None
    azure_openai_deployment: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT") or None
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def has_azure_openai(self) -> bool:
        return bool(
            self.azure_openai_endpoint
            and self.azure_openai_api_key
            and self.azure_openai_deployment
        )

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

