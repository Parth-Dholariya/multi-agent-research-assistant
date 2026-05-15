from __future__ import annotations

import requests

from .config import Settings


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.has_azure_openai or self.settings.has_openai

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str | None:
        if self.settings.has_azure_openai:
            return self._complete_azure(system_prompt, user_prompt, max_tokens)
        if self.settings.has_openai:
            return self._complete_openai(system_prompt, user_prompt, max_tokens)
        return None

    def _complete_azure(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        endpoint = self.settings.azure_openai_endpoint.rstrip("/")
        deployment = self.settings.azure_openai_deployment
        url = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.settings.azure_openai_api_version}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            url,
            headers={"api-key": self.settings.azure_openai_api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=40,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _complete_openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=40,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

