"""LLM provider abstraction. Add providers to PROVIDERS dict."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    LLM provider configuration.

    provider: anthropic | google | openai
    name: Model identifier (e.g. claude-sonnet-4-5-latest)
    api_key_env: Environment variable containing API key
    temperature: Sampling temperature, 0.0 = deterministic
    max_tokens: Maximum output tokens
    """

    provider: str
    name: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048

    def get_api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EnvironmentError(f"{self.api_key_env} not set")
        return key


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class ModelClient(ABC):
    """Interface for LLM providers."""

    def __init__(self, config: ModelConfig):
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @abstractmethod
    def complete(self, system: str, user: str) -> CompletionResult:
        """Complete with system prompt and user message."""
        ...


class AnthropicClient(ModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._config.get_api_key())
        return self._client

    def complete(self, system: str, user: str) -> CompletionResult:
        client = self._get_client()
        response = client.messages.create(
            model=self._config.name,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return CompletionResult(
            text=response.content[0].text
            if hasattr(response.content[0], "text")
            else str(response.content[0]),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._config.name,
        )


class GoogleClient(ModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._config.get_api_key())
        return self._client

    def complete(self, system: str, user: str) -> CompletionResult:
        client = self._get_client()
        response = client.models.generate_content(
            model=self._config.name,
            contents=user,
            config={
                "temperature": self._config.temperature,
                "max_output_tokens": self._config.max_tokens,
                "system_instruction": system,
                "response_mime_type": "application/json",
            },
        )
        return CompletionResult(
            text=response.text or "",
            input_tokens=(response.usage_metadata.prompt_token_count or 0)
            if response.usage_metadata
            else 0,
            output_tokens=(response.usage_metadata.candidates_token_count or 0)
            if response.usage_metadata
            else 0,
            model=self._config.name,
        )


class OpenAIClient(ModelClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._config.get_api_key())
        return self._client

    def complete(self, system: str, user: str) -> CompletionResult:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._config.name,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return CompletionResult(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=self._config.name,
        )


PROVIDERS = {
    "anthropic": AnthropicClient,
    "google": GoogleClient,
    "openai": OpenAIClient,
}


def get_client(config: ModelConfig) -> ModelClient:
    """Factory. Raises KeyError if provider unknown."""
    if config.provider not in PROVIDERS:
        raise KeyError(f"Unknown provider: {config.provider}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[config.provider](config)
