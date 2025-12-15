"""Network client with retry/backoff."""
from dataclasses import dataclass
from typing import Protocol

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
)

from ..core.config import ModelConfig

RETRYABLE_CODES = ["429", "500", "502", "503", "504", "529", "overloaded"]

@dataclass(frozen=True)
class CompletionResult:
    """Result from a completion API call."""
    text: str
    input_tokens: int
    output_tokens: int


class CompleteFunction(Protocol):
    def __call__(self, system: str, user: str) -> CompletionResult: ...


def is_retryable(e: BaseException) -> bool:
    """Check if exception is retryable."""
    error_str = str(e).lower()
    return any(code in error_str for code in RETRYABLE_CODES)


def _log_retry(retry_state):
    print(f"    Retry {retry_state.attempt_number} after {retry_state.outcome.exception()}")


def _make_anthropic_client(config: ModelConfig, retry_decorator) -> CompleteFunction:
    from anthropic import Anthropic
    
    client = Anthropic()
    
    @retry_decorator
    def complete(system: str, user: str) -> CompletionResult:
        response = client.messages.create(
            model=config.name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        
        text = "".join(
            block.text for block in response.content 
            if hasattr(block, "text")
        )
        
        return CompletionResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    
    return complete


def _make_google_client(config: ModelConfig, retry_decorator) -> CompleteFunction:
    from google import genai
    
    client = genai.Client()
    
    @retry_decorator
    def complete(system: str, user: str) -> CompletionResult:
        response = client.models.generate_content(
            model=config.name,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            }
        )
        
        text = response.text or ""
        usage = response.usage_metadata
        
        return CompletionResult(
            text=text,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )
    
    return complete


def _make_openai_client(config: ModelConfig, retry_decorator) -> CompleteFunction:
    import openai
    
    client = openai.OpenAI()
    
    @retry_decorator
    def complete(system: str, user: str) -> CompletionResult:
        response = client.chat.completions.create(
            model=config.name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        
        return CompletionResult(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
    
    return complete


_FACTORIES = {
    "anthropic": _make_anthropic_client,
    "google": _make_google_client,
    "openai": _make_openai_client,
}

def make_client(config: ModelConfig) -> CompleteFunction:
    retry_decorator = retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        retry=retry_if_exception(is_retryable),
        before_sleep=_log_retry,
    )
    
    factory = _FACTORIES.get(config.provider)
    if not factory:
        raise ValueError(f"Unknown provider: {config.provider}")
    return factory(config, retry_decorator)