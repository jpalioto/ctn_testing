"""Network client with retry/backoff and multimodal support."""
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
from anthropic.types import (
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
    Base64ImageSourceParam,
    Base64PDFSourceParam,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
)

from ..core.config import ModelConfig

if TYPE_CHECKING:
    from ..runners.kernel import RenderedPrompt

RETRYABLE_CODES = ["429", "500", "502", "503", "504", "529", "overloaded"]


@dataclass(frozen=True)
class CompletionResult:
    """Result from a completion API call."""
    text: str
    input_tokens: int
    output_tokens: int


class CompleteFunction(Protocol):
    def __call__(self, prompt: "RenderedPrompt") -> CompletionResult: ...


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
    def complete(prompt: "RenderedPrompt") -> CompletionResult:
    
        content: list[TextBlockParam | ImageBlockParam | DocumentBlockParam] = []
        
        if prompt.has_file and prompt.document:
            doc = prompt.document
            if not doc.media_type:
                raise ValueError(f"Document {doc.id} has no media type")
            
            if doc.is_pdf:
                content.append(DocumentBlockParam(
                    type="document",
                    source=Base64PDFSourceParam(
                        type="base64",
                        media_type="application/pdf",
                        data=doc.file_base64(),
                    ),
                ))
            elif doc.is_image:
                content.append(ImageBlockParam(
                    type="image",
                    source=Base64ImageSourceParam(
                        type="base64",
                        media_type=doc.media_type,  # type: ignore[arg-type]
                        data=doc.file_base64(),
                    ),
                ))
        
        content.append(TextBlockParam(type="text", text=prompt.user))
        
        response = client.messages.create(
            model=config.name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=prompt.system,
            messages=[{"role": "user", "content": content}],
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
    from google.genai import types
    
    client = genai.Client()
    
    @retry_decorator
    def complete(prompt: "RenderedPrompt") -> CompletionResult:
        # Build content parts
        parts: list = []
        
        # Add document file if present
        if prompt.has_file and prompt.document:
            doc = prompt.document
            if not doc.media_type:
                raise ValueError(f"Document {doc.id} has no media type")
            parts.append(types.Part.from_bytes(
                data=doc.file_bytes(),
                mime_type=doc.media_type,
            ))
        
        # Add text prompt
        parts.append(types.Part.from_text(text=prompt.user))
        
        response = client.models.generate_content(
            model=config.name,
            contents=parts,
            config={
                "system_instruction": prompt.system,
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
    def complete(prompt: "RenderedPrompt") -> CompletionResult:
        # Build content array
        content: list[dict] = []
        
        # Add document file if present (OpenAI only supports images)
        if prompt.has_file and prompt.document:
            doc = prompt.document
            if doc.is_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{doc.media_type};base64,{doc.file_base64()}"
                    }
                })
            elif doc.is_pdf:
                # OpenAI doesn't support native PDF - fall back to text if available
                if doc.has_text and doc.text:
                    content.append({"type": "text", "text": f"[Document text]:\n{doc.text}"})
                else:
                    raise ValueError("OpenAI doesn't support native PDF. Provide text fallback.")
        
        # Add text prompt
        content.append({"type": "text", "text": prompt.user})
        
        response = client.chat.completions.create(
            model=config.name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": content},
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