"""Kernel loading and prompt rendering."""
from ..core.schemas import parse_llm_response
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..core.types import DocumentSchema, Extraction, Evidence, ExtractionStatus


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt ready for LLM."""
    system: str       # The kernel (system prompt)
    user: str         # The document (user message)
    kernel_name: str

    @property
    def token_estimate(self) -> int:
        return len(self.system.split()) + len(self.user.split())


class Kernel(ABC):
    """Base class for extraction kernels."""

    name: str

    @abstractmethod
    def render(self, document: str) -> RenderedPrompt:
        """Build prompt from document."""
        ...

    def parse_response(self, response: str) -> list[Extraction]:
        """Parse LLM response into extractions."""
        return parse_llm_response(response)


class TextKernel(Kernel):
    """Kernel loaded from a .txt template file."""

    def __init__(self, path: Path):
        self._template = path.read_text(encoding="utf-8")
        self._path = path
        self.name = path.stem

    @property
    def raw_content(self) -> str:
        return self._template

    def render(self, document: str) -> RenderedPrompt:
        return RenderedPrompt(
            system=self._template,
            user=document,
            kernel_name=self.name
        )


class NullBaseline(Kernel):
    """Always outputs missing. Bypasses LLM."""

    name = "null_baseline"

    def __init__(self, schema: DocumentSchema):
        self._schema = schema

    def render(self, document: str) -> RenderedPrompt:
        return RenderedPrompt(
            system="",
            user="__NULL_BASELINE__",
            kernel_name=self.name
        )

    def extract_directly(self) -> list[Extraction]:
        """Direct extraction without LLM call."""
        return [
            Extraction(
                field_name=f.name,
                value=None,
                evidence=Evidence(quote=None, page=None),
                status=ExtractionStatus.MISSING,
                candidates=[],
            )
            for f in self._schema.fields
        ]


def load_kernel(path: Path, schema: DocumentSchema) -> Kernel:
    """Load kernel from path. Special-cases null_baseline."""
    if path.name == "__null__" or path.stem == "null_baseline":
        return NullBaseline(schema)

    if not path.exists():
        raise FileNotFoundError(f"Kernel not found: {path}")

    return TextKernel(path)
