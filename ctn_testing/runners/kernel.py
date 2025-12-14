"""Kernel loading and prompt rendering."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json

from ..core.types import DocumentSchema, Extraction, Evidence, ExtractionStatus


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt ready for LLM."""
    text: str
    kernel_name: str
    
    @property
    def token_estimate(self) -> int:
        return len(self.text.split())


class Kernel(ABC):
    """Base class for extraction kernels."""
    
    name: str
    
    @abstractmethod
    def render(self, schema: DocumentSchema, document: str, output_format: dict) -> RenderedPrompt:
        """Build prompt from schema and document."""
        ...
    
    def parse_response(self, response: str) -> list[Extraction]:
        """Parse LLM response into extractions."""
        text = response.strip()
        
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Find closing fence
            end_idx = len(lines) - 1
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[1:end_idx])
        
        data = json.loads(text)
        extractions = data.get("extractions", [])
        return [Extraction.from_dict(e) for e in extractions]


class TextKernel(Kernel):
    """Kernel loaded from a .txt template file."""
    
    def __init__(self, path: Path):
        self._template = path.read_text(encoding="utf-8")
        self._path = path
        self.name = path.stem
    
    def render(self, schema: DocumentSchema, document: str, output_format: dict) -> RenderedPrompt:
        prompt = self._template.format(
            schema=schema.to_prompt(),
            output_format=json.dumps(output_format, indent=2),
        )
        # Append document
        prompt = prompt + "\n" + document
        
        return RenderedPrompt(text=prompt, kernel_name=self.name)


class NullBaseline(Kernel):
    """Always outputs missing. Bypasses LLM."""
    
    name = "null_baseline"
    
    def __init__(self, schema: DocumentSchema):
        self._schema = schema
    
    def render(self, schema: DocumentSchema, document: str, output_format: dict) -> RenderedPrompt:
        return RenderedPrompt(text="__NULL_BASELINE__", kernel_name=self.name)
    
    def extract_directly(self) -> list[Extraction]:
        """Direct extraction without LLM call."""
        return [
            Extraction(
                field=f.name,
                value=None,
                evidence=Evidence(quote=None, page=None),
                status=ExtractionStatus.MISSING,
                candidates=[],
            )
            for f in self._schema.fields
        ]
    
    def parse_response(self, response: str) -> list[Extraction]:
        raise NotImplementedError("NullBaseline doesn't use LLM")


def load_kernel(path: Path, schema: DocumentSchema) -> Kernel:
    """Load kernel from path. Special-cases null_baseline."""
    if path.name == "__null__" or path.stem == "null_baseline":
        return NullBaseline(schema)
    
    if not path.exists():
        raise FileNotFoundError(f"Kernel not found: {path}")
    
    return TextKernel(path)


# Standard output format for all kernels
OUTPUT_FORMAT = {
    "extractions": [
        {
            "field": "<field_name>",
            "value": "<extracted_value_or_null>",
            "evidence": {
                "quote": "<verbatim_quote_from_document_or_null>",
                "page": "<page_number_int_or_null>"
            },
            "status": "ok | missing | ambiguous",
            "confidence": "high | medium | low",
            "candidates": [
                {
                    "value": "<candidate_value>",
                    "quote": "<supporting_quote>",
                    "page": "<page_number>"
                }
            ]
        }
    ]
}
