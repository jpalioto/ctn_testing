"""Core types for CTN evaluation."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExtractionStatus(Enum):
    OK = "ok"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Evidence:
    quote: str | None
    page: int | None


@dataclass(frozen=True)
class Candidate:
    value: Any
    quote: str | None
    page: int | None = None


@dataclass
class Extraction:
    field: str
    value: Any
    evidence: Evidence
    status: ExtractionStatus
    confidence: Confidence = Confidence.HIGH
    candidates: list[Candidate] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, d: dict) -> "Extraction":
        """Parse from model output."""
        ev = d.get("evidence") or {}
        if ev is None:
            ev = {}
        
        candidates = [
            Candidate(
                value=c.get("value"),
                quote=c.get("quote"),
                page=c.get("page")
            )
            for c in d.get("candidates", [])
        ]
        
        return cls(
            field=d["field"],
            value=d.get("value"),
            evidence=Evidence(
                quote=ev.get("quote"),
                page=ev.get("page")
            ),
            status=ExtractionStatus(d.get("status", "ok")),
            confidence=Confidence(d.get("confidence", "high")),
            candidates=candidates
        )


@dataclass
class GroundTruth:
    field: str
    exists_in_document: bool
    is_ambiguous: bool = False
    value: Any = None
    acceptable_values: list[Any] = field(default_factory=list)
    candidate_values: list[Any] = field(default_factory=list)
    evidence_quote: str | None = None
    evidence_page: int | None = None


@dataclass(frozen=True)
class CompositeResult:
    composite: float
    value: float
    evidence: float
    page: float
    status: float
    schema: float


@dataclass
class FieldSchema:
    name: str
    description: str
    type: str = "string"
    required: bool = True


@dataclass
class DocumentSchema:
    name: str
    fields: list[FieldSchema]
    
    def to_prompt(self) -> str:
        lines = [f"Schema: {self.name}", "Fields:"]
        for f in self.fields:
            lines.append(f"  - {f.name}: {f.description} ({f.type})")
        return "\n".join(lines)
