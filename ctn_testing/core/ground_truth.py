"""Ground truth models and document composition for evaluation.

GroundTruth: Expected extraction values for a single field
DocumentWithGroundTruth: Composition pairing a Document with its labels
"""
from dataclasses import dataclass, field
from typing import Any

from .document import Document


@dataclass
class GroundTruth:
    """Ground truth for a single extractable field.
    
    Supports:
    - Simple exact match (value)
    - Multiple acceptable values (acceptable_values)
    - Ambiguous fields with candidates (candidate_values)
    - Fields that don't exist in document (exists_in_document=False)
    - Evidence location for validation (evidence_quote, evidence_page)
    """
    field_name: str
    
    # Expected value(s)
    value: Any = None                           # Primary expected value
    acceptable_values: list[Any] = field(default_factory=list)  # Also correct
    candidate_values: list[Any] = field(default_factory=list)   # Ambiguous options
    
    # Field characteristics
    exists_in_document: bool = True             # False if field should be null
    is_ambiguous: bool = False                  # True if multiple valid interpretations
    
    # Evidence location
    evidence_quote: str | None = None           # Supporting text from document
    evidence_page: int | None = None            # Page number (0-indexed)
    
    # Metadata
    notes: str | None = None                    # Annotation notes
    
    def matches(self, extracted: Any) -> bool:
        """Check if extracted value matches ground truth.
        
        Returns True if:
        - extracted == value
        - extracted in acceptable_values
        - Field doesn't exist and extracted is None/empty
        """
        if not self.exists_in_document:
            return extracted is None or extracted == ""
        
        # Normalize for comparison
        extracted_norm = self._normalize(extracted)
        
        if extracted_norm == self._normalize(self.value):
            return True
        
        for acceptable in self.acceptable_values:
            if extracted_norm == self._normalize(acceptable):
                return True
        
        return False
    
    def _normalize(self, val: Any) -> Any:
        """Normalize value for comparison."""
        if val is None:
            return None
        if isinstance(val, str):
            return val.strip().lower()
        return val
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "field_name": self.field_name,
            "value": self.value,
            "acceptable_values": self.acceptable_values,
            "candidate_values": self.candidate_values,
            "exists_in_document": self.exists_in_document,
            "is_ambiguous": self.is_ambiguous,
            "evidence_quote": self.evidence_quote,
            "evidence_page": self.evidence_page,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruth":
        """Deserialize from dictionary."""
        return cls(
            field_name=d["field_name"],
            value=d.get("value"),
            acceptable_values=d.get("acceptable_values", []),
            candidate_values=d.get("candidate_values", []),
            exists_in_document=d.get("exists_in_document", True),
            is_ambiguous=d.get("is_ambiguous", False),
            evidence_quote=d.get("evidence_quote"),
            evidence_page=d.get("evidence_page"),
            notes=d.get("notes"),
        )


@dataclass
class DocumentWithGroundTruth:
    """A Document paired with its evaluation labels.
    
    This is a composition - Document is the data, ground_truth is
    the test harness. Keeps Document clean and reusable.
    """
    document: Document
    ground_truth: dict[str, GroundTruth]
    
    @property
    def id(self) -> str:
        """Convenience: document ID."""
        return self.document.id
    
    @property
    def field_names(self) -> list[str]:
        """List of fields to extract."""
        return list(self.ground_truth.keys())
    
    def expected_value(self, field_name: str) -> Any:
        """Get expected value for a field."""
        gt = self.ground_truth.get(field_name)
        return gt.value if gt else None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "document": self.document.to_dict(),
            "ground_truth": {
                name: gt.to_dict() 
                for name, gt in self.ground_truth.items()
            },
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DocumentWithGroundTruth":
        """Deserialize from dictionary."""
        return cls(
            document=Document.from_dict(d["document"]),
            ground_truth={
                name: GroundTruth.from_dict(gt_dict)
                for name, gt_dict in d.get("ground_truth", {}).items()
            },
        )
