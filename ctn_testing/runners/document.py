"""Document and ground truth loading."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

from ..core.types import GroundTruth
from ..core.schemas import GroundTruthDocumentSchema, ground_truth_from_schema


@dataclass
class Document:
    """
    A document to evaluate.
    
    id: Unique identifier (e.g., "invoice_001")
    text: Full document text
    pages: Text split by page (for page number validation)
    doc_type: Document type (e.g., "invoice", "contract")
    difficulty: easy | medium | hard | adversarial
    metadata: Additional info (source, notes, etc.)
    """
    id: str
    text: str
    pages: list[str]
    doc_type: str = "unknown"
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: Path) -> "Document":
        """Load document from .txt or .json file."""
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            return cls(
                id=data.get("id", path.stem),
                text=data["text"],
                pages=data.get("pages", [data["text"]]),
                doc_type=data.get("doc_type", "unknown"),
                difficulty=data.get("difficulty", "medium"),
                metadata=data.get("metadata", {}),
            )
        else:
            text = path.read_text(encoding="utf-8")
            return cls(
                id=path.stem,
                text=text,
                pages=[text],
            )


@dataclass
class DocumentWithGroundTruth:
    """Document paired with its ground truth."""
    document: Document
    ground_truth: dict[str, GroundTruth]
    
    @classmethod
    def from_files(cls, doc_path: Path, gt_path: Path) -> "DocumentWithGroundTruth":
        """Load document and its ground truth."""
        document = Document.from_file(doc_path)
        
        with open(gt_path) as f:
            if gt_path.suffix == ".yaml":
                gt_data = yaml.safe_load(f)
            else:
                gt_data = json.load(f)
        
        gt_validated = GroundTruthDocumentSchema.model_validate(gt_data)
        ground_truth = {
            name: ground_truth_from_schema(name, field_schema)
            for name, field_schema in gt_validated.fields.items()
        }
        
        for field_name, field_data in gt_data.get("fields", {}).items():
            ground_truth[field_name] = GroundTruth(
                field_name=field_name,
                exists_in_document=field_data.get("exists", True),
                is_ambiguous=field_data.get("ambiguous", False),
                value=field_data.get("value"),
                acceptable_values=field_data.get("acceptable_values", []),
                candidate_values=field_data.get("candidate_values", []),
                evidence_quote=field_data.get("quote"),
                evidence_page=field_data.get("page"),
            )
        
        return cls(document=document, ground_truth=ground_truth)


def load_document_set(data_dir: Path) -> list[DocumentWithGroundTruth]:
    """
    Load all documents from a directory.
    
    Expected structure:
        data_dir/
            documents/
                doc_001.json
                doc_002.json
            ground_truth/
                doc_001.yaml
                doc_002.yaml
    """
    docs_dir = data_dir / "documents"
    gt_dir = data_dir / "ground_truth"
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    results = []
    for doc_path in sorted(docs_dir.glob("*")):
        if doc_path.suffix not in (".json", ".txt"):
            continue
        
        gt_path = gt_dir / f"{doc_path.stem}.yaml"
        if not gt_path.exists():
            gt_path = gt_dir / f"{doc_path.stem}.json"
        
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found for {doc_path.name}")
        
        results.append(DocumentWithGroundTruth.from_files(doc_path, gt_path))
    
    return results
