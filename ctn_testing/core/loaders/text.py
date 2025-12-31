"""Loaders for text-based documents (.txt, .json) and YAML ground truth.

This is the default loader for our custom evaluation datasets.
"""

import json
from pathlib import Path
from typing import Iterator

import yaml

from ..document import SUPPORTED_MEDIA_TYPES, Document
from ..ground_truth import DocumentWithGroundTruth, GroundTruth
from .base import DatasetLoader, DocumentLoader, GroundTruthLoader


class TextDocumentLoader(DocumentLoader):
    """Loader for .txt files."""

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = path.read_text(encoding="utf-8")
        return Document(
            id=path.stem,
            text=text,
            pages=[text],  # Single page for plain text
        )

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"


class JsonDocumentLoader(DocumentLoader):
    """Loader for .json document files.

    Expected format:
    {
        "id": "doc_001",           # Optional, defaults to filename
        "text": "...",             # Document text (required if no file_path)
        "pages": ["...", "..."],   # Optional per-page text
        "file_path": "path/to.pdf", # Optional file reference
        "doc_type": "invoice",     # Optional
        "difficulty": "medium",    # Optional
        "metadata": {}             # Optional
    }
    """

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        file_path = None
        if "file_path" in data:
            # Resolve relative to JSON file location
            fp = Path(data["file_path"])
            if not fp.is_absolute():
                fp = path.parent / fp
            file_path = fp

        return Document(
            id=data.get("id", path.stem),
            text=data.get("text"),
            pages=data.get("pages"),
            file_path=file_path,
            doc_type=data.get("doc_type", "unknown"),
            difficulty=data.get("difficulty", "medium"),
            metadata=data.get("metadata", {}),
        )

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"


class FileDocumentLoader(DocumentLoader):
    """Loader for PDF and image files."""

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"Unsupported file type: {suffix}")

        return Document(
            id=path.stem,
            file_path=path,
        )

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_MEDIA_TYPES


class YamlGroundTruthLoader(GroundTruthLoader):
    """Loader for YAML ground truth files.

    Expected format:
    ```yaml
    fields:
      invoice_number:
        value: "INV-001"
        acceptable_values: ["INV001", "INV-001"]
        exists: true
        ambiguous: false
        quote: "Invoice Number: INV-001"
        page: 0

      # Simple format also supported:
      vendor_name: "Acme Corp"
    ```
    """

    def load(self, path: Path) -> dict[str, GroundTruth]:
        if not path.exists():
            raise FileNotFoundError(f"Ground truth not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Handle both wrapped and unwrapped formats
        fields = data.get("fields", data)

        ground_truth = {}
        for field_name, field_data in fields.items():
            if field_name == "fields":
                continue  # Skip if we got the wrapper

            if isinstance(field_data, dict):
                ground_truth[field_name] = GroundTruth(
                    field_name=field_name,
                    value=field_data.get("value"),
                    acceptable_values=field_data.get("acceptable_values", []),
                    candidate_values=field_data.get("candidate_values", []),
                    exists_in_document=field_data.get("exists", True),
                    is_ambiguous=field_data.get("ambiguous", False),
                    evidence_quote=field_data.get("quote"),
                    evidence_page=field_data.get("page"),
                    notes=field_data.get("notes"),
                )
            else:
                # Simple format: field_name: value
                ground_truth[field_name] = GroundTruth(
                    field_name=field_name,
                    value=field_data,
                )

        return ground_truth

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".yaml", ".yml")


class StandardDatasetLoader(DatasetLoader):
    """Loader for our standard dataset directory structure.

    Expected structure:
        dataset_dir/
            documents/
                doc_001.txt (or .json, .pdf, .png, etc.)
                doc_002.json
            ground_truth/
                doc_001.yaml
                doc_002.yaml
    """

    def __init__(self):
        self._doc_loaders = [
            JsonDocumentLoader(),
            TextDocumentLoader(),
            FileDocumentLoader(),
        ]
        self._gt_loader = YamlGroundTruthLoader()

    def _load_document(self, path: Path) -> Document:
        """Load document using appropriate loader."""
        for loader in self._doc_loaders:
            if loader.supports(path):
                return loader.load(path)
        raise ValueError(f"No loader for document: {path}")

    def _find_ground_truth(self, doc_id: str, gt_dir: Path) -> Path | None:
        """Find ground truth file for document."""
        for ext in (".yaml", ".yml", ".json"):
            gt_path = gt_dir / f"{doc_id}{ext}"
            if gt_path.exists():
                return gt_path
        return None

    def iter_documents(self, dataset_path: Path) -> Iterator[DocumentWithGroundTruth]:
        docs_dir = dataset_path / "documents"
        gt_dir = dataset_path / "ground_truth"

        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        # Find all document files
        all_files = sorted(docs_dir.iterdir())

        for doc_path in all_files:
            if doc_path.name.startswith("."):
                continue  # Skip hidden files

            # Check if any loader supports this file
            if not any(loader.supports(doc_path) for loader in self._doc_loaders):
                continue

            # Find ground truth
            gt_path = self._find_ground_truth(doc_path.stem, gt_dir)
            if not gt_path:
                print(f"Warning: No ground truth for {doc_path.name}, skipping")
                continue

            document = self._load_document(doc_path)
            ground_truth = self._gt_loader.load(gt_path)

            yield DocumentWithGroundTruth(
                document=document,
                ground_truth=ground_truth,
            )

    def load_all(self, dataset_path: Path) -> list[DocumentWithGroundTruth]:
        return list(self.iter_documents(dataset_path))


# Convenience function matching original API
def load_document_set(data_dir: Path) -> list[DocumentWithGroundTruth]:
    """Load all documents from a standard dataset directory."""
    loader = StandardDatasetLoader()
    return loader.load_all(data_dir)
