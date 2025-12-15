"""DocILE dataset loader.

DocILE is a large-scale document information extraction benchmark
with native PDF documents and fine-grained annotations.

Dataset structure:
    docile/
        pdfs/
            {doc_id}.pdf
        ocr/
            {doc_id}.json       # Optional OCR output
        annotations/
            {doc_id}.json       # Field-level annotations

Reference: https://github.com/rossumai/docile
"""
import json
from pathlib import Path
from typing import Iterator

from ..document import Document
from ..ground_truth import DocumentWithGroundTruth, GroundTruth
from .base import DatasetLoader


class DocILELoader(DatasetLoader):
    """Loader for DocILE dataset format.
    
    DocILE provides:
    - Native PDF documents (no synthetic generation)
    - 55 field types for invoices
    - Line item annotations
    - Bounding box coordinates
    
    We extract field-level annotations and ignore spatial data
    (our models use full document understanding, not layout).
    """
    
    # DocILE field types we care about (subset)
    CORE_FIELDS = {
        "invoice_id",
        "invoice_date",
        "due_date",
        "vendor_name",
        "vendor_address",
        "customer_name", 
        "customer_address",
        "subtotal",
        "total_tax",
        "total_amount",
        "currency",
        "payment_terms",
        "purchase_order",
    }
    
    def __init__(self, include_all_fields: bool = False):
        """Initialize loader.
        
        Args:
            include_all_fields: If True, include all 55 DocILE fields.
                               If False, only include CORE_FIELDS.
        """
        self.include_all_fields = include_all_fields
    
    def iter_documents(self, dataset_path: Path) -> Iterator[DocumentWithGroundTruth]:
        pdfs_dir = dataset_path / "pdfs"
        annotations_dir = dataset_path / "annotations"
        ocr_dir = dataset_path / "ocr"
        
        if not pdfs_dir.exists():
            raise FileNotFoundError(f"DocILE pdfs directory not found: {pdfs_dir}")
        
        for pdf_path in sorted(pdfs_dir.glob("*.pdf")):
            doc_id = pdf_path.stem
            ann_path = annotations_dir / f"{doc_id}.json"
            
            if not ann_path.exists():
                print(f"Warning: No annotations for {doc_id}, skipping")
                continue
            
            # Load PDF as document
            document = Document(
                id=doc_id,
                file_path=pdf_path,
                doc_type="invoice",  # DocILE is invoice-focused
                source="docile",
            )
            
            # Optionally load OCR text as fallback
            ocr_path = ocr_dir / f"{doc_id}.json"
            if ocr_path.exists():
                document.text = self._load_ocr_text(ocr_path)
            
            # Load and convert annotations
            with open(ann_path, encoding="utf-8") as f:
                annotations = json.load(f)
            
            ground_truth = self._convert_annotations(annotations)
            
            if not ground_truth:
                print(f"Warning: No usable fields for {doc_id}, skipping")
                continue
            
            yield DocumentWithGroundTruth(
                document=document,
                ground_truth=ground_truth,
            )
    
    def load_all(self, dataset_path: Path) -> list[DocumentWithGroundTruth]:
        return list(self.iter_documents(dataset_path))
    
    def _load_ocr_text(self, ocr_path: Path) -> str:
        """Extract text from DocILE OCR JSON."""
        with open(ocr_path, encoding="utf-8") as f:
            ocr_data = json.load(f)
        
        # DocILE OCR format has pages with words
        text_parts = []
        for page in ocr_data.get("pages", []):
            for word in page.get("words", []):
                text_parts.append(word.get("text", ""))
        
        return " ".join(text_parts)
    
    def _convert_annotations(self, annotations: dict) -> dict[str, GroundTruth]:
        """Convert DocILE annotation format to our GroundTruth format.
        
        DocILE structure:
        {
            "field_annotations": [
                {
                    "fieldtype": "invoice_id",
                    "text": "INV-001",
                    "page": 0,
                    "bbox": [x1, y1, x2, y2]
                },
                ...
            ],
            "line_item_annotations": [...]  # We skip these for now
        }
        """
        ground_truth = {}
        
        for field_info in annotations.get("field_annotations", []):
            field_type = field_info.get("fieldtype")
            text = field_info.get("text", "").strip()
            page = field_info.get("page", 0)
            
            # Skip if no field type or text
            if not field_type or not text:
                continue
            
            # Filter to core fields unless include_all_fields
            if not self.include_all_fields and field_type not in self.CORE_FIELDS:
                continue
            
            # Handle duplicate field types (take first occurrence)
            if field_type in ground_truth:
                # Add to acceptable_values instead
                existing = ground_truth[field_type]
                if text not in existing.acceptable_values and text != existing.value:
                    existing.acceptable_values.append(text)
                continue
            
            ground_truth[field_type] = GroundTruth(
                field_name=field_type,
                value=text,
                evidence_page=page,
            )
        
        return ground_truth


def load_docile(dataset_path: Path, n: int | None = None) -> list[DocumentWithGroundTruth]:
    """Convenience function to load DocILE dataset.
    
    Args:
        dataset_path: Path to DocILE dataset root
        n: Optional limit on number of documents
        
    Returns:
        List of documents with ground truth
    """
    loader = DocILELoader()
    if n is not None:
        return loader.load_sample(dataset_path, n)
    return loader.load_all(dataset_path)
