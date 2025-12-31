"""DocILE dataset loader using official docile-benchmark library.

DocILE is a large-scale document information extraction benchmark
with native PDF documents and fine-grained annotations.

Requires: pip install docile-benchmark (or uv add docile-benchmark)

Dataset structure (after download):
    docile/
        pdfs/
            {doc_id}.pdf
        ocr/
            {doc_id}.json
        annotations/
            {doc_id}.json

Reference: https://github.com/rossumai/docile
Paper: https://arxiv.org/abs/2302.05658
"""

from pathlib import Path
from typing import Iterator

from ..document import Document
from ..ground_truth import DocumentWithGroundTruth, GroundTruth
from .base import DatasetLoader

# Optional dependency - graceful degradation
try:
    from docile.dataset import Dataset as DocileDataset

    HAS_DOCILE = True
except ImportError:
    DocileDataset = None  # type: ignore
    HAS_DOCILE = False


# Field types available in DocILE (KILE task)
# https://docile.rossum.ai/
DOCILE_FIELD_TYPES = {
    # Document identifiers
    "document_id",
    "order_id",
    # Dates
    "date_issue",
    "date_due",
    # Vendor info
    "vendor_name",
    "vendor_address",
    "vendor_tax_id",
    "vendor_email",
    "vendor_phone",
    # Customer info
    "customer_billing_name",
    "customer_billing_address",
    "customer_shipping_name",
    "customer_shipping_address",
    "customer_tax_id",
    "customer_email",
    "customer_phone",
    # Amounts
    "amount_total_gross",
    "amount_total_net",
    "amount_total_tax",
    "amount_due",
    "amount_paid",
    "amount_rounding",
    # Currency
    "currency_code_amount_due",
    # Payment
    "payment_terms",
    "payment_reference",
    "bank_account_number",
    "bank_iban",
    "bank_swift",
    # Line item fields (for LIR task - we skip these by default)
    "line_item_description",
    "line_item_quantity",
    "line_item_unit_price",
    "line_item_amount_gross",
    "line_item_amount_net",
    "line_item_tax_rate",
    "line_item_tax_amount",
    "line_item_order_id",
    "line_item_product_code",
}


class DocILELoader(DatasetLoader):
    """Loader for DocILE dataset using official library.

    Usage:
        loader = DocILELoader(split="val")
        docs = loader.load_all(Path("C:/path/to/docile"))

        # Or load a sample
        docs = loader.load_sample(Path("C:/path/to/docile"), n=10)

        # Or iterate (memory efficient)
        for doc in loader.iter_documents(Path("C:/path/to/docile")):
            process(doc)
    """

    def __init__(
        self,
        split: str = "val",
        include_line_items: bool = False,
        load_ocr: bool = False,
    ):
        """Initialize DocILE loader.

        Args:
            split: Dataset split - "train", "val", "test", or "trainval"
            include_line_items: If True, include line item fields (LIR task).
                               If False, only include document-level fields (KILE task).
            load_ocr: If True, load OCR data into memory (slower but enables text access)
        """
        if not HAS_DOCILE:
            raise ImportError(
                "docile-benchmark not installed. Install with: uv add docile-benchmark"
            )

        self.split = split
        self.include_line_items = include_line_items
        self.load_ocr = load_ocr

    def _get_pdf_path(self, dataset_path: Path, doc_id: str) -> Path | None:
        """Find PDF file for document."""
        # Try standard location
        pdf_path = dataset_path / "pdfs" / f"{doc_id}.pdf"
        if pdf_path.exists():
            return pdf_path

        # Try annotated subdirectory structure
        pdf_path = dataset_path / "annotated-trainval" / "pdfs" / f"{doc_id}.pdf"
        if pdf_path.exists():
            return pdf_path

        return None

    def _extract_ocr_text(self, doc) -> str | None:
        """Extract full text from OCR if available."""
        if not self.load_ocr:
            return None

        try:
            text_parts = []
            for page_idx in range(doc.page_count):
                words = doc.ocr.get_all_words(page_idx)
                page_text = " ".join(w.text for w in words)
                text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            return None

    def _is_line_item_field(self, fieldtype: str) -> bool:
        """Check if field is a line item field."""
        return fieldtype.startswith("line_item_")

    def iter_documents(self, dataset_path: Path) -> Iterator[DocumentWithGroundTruth]:
        """Iterate over DocILE documents.

        Args:
            dataset_path: Path to DocILE dataset root

        Yields:
            DocumentWithGroundTruth for each document
        """
        dataset = DocileDataset(  # type: ignore[misc]
            self.split,
            str(dataset_path),
            load_ocr=self.load_ocr,
        )

        for doc in dataset:
            # Find PDF path
            pdf_path = self._get_pdf_path(dataset_path, doc.docid)

            # Extract OCR text if enabled
            ocr_text = self._extract_ocr_text(doc) if self.load_ocr else None

            # Build Document
            document = Document(
                id=doc.docid,
                text=ocr_text,
                file_path=pdf_path,
                doc_type="invoice",
                source="docile",
                metadata={
                    "pages": doc.page_count,
                    "split": self.split,
                },
            )

            # Build GroundTruth from annotations
            ground_truth: dict[str, GroundTruth] = {}

            for field in doc.annotation.fields:
                fieldtype = field.fieldtype

                # Skip fields without fieldtype
                if not fieldtype:
                    continue

                # Skip line items unless requested
                if self._is_line_item_field(fieldtype) and not self.include_line_items:
                    continue

                # Handle duplicate field types - keep first, add others to acceptable_values
                if fieldtype in ground_truth:
                    existing = ground_truth[fieldtype]
                    if field.text != existing.value:
                        existing.acceptable_values.append(field.text)
                    continue

                ground_truth[fieldtype] = GroundTruth(
                    field_name=fieldtype,
                    value=field.text,
                    evidence_page=field.page,
                    evidence_quote=field.text,
                )

            # Skip documents with no usable fields
            if not ground_truth:
                continue

            yield DocumentWithGroundTruth(
                document=document,
                ground_truth=ground_truth,
            )

    def load_all(self, dataset_path: Path) -> list[DocumentWithGroundTruth]:
        """Load all documents from dataset."""
        return list(self.iter_documents(dataset_path))


def load_docile(
    dataset_path: Path | str,
    split: str = "val",
    n: int | None = None,
    include_line_items: bool = False,
    load_ocr: bool = False,
) -> list[DocumentWithGroundTruth]:
    """Convenience function to load DocILE dataset.

    Args:
        dataset_path: Path to DocILE dataset root
        split: Dataset split - "train", "val", "test", or "trainval"
        n: Optional limit on number of documents
        include_line_items: If True, include line item fields
        load_ocr: If True, load OCR text (slower)

    Returns:
        List of documents with ground truth

    Example:
        # Load 10 validation documents
        docs = load_docile("C:/data/docile", split="val", n=10)

        # Load all training documents with line items
        docs = load_docile("C:/data/docile", split="train", include_line_items=True)
    """
    path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
    loader = DocILELoader(
        split=split,
        include_line_items=include_line_items,
        load_ocr=load_ocr,
    )

    if n is not None:
        return loader.load_sample(path, n)
    return loader.load_all(path)
