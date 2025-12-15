"""Document model with multimodal support.

Supports text-only documents (legacy) and file-based documents (PDF, images).
"""
import base64
from dataclasses import dataclass, field
from pathlib import Path


# Supported file types for native multimodal
SUPPORTED_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class Document:
    """A document for extraction.
    
    Supports two modes:
    1. Text mode: document content provided as text string
    2. File mode: document content as PDF or image file
    
    For file mode, models receive the raw file bytes (base64 encoded)
    allowing them to use native vision/document understanding.
    """
    id: str
    
    text: str | None = None
    pages: list[str] | None = None 
    
    file_path: Path | None = None

    doc_type: str = "unknown"
    difficulty: str = "medium"
    source: str | None = None  #
    metadata: dict = field(default_factory=dict) 
    
    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
    
    @property
    def has_file(self) -> bool:
        """True if this document has a file (PDF/image)."""
        return self.file_path is not None and self.file_path.exists()
    
    @property
    def has_content(self) -> bool:
        """True if document has any content (text or file)."""
        return self.has_text or self.has_file
    
    @property
    def has_text(self) -> bool:
        """True if this document has text content."""
        return self.text is not None
    
    @property
    def media_type(self) -> str | None:
        """Get MIME type for file, or None if no file."""
        if not self.file_path:
            return None
        suffix = self.file_path.suffix.lower()
        return SUPPORTED_MEDIA_TYPES.get(suffix)
    
    @property
    def is_pdf(self) -> bool:
        """True if document is a PDF."""
        return self.media_type == "application/pdf"
    
    @property
    def is_image(self) -> bool:
        """True if document is an image."""
        return self.media_type is not None and self.media_type.startswith("image/")
    
    def file_bytes(self) -> bytes:
        """Read file as bytes. Raises if no file."""
        if not self.file_path:
            raise ValueError(f"Document {self.id} has no file path")
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document file not found: {self.file_path}")
        return self.file_path.read_bytes()
    
    def file_base64(self) -> str:
        """Read file as base64 string. Raises if no file."""
        return base64.b64encode(self.file_bytes()).decode("ascii")
    
    def content_for_prompt(self) -> str:
        """Get text content for embedding in prompts (legacy mode)."""
        if self.text:
            return self.text
        raise ValueError(f"Document {self.id} has no text content. Use file mode.")
    
    @classmethod
    def from_text(cls, id: str, text: str, pages: list[str] | None = None) -> "Document":
        """Create a text-mode document."""
        return cls(id=id, text=text, pages=pages)
    
    @classmethod
    def from_file(cls, id: str, file_path: Path | str, text: str | None = None) -> "Document":
        """Create a file-mode document.
        
        Args:
            id: Document identifier
            file_path: Path to PDF or image file
            text: Optional extracted text (for hybrid mode or fallback)
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")
        
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(SUPPORTED_MEDIA_TYPES.keys())}")
        
        return cls(id=id, file_path=path, text=text)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "pages": self.pages,
            "file_path": str(self.file_path) if self.file_path else None,
            "doc_type": self.doc_type,
            "difficulty": self.difficulty,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            id=d["id"],
            text=d.get("text"),
            pages=d.get("pages"),
            file_path=Path(d["file_path"]) if d.get("file_path") else None,
            doc_type=d.get("doc_type", "unknown"),
            difficulty=d.get("difficulty", "medium"),
            source=d.get("source"),
            metadata=d.get("metadata", {}),
        )
