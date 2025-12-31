"""Document and dataset loaders.

Usage:
    from ctn_testing.core.loaders import load_document_set, load_docile

    # Standard format
    docs = load_document_set(Path("data/invoices"))

    # DocILE format
    docs = load_docile(Path("data/docile"), n=100)
"""

from .base import DatasetLoader, DocumentLoader, GroundTruthLoader
from .docile import DocILELoader, load_docile
from .text import (
    FileDocumentLoader,
    JsonDocumentLoader,
    StandardDatasetLoader,
    TextDocumentLoader,
    YamlGroundTruthLoader,
    load_document_set,
)

__all__ = [
    # Protocols
    "DocumentLoader",
    "DatasetLoader",
    "GroundTruthLoader",
    # Implementations
    "TextDocumentLoader",
    "JsonDocumentLoader",
    "FileDocumentLoader",
    "YamlGroundTruthLoader",
    "StandardDatasetLoader",
    "DocILELoader",
    # Convenience functions
    "load_document_set",
    "load_docile",
]
