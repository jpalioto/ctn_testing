"""Document and dataset loaders.

Usage:
    from ctn_testing.core.loaders import load_document_set, load_docile
    
    # Standard format
    docs = load_document_set(Path("data/invoices"))
    
    # DocILE format
    docs = load_docile(Path("data/docile"), n=100)
"""
from .base import DocumentLoader, DatasetLoader, GroundTruthLoader
from .text import (
    TextDocumentLoader,
    JsonDocumentLoader,
    FileDocumentLoader,
    YamlGroundTruthLoader,
    StandardDatasetLoader,
    load_document_set,
)
from .docile import DocILELoader, load_docile

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
