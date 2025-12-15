"""Evaluation runners."""
from ..core import Document, DocumentWithGroundTruth
from ..core.loaders import load_document_set
from .kernel import Kernel, TextKernel, NullBaseline, load_kernel
from .results import FieldResult, DocumentResult, RunResults
from .runner import Runner, RunConfig, run_evaluation

__all__ = [
    # Document (re-exported from core)
    "Document",
    "DocumentWithGroundTruth",
    "load_document_set",
    # Kernel
    "Kernel",
    "TextKernel",
    "NullBaseline",
    "load_kernel",
    # Results
    "FieldResult",
    "DocumentResult",
    "RunResults",
    # Runner
    "Runner",
    "RunConfig",
    "run_evaluation",
]