"""Evaluation runners."""

from ..core import Document, DocumentWithGroundTruth
from ..core.loaders import load_document_set
from .constraint_runner import (
    ConstraintConfig,
    ConstraintRunner,
    PromptConfig,
    RunResult,
    load_constraints,
    load_prompts,
)
from .http_runner import SDKError, SDKResponse, SDKRunner
from .kernel import Kernel, NullBaseline, TextKernel, load_kernel
from .output import PersistenceError
from .results import DocumentResult, FieldResult, RunResults
from .runner import RunConfig, Runner, run_evaluation

# Note: evaluation module imported lazily to avoid circular import with judging
# Use: from ctn_testing.runners.evaluation import ConstraintEvaluator
# Progress output colors: from ctn_testing.runners.evaluation import format_status, GREEN, RED, RESET

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
    # HTTP Runner (SDK integration)
    "SDKRunner",
    "SDKResponse",
    "SDKError",
    # Constraint Runner
    "ConstraintConfig",
    "PromptConfig",
    "RunResult",
    "ConstraintRunner",
    "load_prompts",
    "load_constraints",
    # Output management
    "PersistenceError",
    # Note: Evaluation module classes available via:
    # from ctn_testing.runners.evaluation import ConstraintEvaluator, ...
]
