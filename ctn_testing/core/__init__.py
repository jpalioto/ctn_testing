"""Core types and abstractions."""

from .config import (
    ComparisonConfig,
    DocumentConfig,
    EvaluationConfig,
    KernelConfig,
)
from .document import SUPPORTED_MEDIA_TYPES, Document
from .ground_truth import DocumentWithGroundTruth, GroundTruth
from .models import (
    PROVIDERS,
    CompletionResult,
    ModelClient,
    ModelConfig,
    get_client,
)
from .schemas import (
    ExtractionSchema,
    ExtractionsResponseSchema,
    GroundTruthDocumentSchema,
    GroundTruthFieldSchema,
    ground_truth_from_schema,
    parse_llm_response,
)
from .types import (
    Candidate,
    CompositeResult,
    Confidence,
    DocumentSchema,
    Evidence,
    Extraction,
    ExtractionStatus,
    FieldSchema,
)

__all__ = [
    # Types
    "Extraction",
    "Evidence",
    "Candidate",
    "ExtractionStatus",
    "Confidence",
    "CompositeResult",
    "FieldSchema",
    "DocumentSchema",
    # Models
    "ModelConfig",
    "ModelClient",
    "CompletionResult",
    "get_client",
    "PROVIDERS",
    # Config
    "EvaluationConfig",
    "KernelConfig",
    "ComparisonConfig",
    "DocumentConfig",
    # Schema
    "parse_llm_response",
    "ExtractionSchema",
    "ExtractionsResponseSchema",
    "GroundTruthFieldSchema",
    "GroundTruthDocumentSchema",
    "ground_truth_from_schema",
    # Documents
    "Document",
    "GroundTruth",
    "DocumentWithGroundTruth",
    "SUPPORTED_MEDIA_TYPES",
]
