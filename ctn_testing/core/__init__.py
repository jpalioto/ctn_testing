"""Core types and abstractions."""
from .types import (
    Extraction,
    GroundTruth,
    Evidence,
    Candidate,
    ExtractionStatus,
    Confidence,
    CompositeResult,
    FieldSchema,
    DocumentSchema,
)
from .models import (
    ModelConfig,
    ModelClient,
    CompletionResult,
    get_client,
    PROVIDERS,
)
from .config import (
    EvaluationConfig,
    KernelConfig,
    ComparisonConfig,
    DocumentConfig,
)
from .schemas import (
    parse_llm_response,
    ExtractionSchema,
    ExtractionsResponseSchema,
    GroundTruthFieldSchema,
    GroundTruthDocumentSchema,
    ground_truth_from_schema,
)

__all__ = [
    # Types
    "Extraction",
    "GroundTruth", 
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
]
