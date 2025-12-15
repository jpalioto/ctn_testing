"""Core types and abstractions."""
from .types import (
    Extraction,
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
from .document import (
    Document, 
    SUPPORTED_MEDIA_TYPES
)
from .ground_truth import(
    GroundTruth, 
    DocumentWithGroundTruth
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
    "SUPPORTED_MEDIA_TYPES"
]
