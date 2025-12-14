"""Pydantic schemas for external data validation.

These models validate data at boundaries (API responses, config files, etc.)
then convert to internal dataclass types for processing.
"""
from typing import Any
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# LLM Response Parsing
# =============================================================================

class EvidenceSchema(BaseModel):
    """Evidence from LLM extraction response."""
    quote: str | None = None
    page: int | None = None


class CandidateSchema(BaseModel):
    """Candidate value from LLM response."""
    value: Any = None
    quote: str | None = None
    page: int | None = None


class ExtractionSchema(BaseModel):
    """Single field extraction from LLM response."""
    field: str
    value: Any = None
    evidence: EvidenceSchema = Field(default_factory=EvidenceSchema)
    status: str = "ok"
    confidence: str = "high"
    candidates: list[CandidateSchema] = Field(default_factory=list)
    
    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"ok", "missing", "ambiguous"}
        if v.lower() not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v}")
        return v.lower()
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        if v.lower() not in allowed:
            raise ValueError(f"confidence must be one of {allowed}, got {v}")
        return v.lower()


class ExtractionsResponseSchema(BaseModel):
    """Full LLM extraction response."""
    extractions: list[ExtractionSchema]


# =============================================================================
# Config Loading
# =============================================================================

class ModelConfigSchema(BaseModel):
    """Model configuration from YAML."""
    provider: str
    name: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning: bool = False
    max_reasoning_tokens: int = 0
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"anthropic", "google", "openai"}
        if v.lower() not in allowed:
            raise ValueError(f"provider must be one of {allowed}, got {v}")
        return v.lower()
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {v}")
        return v


class KernelConfigSchema(BaseModel):
    """Kernel configuration from YAML."""
    name: str
    path: str
    enabled: bool = True


class ComparisonConfigSchema(BaseModel):
    """Comparison pair configuration."""
    a: str
    b: str
    primary: bool = False


class ExecutionConfigSchema(BaseModel):
    """Execution settings."""
    parallel: bool = False
    max_workers: int = 4
    retry_count: int = 3
    retry_delay: float = 1.0
    delay: float = 0.0


class EvaluationConfigSchema(BaseModel):
    """Full evaluation configuration from YAML."""
    name: str
    description: str = ""
    models: list[ModelConfigSchema]
    kernels: list[KernelConfigSchema]
    comparisons: list[ComparisonConfigSchema] = Field(default_factory=list)
    judge_models: list[ModelConfigSchema] = Field(default_factory=list)
    judge_policy: str = "match_provider"
    execution: ExecutionConfigSchema = Field(default_factory=ExecutionConfigSchema)
    
    @field_validator("judge_policy")
    @classmethod
    def validate_judge_policy(cls, v: str) -> str:
        allowed = {"single", "match_provider", "full_cross"}
        if v.lower() not in allowed:
            raise ValueError(f"judge_policy must be one of {allowed}, got {v}")
        return v.lower()


# =============================================================================
# Ground Truth Loading
# =============================================================================

class GroundTruthFieldSchema(BaseModel):
    """Single field ground truth from YAML."""
    exists: bool = True
    ambiguous: bool = False
    value: Any = None
    acceptable_values: list[Any] = Field(default_factory=list)
    candidate_values: list[Any] = Field(default_factory=list)
    quote: str | None = None
    page: int | None = None


class GroundTruthDocumentSchema(BaseModel):
    """Document ground truth from YAML."""
    fields: dict[str, GroundTruthFieldSchema]


# =============================================================================
# Converters to Internal Types
# =============================================================================

def extraction_from_schema(schema: ExtractionSchema):
    """Convert validated schema to internal Extraction type."""
    from .types import Extraction, Evidence, Candidate, ExtractionStatus, Confidence
    
    return Extraction(
        field_name=schema.field,
        value=schema.value,
        evidence=Evidence(
            quote=schema.evidence.quote,
            page=schema.evidence.page,
        ),
        status=ExtractionStatus(schema.status),
        confidence=Confidence(schema.confidence),
        candidates=[
            Candidate(value=c.value, quote=c.quote, page=c.page)
            for c in schema.candidates
        ],
    )


def ground_truth_from_schema(
    field_name: str, 
    schema: GroundTruthFieldSchema
):
    """Convert validated schema to internal GroundTruth type."""
    from .types import GroundTruth
    
    return GroundTruth(
        field_name=field_name,
        exists_in_document=schema.exists,
        is_ambiguous=schema.ambiguous,
        value=schema.value,
        acceptable_values=schema.acceptable_values,
        candidate_values=schema.candidate_values,
        evidence_quote=schema.quote,
        evidence_page=schema.page,
    )


def parse_llm_response(response_text: str):
    """Parse and validate LLM JSON response into Extraction objects.
    
    Handles markdown code fences and validates structure.
    Raises: pydantic.ValidationError if response doesn't match schema.
    """
    import json
    
    text = response_text.strip()
    
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[1:end_idx])
    
    data = json.loads(text)
    validated = ExtractionsResponseSchema.model_validate(data)
    
    return [extraction_from_schema(e) for e in validated.extractions]
