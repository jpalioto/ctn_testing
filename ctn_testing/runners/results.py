"""Result storage and aggregation."""
from dataclasses import dataclass, field
from ..statistics.comparison import paired_comparison, ComparisonResult
from datetime import datetime
from pathlib import Path
from typing import Any
import json


@dataclass
class FieldResult:
    """Result for a single field extraction."""
    field_name: str
    extracted_value: Any
    expected_value: Any
    quote: str | None
    page: int | None
    status: str
    
    # Scores
    value_score: float
    evidence_score: float
    page_score: float
    status_score: float
    schema_score: float
    composite_score: float
    
    # Judge results (if applicable)
    judge_scores: dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: dict) -> "FieldResult":
        """Reconstruct FieldResult from dict."""
        scores = d.get("scores", {})
        return cls(
            field_name=d["field"],
            extracted_value=d["extracted"],
            expected_value=d["expected"],
            quote=d["quote"],
            page=d["page"],
            status=d["status"],
            value_score=scores.get("value", 0.0),
            evidence_score=scores.get("evidence", 0.0),
            page_score=scores.get("page", 0.0),
            status_score=scores.get("status", 0.0),
            schema_score=scores.get("schema", 0.0),
            composite_score=scores.get("composite", 0.0),
            judge_scores=d.get("judge_scores", {}),
        )


@dataclass
class DocumentResult:
    """Result for a single document."""
    # Identity
    doc_id: str
    model: str
    kernel: str
    fields: list[FieldResult]
    
    # Raw outputs
    raw_response: str | None = None
    judge_raw_response: str | None = None
    judge_outcome: str | None = None
    
    # Timing
    timestamp: datetime | None = None
    latency_ms: float = 0.0
    
    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Errors
    error: str | None = None
    cache_prefix: str | None = None
    
    # Reproducibility - hashes (verification)
    document_hash: str | None = None
    kernel_hash: str | None = None
    gt_hash: str | None = None
    judge_prompt_hash: str | None = None
    model_config_hash: str | None = None
    judge_config_hash: str | None = None
    
    # Reproducibility - explicit params (readability)
    model_temperature: float | None = None
    model_max_tokens: int | None = None
    judge_model: str | None = None
    judge_temperature: float | None = None
    judge_max_tokens: int | None = None
    
    # Debug
    api_request_id: str | None = None
    
    @property
    def composite_score(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.composite_score for f in self.fields) / len(self.fields)
    
    @property
    def value_score(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.value_score for f in self.fields) / len(self.fields)
    
    def to_dict(self) -> dict:
        return {
            # Identity
            "doc_id": self.doc_id,
            "model": self.model,
            "kernel": self.kernel,
            
            # Computed scores
            "composite_score": self.composite_score,
            "value_score": self.value_score,
            
            # Timing
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latency_ms": self.latency_ms,
            
            # Tokens
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            
            # Errors
            "cache_prefix": self.cache_prefix,
            "error": self.error,
            
            # Raw outputs
            "raw_response": self.raw_response,
            "judge_raw_response": self.judge_raw_response,
            "judge_outcome": self.judge_outcome,
            
            # Reproducibility - hashes
            "document_hash": self.document_hash,
            "kernel_hash": self.kernel_hash,
            "gt_hash": self.gt_hash,
            "judge_prompt_hash": self.judge_prompt_hash,
            "model_config_hash": self.model_config_hash,
            "judge_config_hash": self.judge_config_hash,
            
            # Reproducibility - explicit params
            "model_temperature": self.model_temperature,
            "model_max_tokens": self.model_max_tokens,
            "judge_model": self.judge_model,
            "judge_temperature": self.judge_temperature,
            "judge_max_tokens": self.judge_max_tokens,
            
            # Debug
            "api_request_id": self.api_request_id,
            
            # Fields
            "fields": [
                {
                    "field": f.field_name,
                    "extracted": f.extracted_value,
                    "expected": f.expected_value,
                    "quote": f.quote,
                    "page": f.page,
                    "status": f.status,
                    "scores": {
                        "value": f.value_score,
                        "evidence": f.evidence_score,
                        "page": f.page_score,
                        "status": f.status_score,
                        "schema": f.schema_score,
                        "composite": f.composite_score,
                    },
                    "judge_scores": f.judge_scores,
                }
                for f in self.fields
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DocumentResult":
        """Reconstruct DocumentResult from dict."""
        # Parse timestamp
        timestamp = None
        if d.get("timestamp"):
            timestamp = datetime.fromisoformat(d["timestamp"])
        
        # Parse fields
        fields = [FieldResult.from_dict(f) for f in d.get("fields", [])]
        
        return cls(
            # Identity
            doc_id=d["doc_id"],
            model=d["model"],
            kernel=d["kernel"],
            fields=fields,
            
            # Raw outputs
            raw_response=d.get("raw_response"),
            judge_raw_response=d.get("judge_raw_response"),
            judge_outcome=d.get("judge_outcome"),
            
            # Timing
            timestamp=timestamp,
            latency_ms=d.get("latency_ms", 0.0),
            
            # Tokens
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            
            # Errors
            error=d.get("error"),
            cache_prefix=d.get("cache_prefix"),
            
            # Reproducibility - hashes
            document_hash=d.get("document_hash"),
            kernel_hash=d.get("kernel_hash"),
            gt_hash=d.get("gt_hash"),
            judge_prompt_hash=d.get("judge_prompt_hash"),
            model_config_hash=d.get("model_config_hash"),
            judge_config_hash=d.get("judge_config_hash"),
            
            # Reproducibility - explicit params
            model_temperature=d.get("model_temperature"),
            model_max_tokens=d.get("model_max_tokens"),
            judge_model=d.get("judge_model"),
            judge_temperature=d.get("judge_temperature"),
            judge_max_tokens=d.get("judge_max_tokens"),
            
            # Debug
            api_request_id=d.get("api_request_id"),
        )


@dataclass
class RunResults:
    """Results for an entire evaluation run."""
    run_id: str
    config_name: str
    started_at: datetime
    completed_at: datetime | None = None
    
    # Results indexed by (model, kernel, doc_id)
    results: dict[tuple[str, str, str], DocumentResult] = field(default_factory=dict)
    
    def add(self, result: DocumentResult):
        key = (result.model, result.kernel, result.doc_id)
        self.results[key] = result
    
    def get(self, model: str, kernel: str, doc_id: str) -> DocumentResult | None:
        return self.results.get((model, kernel, doc_id))
    
    def by_model(self, model: str) -> list[DocumentResult]:
        return [r for r in self.results.values() if r.model == model]
    
    def by_kernel(self, kernel: str) -> list[DocumentResult]:
        return [r for r in self.results.values() if r.kernel == kernel]
    
    def by_model_kernel(self, model: str, kernel: str) -> list[DocumentResult]:
        return [r for r in self.results.values() if r.model == model and r.kernel == kernel]
    
    def composite_scores(self, model: str, kernel: str) -> list[float]:
        """Get composite scores for a model/kernel pair, ordered by doc_id."""
        results = sorted(self.by_model_kernel(model, kernel), key=lambda r: r.doc_id)
        return [r.composite_score for r in results]
    
    def compare(self, model: str, kernel_a: str, kernel_b: str) -> ComparisonResult | None:
        """Compare two kernels for a given model using paired t-test."""
        scores_a = self.composite_scores(model, kernel_a)
        scores_b = self.composite_scores(model, kernel_b)
    
        if len(scores_a) < 2 or len(scores_a) != len(scores_b):
            return None
    
        return paired_comparison(scores_a, scores_b)
    
    def summary(self) -> dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        models = set(r.model for r in self.results.values())
        kernels = set(r.kernel for r in self.results.values())
        
        summary = {
            "run_id": self.run_id,
            "config": self.config_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_documents": len(set(r.doc_id for r in self.results.values())),
            "models": list(models),
            "kernels": list(kernels),
            "by_model_kernel": {},
        
        }
        summary["comparisons"] = {}
        for model in models:
            for kernel in kernels:
                results = self.by_model_kernel(model, kernel)
                if not results:
                    continue
                
                key = f"{model}|{kernel}"
                scores = [r.composite_score for r in results]
                value_scores = [r.value_score for r in results]
                errors = [r for r in results if r.error]
                
                summary["by_model_kernel"][key] = {
                    "count": len(results),
                    "composite_mean": sum(scores) / len(scores),
                    "composite_min": min(scores),
                    "composite_max": max(scores),
                    "value_mean": sum(value_scores) / len(value_scores),
                    "errors": len(errors),
                }
                
            if "ctn" in kernels and "idiomatic" in kernels:
                result = self.compare(model, "ctn", "idiomatic")
                if result:
                    summary["comparisons"][f"{model}|ctn_vs_idiomatic"] = {
                        "mean_diff": result.mean_diff,
                        "p_value": result.p_value,
                        "effect_size": result.effect_size,
                        "effect_interpretation": result.effect_interpretation,
                        "significant": result.significant_at_05,
                        "ci_lower": result.ci_lower,
                        "ci_upper": result.ci_upper,
                        "n": result.n,
                    }

        return summary
    
    def save(self, output_dir: Path):
        """Save results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        
        # Save raw results by model/kernel
        raw_dir = output_dir / "raw"
        for (model, kernel, doc_id), result in self.results.items():
            model_kernel_dir = raw_dir / model / kernel
            model_kernel_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = model_kernel_dir / f"{doc_id}.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, output_dir: Path) -> "RunResults":
        """Load results from disk."""
        # Load summary for metadata
        summary_path = output_dir / "summary.json"
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        # Parse dates
        started_at = datetime.fromisoformat(summary["started_at"])
        completed_at = None
        if summary.get("completed_at"):
            completed_at = datetime.fromisoformat(summary["completed_at"])
        
        run_results = cls(
            run_id=summary["run_id"],
            config_name=summary["config"],
            started_at=started_at,
            completed_at=completed_at,
        )
        
        # Load all document results
        raw_dir = output_dir / "raw"
        if raw_dir.exists():
            for model_dir in raw_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for kernel_dir in model_dir.iterdir():
                    if not kernel_dir.is_dir():
                        continue
                    for result_file in kernel_dir.glob("*.json"):
                        with open(result_file, "r") as f:
                            d = json.load(f)
                        result = DocumentResult.from_dict(d)
                        run_results.add(result)
        
        return run_results
