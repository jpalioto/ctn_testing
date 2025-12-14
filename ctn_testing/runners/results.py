"""Result storage and aggregation."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json


@dataclass
class FieldResult:
    """Result for a single field extraction."""
    field: str
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


@dataclass
class DocumentResult:
    """Result for a single document."""
    doc_id: str
    model: str
    kernel: str
    fields: list[FieldResult]
    
    # Timing
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Errors
    error: str | None = None
    
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
            "doc_id": self.doc_id,
            "model": self.model,
            "kernel": self.kernel,
            "composite_score": self.composite_score,
            "value_score": self.value_score,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "fields": [
                {
                    "field": f.field,
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
