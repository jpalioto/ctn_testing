"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import ModelConfig


@dataclass
class DatasetConfig:
    """External dataset configuration."""

    type: str  # "docile", "funsd", etc.
    path: str  # Path to dataset
    split: str = "val"  # Dataset split
    n: int | None = None  # Limit number of documents


@dataclass
class DocumentConfig:
    source: str = "synthetic"
    per_set: int = 100
    types: list[dict] = field(default_factory=list)


@dataclass
class KernelConfig:
    name: str
    path: str
    enabled: bool = True


@dataclass
class ComparisonConfig:
    a: str
    b: str
    primary: bool = False
    within: str = "model"


@dataclass
class MetricsConfig:
    primary: list[str] = field(default_factory=list)
    secondary: list[str] = field(default_factory=list)
    composite_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class StatisticsConfig:
    method: str = "paired_ttest"
    alpha: float = 0.05
    correction: str = "none"
    cluster_by: str = "document"


@dataclass
class ExecutionConfig:
    delay: float = 0.3
    retries: int = 3
    timeout: int = 60
    parallel: bool = False
    max_workers: int = 4


@dataclass
class OutputConfig:
    results_dir: str = "results"
    save_raw: bool = True
    generate_report: bool = True


def _parse_model_config(data: dict) -> ModelConfig:
    return ModelConfig(
        provider=data["provider"],
        name=data["name"],
        api_key_env=data.get("api_key_env", f"{data['provider'].upper()}_API_KEY"),
        temperature=data.get("temperature", 0.0),
        max_tokens=data.get("max_tokens", 2048),
    )


@dataclass
class EvaluationConfig:
    """
    Full evaluation configuration.

    judge_policy: match_provider | full_cross | single
        - match_provider: Judge matches eval model's provider (default)
        - full_cross: Every judge scores every extraction (expensive, slow)
        - single: First judge only
    """

    name: str
    version: str
    documents: DocumentConfig
    models: list[ModelConfig]
    judge_models: list[ModelConfig]
    kernels: list[KernelConfig]
    comparisons: list[ComparisonConfig]
    metrics: MetricsConfig
    statistics: StatisticsConfig
    execution: ExecutionConfig
    output: OutputConfig
    judge_policy: str = "match_provider"
    dataset: DatasetConfig | None = None  # External dataset (optional)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvaluationConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse dataset if present
        dataset = None
        if "dataset" in data:
            dataset = DatasetConfig(**data["dataset"])

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "0.1"),
            documents=DocumentConfig(**data.get("documents", {})),
            models=[_parse_model_config(m) for m in data.get("models", [])],
            judge_models=[_parse_model_config(m) for m in data.get("judge_models", [])],
            kernels=[KernelConfig(**k) for k in data.get("kernels", [])],
            comparisons=[ComparisonConfig(**c) for c in data.get("comparisons", [])],
            metrics=MetricsConfig(**data.get("metrics", {})),
            statistics=StatisticsConfig(**data.get("statistics", {})),
            execution=ExecutionConfig(**data.get("execution", {})),
            output=OutputConfig(**data.get("output", {})),
            judge_policy=data.get("judge_policy", "match_provider"),
            dataset=dataset,
        )

    def enabled_kernels(self) -> list[KernelConfig]:
        return [k for k in self.kernels if k.enabled]

    def run_matrix(self) -> list[tuple[ModelConfig, KernelConfig]]:
        """Cartesian product of models x enabled kernels."""
        return [(model, kernel) for model in self.models for kernel in self.enabled_kernels()]
