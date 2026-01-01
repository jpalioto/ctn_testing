"""Output management for constraint adherence evaluation results."""

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..statistics.constraint_analysis import ConstraintAnalysis
    from .constraint_runner import RunResult
    from .evaluation import EvaluationConfig, EvaluationResult, PairedComparison


class PersistenceError(Exception):
    """Raised when results cannot be persisted."""

    pass


@dataclass
class RunManifest:
    """Manifest containing run metadata for reproducibility."""

    run_id: str  # Timestamp-based ID
    started_at: str  # ISO timestamp
    completed_at: str | None = None  # Filled at end
    duration_seconds: float | None = None  # Filled at end
    config_file: str = ""  # Original config filename
    prompts_count: int = 0
    constraints: list[str] = field(default_factory=list)
    models: list[dict] = field(default_factory=list)
    judge_models: list[dict] = field(default_factory=list)
    total_sdk_calls: int = 0  # Expected count
    total_judge_calls: int = 0  # Expected count
    errors: list[str] = field(default_factory=list)
    # Rejudge-specific fields (None for regular runs)
    run_type: str = "evaluation"  # "evaluation" or "rejudge"
    source_run_id: str | None = None  # For rejudge: original run ID
    source_responses_path: str | None = None  # For rejudge: relative path to responses
    judge_model_override: dict | None = None  # For rejudge: override model config

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        """Create from dictionary."""
        # Handle older manifests without rejudge fields
        data.setdefault("run_type", "evaluation")
        data.setdefault("source_run_id", None)
        data.setdefault("source_responses_path", None)
        data.setdefault("judge_model_override", None)
        return cls(**data)


class RunOutputManager:
    """Manages output directory structure for evaluation runs.

    Creates timestamped run folders with:
    - Copied config files for reproducibility
    - Manifest with run metadata
    - Space for results (written in later phases)

    Also supports rejudge mode where responses are read from a source run
    and only judging/analysis directories are created.
    """

    def __init__(
        self,
        base_dir: Path,
        config: "EvaluationConfig",
        config_path: Path,
        *,
        rejudge_source: Path | None = None,
        judge_model_override: dict | None = None,
    ):
        """Initialize the output manager.

        Args:
            base_dir: Base output directory (e.g., results/)
            config: Loaded evaluation config
            config_path: Path to the original config file
            rejudge_source: For rejudge mode: path to source run directory
            judge_model_override: For rejudge mode: judge model override config
        """
        self.run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        self.base_dir = base_dir
        self.run_dir = base_dir / self.run_id
        self.config = config
        self.config_path = config_path
        self._start_time: datetime | None = None
        self._manifest: RunManifest | None = None
        # Rejudge mode settings
        self._rejudge_source = rejudge_source
        self._judge_model_override = judge_model_override
        self._is_rejudge = rejudge_source is not None

    @property
    def config_dir(self) -> Path:
        """Directory for copied config files."""
        return self.run_dir / "config"

    @property
    def responses_dir(self) -> Path:
        """Directory for individual response files."""
        return self.run_dir / "responses"

    @property
    def judging_dir(self) -> Path:
        """Directory for individual judging result files."""
        return self.run_dir / "judging"

    @property
    def analysis_dir(self) -> Path:
        """Directory for analysis summary files."""
        return self.run_dir / "analysis"

    @property
    def manifest_path(self) -> Path:
        """Path to manifest.json."""
        return self.run_dir / "manifest.json"

    def initialize(self, prompts_count: int) -> None:
        """Create directories, copy configs, write initial manifest.

        Args:
            prompts_count: Number of prompts that will be run

        Raises:
            PersistenceError: If directories cannot be created, configs cannot
                be copied, or manifest cannot be written.
        """
        self._start_time = datetime.now()

        # Create directories
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Skip responses directory in rejudge mode (responses live in source)
            if not self._is_rejudge:
                self.responses_dir.mkdir(parents=True, exist_ok=True)
            self.judging_dir.mkdir(parents=True, exist_ok=True)
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PersistenceError(f"Cannot create output directory {self.run_dir}: {e}") from e

        # Validate we can actually write to the directory
        self.validate_writable()

        # Copy config files
        try:
            self._copy_configs()
        except OSError as e:
            raise PersistenceError(f"Cannot copy config files to {self.config_dir}: {e}") from e

        # Create initial manifest
        self._manifest = self._create_manifest(prompts_count)
        try:
            self._write_manifest()
        except OSError as e:
            raise PersistenceError(f"Cannot write manifest to {self.manifest_path}: {e}") from e

    def validate_writable(self) -> None:
        """Write and delete a test file to verify permissions.

        Raises:
            PersistenceError: If the run directory is not writable.
        """
        test_file = self.run_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise PersistenceError(f"Cannot write to {self.run_dir}: {e}") from e

    def finalize(self, errors: list[str] | None = None) -> None:
        """Update manifest with completion time and errors.

        Args:
            errors: List of error messages from the run
        """
        if self._manifest is None:
            return

        end_time = datetime.now()
        self._manifest.completed_at = end_time.isoformat()

        if self._start_time:
            duration = (end_time - self._start_time).total_seconds()
            self._manifest.duration_seconds = duration

        if errors:
            self._manifest.errors = errors

        self._write_manifest()

    def add_error(self, error: str) -> None:
        """Add an error to the manifest.

        Args:
            error: Error message to record
        """
        if self._manifest is not None:
            self._manifest.errors.append(error)

    def save_response(
        self,
        run_result: "RunResult",
        prompt_text: str,
    ) -> None:
        """Save individual response to responses/ directory.

        Args:
            run_result: Result of running a prompt with a constraint
            prompt_text: Original prompt text (not included in RunResult)

        Raises:
            PersistenceError: If response cannot be written.
        """
        filename = f"{run_result.prompt_id}_{run_result.constraint_name}.json"

        # Check if we should include raw output
        include_raw = self.config.output.get("include_raw_responses", True)

        # Build response data using RunResult.to_dict() for new format
        response_data = run_result.to_dict()
        response_data["prompt_text"] = prompt_text

        # Apply include_raw filter
        if not include_raw:
            response_data["output"] = None

        # Write atomically with UTF-8 encoding
        self._write_json_file(self.responses_dir, filename, response_data)

    def save_judging(
        self,
        comparison: "PairedComparison",
        judge_model: dict,
        timestamp: str,
    ) -> None:
        """Save individual judging result to judging/ directory.

        Args:
            comparison: Paired comparison result
            judge_model: Judge model config {"provider": ..., "name": ...}
            timestamp: Timestamp of the judging call

        Raises:
            PersistenceError: If judging result cannot be written.
        """
        filename = f"{comparison.prompt_id}_{comparison.test_constraint}_vs_{comparison.baseline_constraint}.json"

        # Check config options
        include_raw_responses = self.config.output.get("include_raw_responses", True)
        include_judge_responses = self.config.output.get("include_judge_responses", True)

        # Build scores dict, converting TraitScore to serializable format
        baseline_scores = {}
        test_scores = {}

        for trait, trait_score in comparison.get_baseline_scores().items():
            baseline_scores[trait] = {
                "score": trait_score.score,
                "reasons": trait_score.reasons,
            }

        for trait, trait_score in comparison.get_test_scores().items():
            test_scores[trait] = {
                "score": trait_score.score,
                "reasons": trait_score.reasons,
            }

        # Build judging data
        judging_data = {
            "prompt_id": comparison.prompt_id,
            "prompt_text": comparison.prompt_text,
            "baseline_constraint": comparison.baseline_constraint,
            "test_constraint": comparison.test_constraint,
            "baseline_was_a": comparison.baseline_was_a,
            "baseline_response": comparison.baseline_response if include_raw_responses else None,
            "test_response": comparison.test_response if include_raw_responses else None,
            "scores": {
                "baseline": baseline_scores,
                "test": test_scores,
            },
            "judge_model": judge_model,
            "judge_raw_response": comparison.judging_result.raw_response
            if include_judge_responses
            else None,
            "timestamp": timestamp,
        }

        # Include error if present
        if comparison.error:
            judging_data["error"] = comparison.error

        # Write atomically with UTF-8 encoding
        self._write_json_file(self.judging_dir, filename, judging_data)

    def save_analysis(
        self,
        result: "EvaluationResult",
        analyses: dict[str, "ConstraintAnalysis"],
    ) -> None:
        """Save summary.json and comparisons.json to analysis/ directory.

        Args:
            result: Evaluation result with all comparisons
            analyses: Dict of constraint name to ConstraintAnalysis

        Raises:
            PersistenceError: If analysis files cannot be written.
        """
        timestamp = datetime.now().isoformat()

        # Get baseline constraint name
        baseline_constraint = self.config.baseline_constraint
        baseline_name = baseline_constraint.name if baseline_constraint else "baseline"

        # Build summary.json
        summary_data = {
            "generated_at": timestamp,
            "config_name": self.config.name,
            "prompts_count": len(set(c.prompt_id for c in result.comparisons)),
            "constraints_tested": list(analyses.keys()),
            "baseline_constraint": baseline_name,
            "results": {},
        }

        for constraint, analysis in analyses.items():
            traits_data = {}
            for trait_name, comp in analysis.trait_comparisons.items():
                traits_data[trait_name] = {
                    "baseline_mean": comp.baseline_mean,
                    "test_mean": comp.test_mean,
                    "mean_diff": comp.mean_diff,
                    "p_value": comp.p_value,
                    "effect_size": comp.effect_size,
                    "effect_interpretation": comp.effect_interpretation,
                    "significant": comp.significant,
                }

            summary_data["results"][constraint] = {
                "n_prompts": analysis.n_prompts,
                "significant_improvements": analysis.improved_traits(),
                "significant_degradations": analysis.degraded_traits(),
                "traits": traits_data,
            }

        # Build comparisons.json
        comparisons_data = {
            "generated_at": timestamp,
            "comparisons": [],
        }

        for comp in result.comparisons:
            if comp.error or comp.judging_result.error:
                continue

            baseline_scores = {}
            test_scores = {}
            deltas = {}

            for trait, score in comp.get_baseline_scores().items():
                baseline_scores[trait] = score.score

            for trait, score in comp.get_test_scores().items():
                test_scores[trait] = score.score

            for trait in baseline_scores:
                if trait in test_scores:
                    deltas[trait] = test_scores[trait] - baseline_scores[trait]

            comparisons_data["comparisons"].append(
                {
                    "prompt_id": comp.prompt_id,
                    "test_constraint": comp.test_constraint,
                    "baseline_scores": baseline_scores,
                    "test_scores": test_scores,
                    "deltas": deltas,
                }
            )

        # Write both files atomically
        self._write_analysis_file("summary.json", summary_data)
        self._write_analysis_file("comparisons.json", comparisons_data)

    def _write_json_file(self, directory: Path, filename: str, data: dict) -> None:
        """Write a JSON file atomically with UTF-8 encoding.

        Uses ensure_ascii=False to preserve Unicode characters like Greek
        symbols (Σ, Ψ, Ω, τ) in the output.

        Args:
            directory: Directory to write file in
            filename: Name of the file (e.g., "response.json")
            data: Data to write as JSON

        Raises:
            PersistenceError: If file cannot be written.
        """
        dest_path = directory / filename

        try:
            fd, temp_path = tempfile.mkstemp(
                dir=directory,
                suffix=".tmp",
                prefix=f".{filename}.",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, dest_path)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception as e:
            raise PersistenceError(f"Cannot save to {dest_path}: {e}") from e

    def _write_analysis_file(self, filename: str, data: dict) -> None:
        """Write an analysis file atomically.

        Args:
            filename: Name of the file (e.g., "summary.json")
            data: Data to write as JSON

        Raises:
            PersistenceError: If file cannot be written.
        """
        self._write_json_file(self.analysis_dir, filename, data)

    def _copy_configs(self) -> None:
        """Copy config files to the run directory."""
        # Copy main config
        if self.config_path.exists():
            dest = self.config_dir / self.config_path.name
            shutil.copy2(self.config_path, dest)

        # Copy traits definition
        traits_path = self.config.traits_path
        if traits_path.exists():
            dest = self.config_dir / "traits.yaml"
            shutil.copy2(traits_path, dest)

        # Copy prompts
        prompts_path = self.config.prompts_path
        if prompts_path.exists():
            dest = self.config_dir / "prompts.yaml"
            shutil.copy2(prompts_path, dest)

    def _create_manifest(self, prompts_count: int) -> RunManifest:
        """Create the initial manifest."""
        constraints = [c.name for c in self.config.constraints]
        n_constraints = len(constraints)

        # Calculate expected call counts
        # SDK calls: prompts × constraints (0 for rejudge)
        total_sdk_calls = 0 if self._is_rejudge else prompts_count * n_constraints

        # Judge calls: prompts × (constraints - 1 baseline)
        # Each non-baseline constraint compared with baseline
        n_test_constraints = len(self.config.test_constraints)
        total_judge_calls = prompts_count * n_test_constraints

        # Rejudge-specific fields
        if self._is_rejudge and self._rejudge_source is not None:
            run_type = "rejudge"
            source_run_id = self._rejudge_source.name
            # Calculate relative path from new run to source responses
            source_responses_path = f"../{source_run_id}/responses"
            judge_model_override = self._judge_model_override
        else:
            run_type = "evaluation"
            source_run_id = None
            source_responses_path = None
            judge_model_override = None

        return RunManifest(
            run_id=self.run_id,
            started_at=self._start_time.isoformat() if self._start_time else "",
            config_file=self.config_path.name,
            prompts_count=prompts_count,
            constraints=constraints,
            models=self.config.models,
            judge_models=self.config.judge_models,
            total_sdk_calls=total_sdk_calls,
            total_judge_calls=total_judge_calls,
            run_type=run_type,
            source_run_id=source_run_id,
            source_responses_path=source_responses_path,
            judge_model_override=judge_model_override,
        )

    def _write_manifest(self) -> None:
        """Write manifest to disk with UTF-8 encoding."""
        if self._manifest is None:
            return

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest.to_dict(), f, indent=2, ensure_ascii=False)


class NullOutputManager:
    """No-op output manager when persistence is disabled."""

    def __init__(self):
        self.run_id = "none"
        self.run_dir = Path("/dev/null")

    def initialize(self, prompts_count: int) -> None:
        """No-op."""
        pass

    def finalize(self, errors: list[str] | None = None) -> None:
        """No-op."""
        pass

    def add_error(self, error: str) -> None:
        """No-op."""
        pass

    def save_response(
        self,
        run_result: "RunResult",
        prompt_text: str,
    ) -> None:
        """No-op."""
        pass

    def save_judging(
        self,
        comparison: "PairedComparison",
        judge_model: dict,
        timestamp: str,
    ) -> None:
        """No-op."""
        pass

    def save_analysis(
        self,
        result: "EvaluationResult",
        analyses: dict[str, "ConstraintAnalysis"],
    ) -> None:
        """No-op."""
        pass
