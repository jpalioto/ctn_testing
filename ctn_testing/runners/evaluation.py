"""Evaluation orchestrator for constraint adherence testing."""
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from .http_runner import SDKRunner
from .constraint_runner import (
    ConstraintConfig,
    PromptConfig,
    RunResult,
    ConstraintRunner,
    load_prompts,
    load_constraints,
)
from .output import RunOutputManager, NullOutputManager
# Import directly from module to avoid circular import through __init__.py
from ..judging.blind_judge import BlindJudge, JudgingResult  # noqa: E402


@dataclass
class EvaluationConfig:
    """Configuration for a constraint adherence evaluation."""
    name: str
    version: str
    description: str
    runner: dict                          # {type, base_url, timeout}
    prompts_source: str                   # Relative path to prompts.yaml
    prompts_include_ids: list[str] | None  # Optional subset
    models: list[dict]                    # [{name, provider}]
    constraints: list[ConstraintConfig]
    judge_models: list[dict]              # [{name, provider, temperature, max_tokens}]
    judging: dict                         # {blind, traits_definition}
    execution: dict                       # {strategy, delay}
    output: dict                          # {dir, include_raw_responses, ...}
    config_dir: Path                      # Directory containing the config file

    @property
    def prompts_path(self) -> Path:
        """Full path to prompts file."""
        return self.config_dir.parent / self.prompts_source

    @property
    def traits_path(self) -> Path:
        """Full path to traits definition file."""
        return self.config_dir.parent / self.judging.get("traits_definition", "")

    @property
    def baseline_constraint(self) -> ConstraintConfig | None:
        """Get the baseline constraint (empty prefix)."""
        for c in self.constraints:
            if c.input_prefix == "" or c.name == "baseline":
                return c
        return None

    @property
    def test_constraints(self) -> list[ConstraintConfig]:
        """Get non-baseline constraints."""
        baseline = self.baseline_constraint
        if baseline is None:
            return self.constraints
        return [c for c in self.constraints if c.name != baseline.name]


@dataclass
class PairedComparison:
    """Result of comparing baseline vs test constraint for one prompt."""
    prompt_id: str
    prompt_text: str
    baseline_constraint: str      # Usually "baseline"
    test_constraint: str          # e.g., "analytical"
    baseline_response: str
    test_response: str
    judging_result: JudgingResult
    baseline_was_a: bool          # For tracking randomization
    error: str | None = None      # If comparison failed

    def get_baseline_scores(self) -> dict:
        """Get scores for baseline response."""
        if self.baseline_was_a:
            return self.judging_result.response_a_scores
        return self.judging_result.response_b_scores

    def get_test_scores(self) -> dict:
        """Get scores for test response."""
        if self.baseline_was_a:
            return self.judging_result.response_b_scores
        return self.judging_result.response_a_scores


@dataclass
class EvaluationResult:
    """Result of a full constraint adherence evaluation."""
    config_name: str
    timestamp: str
    run_results: list[RunResult] = field(default_factory=list)
    comparisons: list[PairedComparison] = field(default_factory=list)

    def get_comparisons_for(self, constraint: str) -> list[PairedComparison]:
        """Get all comparisons for a specific test constraint."""
        return [c for c in self.comparisons if c.test_constraint == constraint]

    def get_comparisons_for_prompt(self, prompt_id: str) -> list[PairedComparison]:
        """Get all comparisons for a specific prompt."""
        return [c for c in self.comparisons if c.prompt_id == prompt_id]

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        summary: dict[str, Any] = {
            "config_name": self.config_name,
            "timestamp": self.timestamp,
            "total_runs": len(self.run_results),
            "total_comparisons": len(self.comparisons),
            "run_errors": len([r for r in self.run_results if r.error]),
            "comparison_errors": len([c for c in self.comparisons if c.error]),
            "by_constraint": {},
        }

        # Group by test constraint
        constraints = set(c.test_constraint for c in self.comparisons)
        for constraint in constraints:
            comps = self.get_comparisons_for(constraint)
            valid_comps = [c for c in comps if not c.error and not c.judging_result.error]

            if not valid_comps:
                continue

            # Aggregate scores across all comparisons
            trait_deltas: dict[str, list[float]] = {}
            for comp in valid_comps:
                baseline_scores = comp.get_baseline_scores()
                test_scores = comp.get_test_scores()

                for trait in baseline_scores:
                    if trait in test_scores:
                        delta = test_scores[trait].score - baseline_scores[trait].score
                        if trait not in trait_deltas:
                            trait_deltas[trait] = []
                        trait_deltas[trait].append(delta)

            # Compute mean deltas
            summary["by_constraint"][constraint] = {
                "count": len(valid_comps),
                "trait_deltas": {
                    trait: sum(deltas) / len(deltas)
                    for trait, deltas in trait_deltas.items()
                },
            }

        return summary


class ConstraintEvaluator:
    """Orchestrator for constraint adherence evaluation.

    Runs all prompt × constraint combinations, then judges baseline vs
    each test constraint with blind, randomized comparison.
    """

    def __init__(
        self,
        config_path: Path,
        sdk_base_url: str | None = None,
        random_seed: int | None = None,
    ):
        """Initialize the evaluator.

        Args:
            config_path: Path to evaluation config (e.g., phase1.yaml)
            sdk_base_url: Override SDK base URL from config
            random_seed: Seed for randomization (for reproducibility)
        """
        self.config = load_config(config_path)
        self._config_path = config_path

        # Set up SDK runner
        base_url = sdk_base_url or self.config.runner.get("base_url", "http://localhost:14380")
        timeout = self.config.runner.get("timeout", 60)
        self.sdk_runner = SDKRunner(base_url=base_url, timeout=float(timeout))

        # Load prompts
        prompts = load_prompts(self.config.prompts_path)
        if self.config.prompts_include_ids:
            prompts = [p for p in prompts if p.id in self.config.prompts_include_ids]
        self.prompts = prompts

        # Set up constraint runner
        model_config = self.config.models[0] if self.config.models else {}
        self.constraint_runner = ConstraintRunner(
            sdk_runner=self.sdk_runner,
            prompts=self.prompts,
            constraints=self.config.constraints,
            provider=model_config.get("provider", "anthropic"),
            model=model_config.get("name"),
        )

        # Set up judge
        judge_config = self.config.judge_models[0] if self.config.judge_models else {}
        self.judge = BlindJudge(
            traits_path=self.config.traits_path,
            sdk_runner=self.sdk_runner,
            judge_provider=judge_config.get("provider", "anthropic"),
            judge_model=judge_config.get("name"),
        )

        # Randomization
        self._rng = random.Random(random_seed)

        # Set up output manager
        self._output_manager = self._create_output_manager()

    def _create_output_manager(self) -> RunOutputManager | NullOutputManager:
        """Create output manager based on config."""
        output_config = self.config.output
        if not output_config or not output_config.get("dir"):
            return NullOutputManager()

        # Resolve output dir relative to config directory
        output_dir_str = output_config.get("dir", "results/")
        output_dir = self.config.config_dir.parent / output_dir_str

        return RunOutputManager(
            base_dir=output_dir,
            config=self.config,
            config_path=self._config_path,
        )

    def run(
        self,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> EvaluationResult:
        """Run full evaluation.

        Args:
            progress_callback: Called with (stage, current, total) during run

        Returns:
            EvaluationResult with all run results and comparisons

        Raises:
            PersistenceError: If output directory cannot be created or written to.
                No SDK calls are made if persistence fails (fail-fast behavior).
        """
        # Initialize output (creates directories, copies configs, writes manifest)
        # PersistenceError propagates here - no SDK calls if we can't persist
        self._output_manager.initialize(prompts_count=len(self.prompts))

        result = EvaluationResult(
            config_name=self.config.name,
            timestamp=datetime.now().isoformat(),
        )

        errors: list[str] = []

        # Phase 1: Run all prompt × constraint combinations
        self._run_phase(result, progress_callback)

        # Collect run errors
        for run_result in result.run_results:
            if run_result.error:
                errors.append(f"Run error [{run_result.prompt_id}×{run_result.constraint_name}]: {run_result.error}")

        # Phase 2: Compare baseline vs each test constraint
        self._judge_phase(result, progress_callback)

        # Collect comparison errors
        for comp in result.comparisons:
            if comp.error:
                errors.append(f"Judge error [{comp.prompt_id}×{comp.test_constraint}]: {comp.error}")

        # Finalize output (updates manifest with completion time and errors)
        self._output_manager.finalize(errors=errors)

        return result

    def _run_phase(
        self,
        result: EvaluationResult,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        """Run all prompt × constraint combinations.

        Raises:
            PersistenceError: If a response cannot be saved (fail-fast).
        """
        total = len(self.prompts) * len(self.config.constraints)
        current = 0

        for prompt in self.prompts:
            prompt_results = self.constraint_runner.run_prompt(prompt)
            for constraint_name, run_result in prompt_results.items():
                result.run_results.append(run_result)

                # Save response to disk (fail-fast on error)
                self._output_manager.save_response(
                    run_result=run_result,
                    prompt_text=prompt.text,
                )

                current += 1
                if progress_callback:
                    progress_callback("running", current, total)

    def _judge_phase(
        self,
        result: EvaluationResult,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        """Judge baseline vs each test constraint.

        Raises:
            PersistenceError: If a judging result cannot be saved (fail-fast).
        """
        baseline = self.config.baseline_constraint
        if baseline is None:
            return

        test_constraints = self.config.test_constraints
        total = len(self.prompts) * len(test_constraints)
        current = 0

        # Get judge model config for saving
        judge_config = self.config.judge_models[0] if self.config.judge_models else {}
        judge_model = {
            "provider": judge_config.get("provider", "anthropic"),
            "name": judge_config.get("name", ""),
        }

        # Group run results by prompt
        results_by_prompt: dict[str, dict[str, RunResult]] = {}
        for run_result in result.run_results:
            if run_result.prompt_id not in results_by_prompt:
                results_by_prompt[run_result.prompt_id] = {}
            results_by_prompt[run_result.prompt_id][run_result.constraint_name] = run_result

        # Compare each prompt
        for prompt in self.prompts:
            prompt_results = results_by_prompt.get(prompt.id, {})
            baseline_result = prompt_results.get(baseline.name)

            if baseline_result is None or baseline_result.error:
                # Skip if baseline failed
                current += len(test_constraints)
                continue

            for test_constraint in test_constraints:
                test_result = prompt_results.get(test_constraint.name)

                if test_result is None or test_result.error:
                    # Skip if test failed
                    current += 1
                    if progress_callback:
                        progress_callback("judging", current, total)
                    continue

                comparison = self._compare_pair(
                    prompt=prompt,
                    baseline_result=baseline_result,
                    test_result=test_result,
                    baseline_constraint=baseline.name,
                    test_constraint=test_constraint.name,
                )
                result.comparisons.append(comparison)

                # Save judging result to disk (fail-fast on error)
                self._output_manager.save_judging(
                    comparison=comparison,
                    judge_model=judge_model,
                    timestamp=datetime.now().isoformat(),
                )

                current += 1
                if progress_callback:
                    progress_callback("judging", current, total)

    def _compare_pair(
        self,
        prompt: PromptConfig,
        baseline_result: RunResult,
        test_result: RunResult,
        baseline_constraint: str,
        test_constraint: str,
    ) -> PairedComparison:
        """Run blind comparison between baseline and test.

        Randomizes which response is A vs B to prevent position bias.
        """
        # Randomize order
        baseline_was_a = self._rng.random() < 0.5

        if baseline_was_a:
            response_a = baseline_result.output
            response_b = test_result.output
        else:
            response_a = test_result.output
            response_b = baseline_result.output

        # Call judge
        try:
            judging_result = self.judge.judge(
                response_a=response_a,
                response_b=response_b,
                prompt_text=prompt.text,
            )
            error = judging_result.error
        except Exception as e:
            judging_result = JudgingResult(raw_response="", error=str(e))
            error = str(e)

        return PairedComparison(
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            baseline_constraint=baseline_constraint,
            test_constraint=test_constraint,
            baseline_response=baseline_result.output,
            test_response=test_result.output,
            judging_result=judging_result,
            baseline_was_a=baseline_was_a,
            error=error,
        )


def load_config(path: Path) -> EvaluationConfig:
    """Load evaluation config from YAML file.

    Args:
        path: Path to config file (e.g., phase1.yaml)

    Returns:
        EvaluationConfig with all settings

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: expected dict, got {type(data)}")

    name = data.get("name")
    if not name:
        raise ValueError("Config missing required 'name' field")

    # Parse constraints
    constraints = load_constraints(data)

    # Parse prompts config
    prompts_config = data.get("prompts", {})
    if isinstance(prompts_config, dict):
        prompts_source = prompts_config.get("source", "prompts/prompts.yaml")
        prompts_include_ids = prompts_config.get("include_ids")
    else:
        prompts_source = "prompts/prompts.yaml"
        prompts_include_ids = None

    return EvaluationConfig(
        name=name,
        version=data.get("version", ""),
        description=data.get("description", ""),
        runner=data.get("runner", {}),
        prompts_source=prompts_source,
        prompts_include_ids=prompts_include_ids,
        models=data.get("models", []),
        constraints=constraints,
        judge_models=data.get("judge_models", []),
        judging=data.get("judging", {}),
        execution=data.get("execution", {}),
        output=data.get("output", {}),
        config_dir=path.parent,
    )
