"""Tests for evaluation orchestrator."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from ctn_testing.runners.constraint_runner import ConstraintConfig, PromptConfig, RunResult
from ctn_testing.runners.http_runner import (
    SDKRunner,
    SDKResponse,
    DryRunInfo,
    CombinedResponse,
)
from ctn_testing.judging.blind_judge import JudgingResult, TraitScore


def make_combined_response(
    output: str = "Test response",
    provider: str = "anthropic",
    model: str = "claude-sonnet-4",
    tokens: dict | None = None,
    kernel: str = "TEST_KERNEL",
) -> CombinedResponse:
    """Create a mock CombinedResponse for testing."""
    tokens = tokens or {"input": 10, "output": 20}
    return CombinedResponse(
        dry_run=DryRunInfo(
            kernel=kernel,
            system_prompt="System prompt...",
            user_prompt="User prompt",
            parameters={"temperature": 0.7},
        ),
        response=SDKResponse(
            output=output,
            provider=provider,
            model=model,
            tokens=tokens,
            kernel=kernel,
        ),
        kernel_match=True,
    )

# Import evaluation separately to avoid circular import issues
from ctn_testing.runners.evaluation import (
    EvaluationConfig,
    PairedComparison,
    EvaluationResult,
    ConstraintEvaluator,
    load_config,
)


# Sample config YAML
SAMPLE_CONFIG_YAML = """
name: test_evaluation
version: "0.1"
description: "Test config"

runner:
  type: sdk_http
  base_url: "http://localhost:14380"
  timeout: 30

prompts:
  source: prompts/prompts.yaml

models:
  - name: claude-sonnet-4
    provider: anthropic

constraints:
  - name: baseline
    input_prefix: ""
  - name: analytical
    input_prefix: "@analytical "
  - name: terse
    input_prefix: "@terse "

judge_models:
  - name: claude-sonnet-4
    provider: anthropic
    temperature: 0.0

judging:
  blind: true
  traits_definition: definitions/traits.yaml

execution:
  strategy: full_cross
  delay: 0.1

output:
  dir: results/
"""

SAMPLE_PROMPTS_YAML = """
version: "1.0"
prompts:
  - id: recursion
    text: "Explain recursion"
    category: technical
  - id: inflation
    text: "What causes inflation?"
    category: economics
"""

SAMPLE_TRAITS_YAML = """
version: "1.0"
dimensions:
  - name: reasoning_depth
    description: "Reasoning"
    scale: 0-100
    anchors:
      0: "None"
      100: "Full"
  - name: conciseness
    description: "Brevity"
    scale: 0-100
    anchors:
      0: "Verbose"
      100: "Concise"
"""


@pytest.fixture
def config_dir():
    """Create a temporary config directory with all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create directory structure
        (tmpdir / "configs").mkdir()
        (tmpdir / "prompts").mkdir()
        (tmpdir / "definitions").mkdir()

        # Write config
        config_path = tmpdir / "configs" / "test.yaml"
        config_path.write_text(SAMPLE_CONFIG_YAML)

        # Write prompts
        prompts_path = tmpdir / "prompts" / "prompts.yaml"
        prompts_path.write_text(SAMPLE_PROMPTS_YAML)

        # Write traits
        traits_path = tmpdir / "definitions" / "traits.yaml"
        traits_path.write_text(SAMPLE_TRAITS_YAML)

        yield tmpdir


@pytest.fixture
def config_path(config_dir):
    """Get path to test config."""
    return config_dir / "configs" / "test.yaml"


class TestLoadConfig:
    """Tests for loading config from YAML."""

    def test_load_config_basic_fields(self, config_path):
        """Load config with basic fields."""
        config = load_config(config_path)

        assert config.name == "test_evaluation"
        assert config.version == "0.1"
        assert "Test config" in config.description

    def test_load_config_runner_settings(self, config_path):
        """Load runner settings from config."""
        config = load_config(config_path)

        assert config.runner["type"] == "sdk_http"
        assert config.runner["base_url"] == "http://localhost:14380"
        assert config.runner["timeout"] == 30

    def test_load_config_constraints(self, config_path):
        """Load constraints from config."""
        config = load_config(config_path)

        assert len(config.constraints) == 3
        assert config.constraints[0].name == "baseline"
        assert config.constraints[0].input_prefix == ""
        assert config.constraints[1].name == "analytical"
        assert config.constraints[1].input_prefix == "@analytical "

    def test_load_config_prompts_path(self, config_path):
        """Config has correct prompts path."""
        config = load_config(config_path)

        assert config.prompts_source == "prompts/prompts.yaml"
        assert config.prompts_path.name == "prompts.yaml"

    def test_load_config_traits_path(self, config_path):
        """Config has correct traits path."""
        config = load_config(config_path)

        assert "traits.yaml" in str(config.traits_path)

    def test_load_config_file_not_found(self):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_missing_name(self, config_dir):
        """Raise ValueError for config without name."""
        config_path = config_dir / "configs" / "bad.yaml"
        config_path.write_text("version: '1.0'")

        with pytest.raises(ValueError, match="missing required 'name'"):
            load_config(config_path)


class TestEvaluationConfig:
    """Tests for EvaluationConfig properties."""

    def test_baseline_constraint(self, config_path):
        """Get baseline constraint."""
        config = load_config(config_path)

        baseline = config.baseline_constraint
        assert baseline is not None
        assert baseline.name == "baseline"
        assert baseline.input_prefix == ""

    def test_test_constraints(self, config_path):
        """Get non-baseline constraints."""
        config = load_config(config_path)

        test_constraints = config.test_constraints
        assert len(test_constraints) == 2
        names = [c.name for c in test_constraints]
        assert "analytical" in names
        assert "terse" in names
        assert "baseline" not in names


class TestPairedComparison:
    """Tests for PairedComparison dataclass."""

    def test_get_baseline_scores_when_baseline_was_a(self):
        """Get baseline scores when baseline was response A."""
        score_a = TraitScore("reasoning_depth", 80, ["Good"])
        score_b = TraitScore("reasoning_depth", 60, ["OK"])

        judging_result = JudgingResult(
            response_a_scores={"reasoning_depth": score_a},
            response_b_scores={"reasoning_depth": score_b},
        )

        comparison = PairedComparison(
            prompt_id="test",
            prompt_text="Test prompt",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="Baseline response",
            test_response="Test response",
            judging_result=judging_result,
            baseline_was_a=True,
        )

        baseline_scores = comparison.get_baseline_scores()
        assert baseline_scores["reasoning_depth"].score == 80

    def test_get_baseline_scores_when_baseline_was_b(self):
        """Get baseline scores when baseline was response B."""
        score_a = TraitScore("reasoning_depth", 80, ["Good"])
        score_b = TraitScore("reasoning_depth", 60, ["OK"])

        judging_result = JudgingResult(
            response_a_scores={"reasoning_depth": score_a},
            response_b_scores={"reasoning_depth": score_b},
        )

        comparison = PairedComparison(
            prompt_id="test",
            prompt_text="Test prompt",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="Baseline response",
            test_response="Test response",
            judging_result=judging_result,
            baseline_was_a=False,
        )

        baseline_scores = comparison.get_baseline_scores()
        assert baseline_scores["reasoning_depth"].score == 60  # B's score

    def test_get_test_scores(self):
        """Get test constraint scores."""
        score_a = TraitScore("reasoning_depth", 80, ["Good"])
        score_b = TraitScore("reasoning_depth", 60, ["OK"])

        judging_result = JudgingResult(
            response_a_scores={"reasoning_depth": score_a},
            response_b_scores={"reasoning_depth": score_b},
        )

        comparison = PairedComparison(
            prompt_id="test",
            prompt_text="Test prompt",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="Baseline response",
            test_response="Test response",
            judging_result=judging_result,
            baseline_was_a=True,  # So test is B
        )

        test_scores = comparison.get_test_scores()
        assert test_scores["reasoning_depth"].score == 60  # B's score


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_get_comparisons_for_constraint(self):
        """Filter comparisons by constraint."""
        comparisons = [
            PairedComparison(
                prompt_id="p1",
                prompt_text="P1",
                baseline_constraint="baseline",
                test_constraint="analytical",
                baseline_response="",
                test_response="",
                judging_result=JudgingResult(),
                baseline_was_a=True,
            ),
            PairedComparison(
                prompt_id="p1",
                prompt_text="P1",
                baseline_constraint="baseline",
                test_constraint="terse",
                baseline_response="",
                test_response="",
                judging_result=JudgingResult(),
                baseline_was_a=True,
            ),
            PairedComparison(
                prompt_id="p2",
                prompt_text="P2",
                baseline_constraint="baseline",
                test_constraint="analytical",
                baseline_response="",
                test_response="",
                judging_result=JudgingResult(),
                baseline_was_a=False,
            ),
        ]

        result = EvaluationResult(
            config_name="test",
            timestamp="2025-01-01",
            comparisons=comparisons,
        )

        analytical_comps = result.get_comparisons_for("analytical")
        assert len(analytical_comps) == 2

        terse_comps = result.get_comparisons_for("terse")
        assert len(terse_comps) == 1

    def test_get_comparisons_for_prompt(self):
        """Filter comparisons by prompt."""
        comparisons = [
            PairedComparison(
                prompt_id="p1",
                prompt_text="P1",
                baseline_constraint="baseline",
                test_constraint="analytical",
                baseline_response="",
                test_response="",
                judging_result=JudgingResult(),
                baseline_was_a=True,
            ),
            PairedComparison(
                prompt_id="p2",
                prompt_text="P2",
                baseline_constraint="baseline",
                test_constraint="analytical",
                baseline_response="",
                test_response="",
                judging_result=JudgingResult(),
                baseline_was_a=True,
            ),
        ]

        result = EvaluationResult(
            config_name="test",
            timestamp="2025-01-01",
            comparisons=comparisons,
        )

        p1_comps = result.get_comparisons_for_prompt("p1")
        assert len(p1_comps) == 1
        assert p1_comps[0].prompt_id == "p1"


class TestConstraintEvaluator:
    """Tests for ConstraintEvaluator class."""

    def test_evaluator_initializes(self, config_path):
        """Evaluator initializes with config."""
        with patch.object(SDKRunner, '__init__', return_value=None):
            evaluator = ConstraintEvaluator(config_path)

            assert evaluator.config.name == "test_evaluation"
            assert len(evaluator.prompts) == 2

    def test_evaluator_respects_sdk_base_url_override(self, config_path):
        """Evaluator uses override SDK URL."""
        with patch.object(SDKRunner, '__init__', return_value=None) as mock_init:
            evaluator = ConstraintEvaluator(
                config_path,
                sdk_base_url="http://custom:9999",
            )

            mock_init.assert_called_with(
                base_url="http://custom:9999",
                timeout=30.0,
                strategy=None,
            )

    def test_run_produces_evaluation_result(self, config_path):
        """run() produces EvaluationResult with all combinations."""
        with patch.object(SDKRunner, '__init__', return_value=None):
            with patch.object(SDKRunner, 'send_with_dry_run') as mock_send:
                mock_send.return_value = make_combined_response()

                evaluator = ConstraintEvaluator(config_path, random_seed=42)
                result = evaluator.run()

                assert result.config_name == "test_evaluation"
                # 2 prompts × 3 constraints = 6 runs
                assert len(result.run_results) == 6
                # 2 prompts × 2 test constraints = 4 comparisons
                assert len(result.comparisons) == 4

    def test_comparisons_are_randomized(self, config_path):
        """Comparisons have randomized A/B order."""
        with patch.object(SDKRunner, '__init__', return_value=None):
            with patch.object(SDKRunner, 'send_with_dry_run') as mock_send:
                mock_send.return_value = make_combined_response()

                # Run multiple times with different seeds
                results_seed_1 = []
                results_seed_2 = []

                for _ in range(10):
                    evaluator = ConstraintEvaluator(config_path, random_seed=1)
                    result = evaluator.run()
                    results_seed_1.append([c.baseline_was_a for c in result.comparisons])

                    evaluator = ConstraintEvaluator(config_path, random_seed=2)
                    result = evaluator.run()
                    results_seed_2.append([c.baseline_was_a for c in result.comparisons])

                # Same seed should produce same pattern
                assert all(r == results_seed_1[0] for r in results_seed_1)

                # Different seeds should produce different patterns
                assert results_seed_1[0] != results_seed_2[0]

    def test_progress_callback_called(self, config_path):
        """Progress callback is called during run."""
        with patch.object(SDKRunner, '__init__', return_value=None):
            with patch.object(SDKRunner, 'send_with_dry_run') as mock_send:
                mock_send.return_value = make_combined_response()

                callback = Mock()
                evaluator = ConstraintEvaluator(config_path, random_seed=42)
                evaluator.run(progress_callback=callback)

                # Should be called for running and judging phases
                assert callback.call_count > 0

                # Check call patterns
                calls = callback.call_args_list
                stages = [call[0][0] for call in calls]
                assert "running" in stages
                assert "judging" in stages

    def test_handles_run_errors_gracefully(self, config_path):
        """Evaluator handles run errors and skips failed comparisons."""
        from ctn_testing.runners.http_runner import SDKError

        with patch.object(SDKRunner, '__init__', return_value=None):
            with patch.object(SDKRunner, 'send_with_dry_run') as mock_send:
                # First call succeeds, rest fail
                call_count = [0]

                def side_effect(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] <= 3:  # First 3 calls succeed (baseline for both prompts + 1)
                        return make_combined_response()
                    raise SDKError("Connection failed")

                mock_send.side_effect = side_effect

                evaluator = ConstraintEvaluator(config_path, random_seed=42)
                result = evaluator.run()

                # Should have some results despite errors
                assert len(result.run_results) == 6
                # Some should have errors
                errors = [r for r in result.run_results if r.error]
                assert len(errors) > 0


class TestEvaluationResultSummary:
    """Tests for EvaluationResult.summary()."""

    def test_summary_basic_stats(self):
        """Summary includes basic statistics."""
        result = EvaluationResult(
            config_name="test",
            timestamp="2025-01-01",
            run_results=[
                RunResult(
                    prompt_id="p1",
                    constraint_name="baseline",
                    input_sent="test",
                    output="response",
                    provider="anthropic",
                    model="claude-sonnet-4",
                    tokens={"input": 10, "output": 20},
                    timestamp="2025-01-01",
                ),
            ],
        )

        summary = result.summary()

        assert summary["config_name"] == "test"
        assert summary["total_runs"] == 1
        assert summary["run_errors"] == 0

    def test_summary_trait_deltas(self):
        """Summary computes trait deltas correctly."""
        score_baseline = TraitScore("reasoning_depth", 50, [])
        score_test = TraitScore("reasoning_depth", 80, [])

        judging_result = JudgingResult(
            response_a_scores={"reasoning_depth": score_baseline},
            response_b_scores={"reasoning_depth": score_test},
        )

        comparison = PairedComparison(
            prompt_id="p1",
            prompt_text="P1",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="",
            test_response="",
            judging_result=judging_result,
            baseline_was_a=True,  # baseline=A, test=B
        )

        result = EvaluationResult(
            config_name="test",
            timestamp="2025-01-01",
            comparisons=[comparison],
        )

        summary = result.summary()

        # Delta should be test - baseline = 80 - 50 = 30
        assert "analytical" in summary["by_constraint"]
        assert summary["by_constraint"]["analytical"]["trait_deltas"]["reasoning_depth"] == 30
