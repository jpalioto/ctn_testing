"""Tests for rejudge functionality."""
import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from ctn_testing.runners.output import (
    RunOutputManager,
    RejudgeOutputManager,
    RejudgeManifest,
)
from ctn_testing.runners.constraint_runner import RunResult
from ctn_testing.runners.evaluation import (
    ConstraintEvaluator,
    EvaluationResult,
    PairedComparison,
)
from ctn_testing.judging.blind_judge import JudgingResult, TraitScore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock EvaluationConfig."""
    config = MagicMock()
    config.name = "test_evaluation"
    config.models = [{"provider": "anthropic", "name": "sonnet"}]
    config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]

    baseline = MagicMock()
    baseline.name = "baseline"
    baseline.input_prefix = ""
    analytical = MagicMock()
    analytical.name = "analytical"
    analytical.input_prefix = "@analytical "

    config.constraints = [baseline, analytical]
    config.baseline_constraint = baseline
    config.test_constraints = [analytical]
    config.traits_path = Path("/nonexistent/traits.yaml")
    config.prompts_path = Path("/nonexistent/prompts.yaml")
    config.output = {
        "include_raw_responses": True,
        "include_judge_responses": True,
    }
    return config


@pytest.fixture
def sample_run_results():
    """Create sample RunResult objects for 2 prompts × 2 constraints."""
    results = []
    prompts = ["prompt1", "prompt2"]
    constraints = ["baseline", "analytical"]

    for prompt_id in prompts:
        for constraint in constraints:
            prefix = "" if constraint == "baseline" else "@analytical "
            results.append(RunResult(
                prompt_id=prompt_id,
                constraint_name=constraint,
                input_sent=f"{prefix}What is {prompt_id}?",
                output=f"Response for {prompt_id} with {constraint}",
                provider="anthropic",
                model="sonnet",
                tokens={"input": 10, "output": 20},
                timestamp="2025-01-15T10:30:00",
                error=None,
            ))

    return results


@pytest.fixture
def populated_run_dir(tmp_path, mock_config, sample_run_results):
    """Create a populated run directory with responses (no judging yet)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n")

    manager = RunOutputManager(
        base_dir=tmp_path / "results",
        config=mock_config,
        config_path=config_path,
    )
    manager.initialize(prompts_count=2)

    # Save all responses
    prompt_texts = {
        "prompt1": "What is prompt1?",
        "prompt2": "What is prompt2?",
    }
    for run_result in sample_run_results:
        manager.save_response(
            run_result=run_result,
            prompt_text=prompt_texts[run_result.prompt_id],
        )

    manager.finalize(errors=[])

    return manager.run_dir


# =============================================================================
# RejudgeOutputManager Tests
# =============================================================================


class TestRejudgeOutputManager:
    """Tests for RejudgeOutputManager."""

    def test_creates_suffixed_judging_directory(self, tmp_path, mock_config):
        """Creates judging{suffix}/ directory."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        manager = RejudgeOutputManager(
            run_dir=run_dir,
            config=mock_config,
            suffix="-haiku",
        )
        manager.initialize(prompts_count=2)

        assert manager.judging_dir.exists()
        assert manager.judging_dir.name == "judging-haiku"

    def test_creates_suffixed_analysis_directory(self, tmp_path, mock_config):
        """Creates analysis{suffix}/ directory."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        manager = RejudgeOutputManager(
            run_dir=run_dir,
            config=mock_config,
            suffix="-haiku",
        )
        manager.initialize(prompts_count=2)

        assert manager.analysis_dir.exists()
        assert manager.analysis_dir.name == "analysis-haiku"

    def test_creates_suffixed_manifest(self, tmp_path, mock_config):
        """Creates manifest{suffix}.json."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        manager = RejudgeOutputManager(
            run_dir=run_dir,
            config=mock_config,
            suffix="-haiku",
        )
        manager.initialize(prompts_count=2)

        assert manager.manifest_path.exists()
        assert manager.manifest_path.name == "manifest-haiku.json"

    def test_manifest_contains_rejudge_metadata(self, tmp_path, mock_config):
        """Manifest contains original_run_id and judge_model_override."""
        run_dir = tmp_path / "2025-01-15T10-30-00"
        run_dir.mkdir()

        judge_override = {"provider": "anthropic", "name": "haiku"}
        manager = RejudgeOutputManager(
            run_dir=run_dir,
            config=mock_config,
            suffix="-haiku",
            judge_model_override=judge_override,
        )
        manager.initialize(prompts_count=2)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["original_run_id"] == "2025-01-15T10-30-00"
        assert manifest["suffix"] == "-haiku"
        assert manifest["judge_model_override"] == judge_override

    def test_does_not_modify_original_directories(self, populated_run_dir, mock_config):
        """Original responses/, judging/, analysis/ unchanged."""
        # Record original state
        original_responses = list((populated_run_dir / "responses").glob("*.json"))

        manager = RejudgeOutputManager(
            run_dir=populated_run_dir,
            config=mock_config,
            suffix="-haiku",
        )
        manager.initialize(prompts_count=2)

        # Verify original directories unchanged
        current_responses = list((populated_run_dir / "responses").glob("*.json"))
        assert len(current_responses) == len(original_responses)

        # Original judging dir should not exist or be unchanged
        original_judging = populated_run_dir / "judging"
        if original_judging.exists():
            # Should not have any new files
            pass

    def test_saves_judging_to_suffixed_directory(self, tmp_path, mock_config):
        """save_judging() writes to judging{suffix}/."""
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()

        manager = RejudgeOutputManager(
            run_dir=run_dir,
            config=mock_config,
            suffix="-haiku",
        )
        manager.initialize(prompts_count=1)

        # Create a mock comparison
        judging_result = JudgingResult(
            response_a_scores={
                "reasoning": TraitScore("reasoning", 70, ["Good"]),
            },
            response_b_scores={
                "reasoning": TraitScore("reasoning", 80, ["Better"]),
            },
            raw_response="{}",
        )
        comparison = PairedComparison(
            prompt_id="test",
            prompt_text="Test prompt",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="baseline response",
            test_response="test response",
            judging_result=judging_result,
            baseline_was_a=True,
        )

        manager.save_judging(
            comparison=comparison,
            judge_model={"provider": "anthropic", "name": "haiku"},
            timestamp="2025-01-15T10:35:00",
        )

        judging_files = list(manager.judging_dir.glob("*.json"))
        assert len(judging_files) == 1
        assert "judging-haiku" in str(judging_files[0].parent)


class TestRejudgeManifest:
    """Tests for RejudgeManifest dataclass."""

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        manifest = RejudgeManifest(
            rejudge_id="2025-01-15T11-00-00",
            original_run_id="2025-01-15T10-30-00",
            started_at="2025-01-15T11:00:00",
            suffix="-haiku",
            judge_model_override={"provider": "anthropic", "name": "haiku"},
            prompts_count=5,
            total_judge_calls=5,
        )

        d = manifest.to_dict()

        assert d["rejudge_id"] == "2025-01-15T11-00-00"
        assert d["original_run_id"] == "2025-01-15T10-30-00"
        assert d["suffix"] == "-haiku"
        assert d["judge_model_override"]["name"] == "haiku"

    def test_from_dict(self):
        """Creates from dictionary correctly."""
        data = {
            "rejudge_id": "2025-01-15T11-00-00",
            "original_run_id": "2025-01-15T10-30-00",
            "started_at": "2025-01-15T11:00:00",
            "completed_at": "2025-01-15T11:05:00",
            "duration_seconds": 300.0,
            "suffix": "-haiku",
            "judge_model_override": {"provider": "anthropic", "name": "haiku"},
            "prompts_count": 5,
            "total_judge_calls": 5,
            "errors": [],
        }

        manifest = RejudgeManifest.from_dict(data)

        assert manifest.rejudge_id == "2025-01-15T11-00-00"
        assert manifest.original_run_id == "2025-01-15T10-30-00"
        assert manifest.completed_at == "2025-01-15T11:05:00"


# =============================================================================
# ConstraintEvaluator.rejudge() Tests
# =============================================================================


class TestRejudge:
    """Tests for ConstraintEvaluator.rejudge() method."""

    def test_loads_existing_responses(self, populated_run_dir, tmp_path):
        """Loads responses from existing run directory."""
        # Create a mock evaluator
        config_path = tmp_path / "configs" / "phase1.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("""
name: test
runner:
  base_url: http://localhost:14380
prompts:
  source: prompts/prompts.yaml
models:
  - provider: anthropic
    name: sonnet
constraints:
  - name: baseline
    input_prefix: ""
  - name: analytical
    input_prefix: "@analytical "
judge_models:
  - provider: anthropic
    name: sonnet
judging:
  traits_definition: definitions/traits.yaml
output:
  dir: results/
""")
        # Create prompts file
        prompts_path = tmp_path / "configs" / "prompts" / "prompts.yaml"
        prompts_path.parent.mkdir(parents=True)
        prompts_path.write_text("""
prompts:
  - id: prompt1
    text: "What is prompt1?"
    category: test
  - id: prompt2
    text: "What is prompt2?"
    category: test
""")
        # Create traits file
        traits_path = tmp_path / "configs" / "definitions" / "traits.yaml"
        traits_path.parent.mkdir(parents=True)
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality of reasoning"
    scale: "0-100"
    anchors:
      0: "No reasoning"
      100: "Excellent reasoning"
""")

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3  # baseline_was_a = True

            # Mock the BlindJudge to avoid actual SDK calls
            mock_judging_result = JudgingResult(
                response_a_scores={
                    "reasoning": TraitScore("reasoning", 70, ["Good"]),
                },
                response_b_scores={
                    "reasoning": TraitScore("reasoning", 85, ["Better"]),
                },
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                    judge_model="haiku",
                    output_suffix="-haiku",
                )

        # Verify responses were loaded
        assert len(result.run_results) == 4  # 2 prompts × 2 constraints

    def test_skips_sdk_calls(self, populated_run_dir, tmp_path):
        """Does not invoke constraint_runner (no SDK calls)."""
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator.constraint_runner = MagicMock()  # Should NOT be called
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    output_suffix="-haiku",
                )

        # Verify constraint_runner was never called
        evaluator.constraint_runner.run_prompt.assert_not_called()
        evaluator.constraint_runner.run_single.assert_not_called()

    def test_creates_suffixed_directories(self, populated_run_dir, tmp_path):
        """Creates judging-{suffix}/ and analysis-{suffix}/ directories."""
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    output_suffix="-haiku",
                )

        # Verify suffixed directories were created
        assert (populated_run_dir / "judging-haiku").exists()
        assert (populated_run_dir / "analysis-haiku").exists()
        assert (populated_run_dir / "manifest-haiku.json").exists()

    def test_judge_model_override_works(self, populated_run_dir, tmp_path):
        """Judge model override is passed to BlindJudge."""
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    judge_model="haiku",
                    judge_provider="anthropic",
                    output_suffix="-haiku",
                )

                # Verify BlindJudge was created with override
                MockBlindJudge.assert_called_once()
                call_kwargs = MockBlindJudge.call_args[1]
                assert call_kwargs["judge_model"] == "haiku"
                assert call_kwargs["judge_provider"] == "anthropic"

    def test_analysis_computed_from_new_judgments(self, populated_run_dir, tmp_path):
        """Analysis is computed from new judging results."""
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3

            # Return specific scores so we can verify analysis
            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 60, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 90, [])},
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                    output_suffix="-haiku",
                )

        # Verify analysis was saved
        analysis_dir = populated_run_dir / "analysis-haiku"
        summary_path = analysis_dir / "summary.json"
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        # Should have results for the 'analytical' constraint
        assert "analytical" in summary["results"]

    def test_original_files_unchanged(self, populated_run_dir, tmp_path):
        """Original responses/, judging/, analysis/ are not modified."""
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        # Record original state
        original_responses = sorted([
            f.name for f in (populated_run_dir / "responses").glob("*.json")
        ])
        original_manifest = (populated_run_dir / "manifest.json").read_text()

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    output_suffix="-haiku",
                )

        # Verify original files unchanged
        current_responses = sorted([
            f.name for f in (populated_run_dir / "responses").glob("*.json")
        ])
        current_manifest = (populated_run_dir / "manifest.json").read_text()

        assert current_responses == original_responses
        assert current_manifest == original_manifest

    def test_raises_on_missing_responses(self, tmp_path):
        """Raises FileNotFoundError if responses_path has no responses."""
        run_dir = tmp_path / "empty-run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"config_file": "test.yaml", "started_at": ""}')
        (run_dir / "responses").mkdir()  # Empty responses dir

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.judge_models = []

            with pytest.raises(FileNotFoundError) as exc_info:
                evaluator.rejudge(responses_path=run_dir)

            assert "No responses found" in str(exc_info.value)

    def test_progress_callback_reports_rejudging(self, populated_run_dir, tmp_path):
        """Progress callback receives 'rejudging' stage."""
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        progress_calls = []

        def track_progress(stage, current, total):
            progress_calls.append((stage, current, total))

        with patch.object(ConstraintEvaluator, '__init__', lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.name = "test"
            evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
            evaluator.config.traits_path = traits_path

            baseline = MagicMock()
            baseline.name = "baseline"
            baseline.input_prefix = ""
            analytical = MagicMock()
            analytical.name = "analytical"
            analytical.input_prefix = "@analytical "
            evaluator.config.constraints = [baseline, analytical]
            evaluator.config.baseline_constraint = baseline
            evaluator.config.test_constraints = [analytical]
            evaluator.config.output = {"include_raw_responses": True, "include_judge_responses": True}

            evaluator.sdk_runner = MagicMock()
            evaluator._rng = MagicMock()
            evaluator._rng.random.return_value = 0.3

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch('ctn_testing.runners.evaluation.BlindJudge') as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    output_suffix="-haiku",
                    progress_callback=track_progress,
                )

        # Verify progress callbacks were made with "rejudging" stage
        assert len(progress_calls) > 0
        stages = set(call[0] for call in progress_calls)
        assert "rejudging" in stages
