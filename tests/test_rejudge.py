"""Tests for rejudge functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ctn_testing.judging.blind_judge import JudgingResult, TraitScore
from ctn_testing.runners.constraint_runner import RunResult
from ctn_testing.runners.evaluation import (
    ConstraintEvaluator,
    EvaluationResult,
)
from ctn_testing.runners.output import RunOutputManager

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
            results.append(
                RunResult(
                    prompt_id=prompt_id,
                    constraint_name=constraint,
                    input_sent=f"{prefix}What is {prompt_id}?",
                    output=f"Response for {prompt_id} with {constraint}",
                    provider="anthropic",
                    model="sonnet",
                    tokens={"input": 10, "output": 20},
                    timestamp="2025-01-15T10:30:00",
                    error=None,
                )
            )

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
# RunOutputManager Rejudge Mode Tests
# =============================================================================


class TestRunOutputManagerRejudgeMode:
    """Tests for RunOutputManager in rejudge mode."""

    def test_rejudge_mode_creates_new_timestamped_folder(self, tmp_path, mock_config):
        """Rejudge mode creates new timestamped folder in base_dir."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
        )
        manager.initialize(prompts_count=2)

        # Should create new timestamped folder, not modify source
        assert manager.run_dir != source_run
        assert manager.run_dir.parent == source_run.parent
        assert manager.run_dir.exists()

    def test_rejudge_mode_skips_responses_directory(self, tmp_path, mock_config):
        """Rejudge mode does not create responses/ directory."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
        )
        manager.initialize(prompts_count=2)

        # responses/ should NOT exist in rejudge folder
        assert not (manager.run_dir / "responses").exists()
        # But judging/ and analysis/ should exist
        assert manager.judging_dir.exists()
        assert manager.analysis_dir.exists()

    def test_rejudge_mode_creates_standard_directories(self, tmp_path, mock_config):
        """Rejudge mode creates standard judging/ and analysis/ directories."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
        )
        manager.initialize(prompts_count=2)

        # Should use standard names, NOT suffixed
        assert manager.judging_dir.name == "judging"
        assert manager.analysis_dir.name == "analysis"

    def test_rejudge_manifest_contains_run_type(self, tmp_path, mock_config):
        """Manifest contains run_type='rejudge'."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
        )
        manager.initialize(prompts_count=2)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["run_type"] == "rejudge"

    def test_rejudge_manifest_contains_source_run_id(self, tmp_path, mock_config):
        """Manifest contains source_run_id from source folder name."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
        )
        manager.initialize(prompts_count=2)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["source_run_id"] == "2025-01-15T10-30-00"

    def test_rejudge_manifest_contains_source_responses_path(self, tmp_path, mock_config):
        """Manifest contains source_responses_path relative to run_dir."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
        )
        manager.initialize(prompts_count=2)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["source_responses_path"] is not None
        # Should be relative path like "../2025-01-15T10-30-00"
        assert "2025-01-15T10-30-00" in manifest["source_responses_path"]

    def test_rejudge_manifest_contains_judge_model_override(self, tmp_path, mock_config):
        """Manifest contains judge_model_override when provided."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        source_run = tmp_path / "results" / "2025-01-15T10-30-00"
        source_run.mkdir(parents=True)

        judge_override = {"provider": "anthropic", "name": "haiku"}
        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
            rejudge_source=source_run,
            judge_model_override=judge_override,
        )
        manager.initialize(prompts_count=2)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["judge_model_override"] == judge_override

    def test_regular_mode_has_evaluation_run_type(self, tmp_path, mock_config):
        """Regular mode (not rejudge) has run_type='evaluation'."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
        )
        manager.initialize(prompts_count=2)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["run_type"] == "evaluation"
        assert manifest["source_run_id"] is None
        assert manifest["source_responses_path"] is None

    def test_regular_mode_creates_responses_directory(self, tmp_path, mock_config):
        """Regular mode creates responses/ directory."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=mock_config,
            config_path=config_path,
        )
        manager.initialize(prompts_count=2)

        assert manager.responses_dir.exists()


# =============================================================================
# EvaluationResult.load() Tests
# =============================================================================


class TestEvaluationResultLoadResponsesOnly:
    """Tests for EvaluationResult.load() with load_responses_only parameter."""

    def test_load_responses_only_skips_judging_files(
        self, populated_run_dir, mock_config, tmp_path
    ):
        """load_responses_only=True skips loading judging files."""
        # Add some judging files to the run directory
        judging_dir = populated_run_dir / "judging"
        judging_dir.mkdir(exist_ok=True)

        judging_data = {
            "prompt_id": "prompt1",
            "test_constraint": "analytical",
            "baseline_constraint": "baseline",
            "baseline_was_a": True,
            "response_a_scores": {"reasoning": {"score": 70, "evidence": []}},
            "response_b_scores": {"reasoning": {"score": 85, "evidence": []}},
            "judge_model": {"provider": "anthropic", "name": "sonnet"},
            "timestamp": "2025-01-15T10:35:00",
        }
        (judging_dir / "prompt1_analytical.json").write_text(json.dumps(judging_data))

        # Load with load_responses_only=True
        result = EvaluationResult.load(populated_run_dir, load_responses_only=True)

        # Should have responses but no comparisons
        assert len(result.run_results) == 4  # 2 prompts × 2 constraints
        assert len(result.comparisons) == 0

    def test_load_without_responses_only_loads_judging(
        self, populated_run_dir, mock_config, tmp_path
    ):
        """Default load() loads judging files."""
        # Add some judging files to the run directory
        judging_dir = populated_run_dir / "judging"
        judging_dir.mkdir(exist_ok=True)

        judging_data = {
            "prompt_id": "prompt1",
            "test_constraint": "analytical",
            "baseline_constraint": "baseline",
            "baseline_was_a": True,
            "response_a_scores": {"reasoning": {"score": 70, "evidence": []}},
            "response_b_scores": {"reasoning": {"score": 85, "evidence": []}},
            "judge_model": {"provider": "anthropic", "name": "sonnet"},
            "timestamp": "2025-01-15T10:35:00",
        }
        (judging_dir / "prompt1_analytical.json").write_text(json.dumps(judging_data))

        # Load without load_responses_only (default)
        result = EvaluationResult.load(populated_run_dir, load_responses_only=False)

        # Should have both responses and comparisons
        assert len(result.run_results) == 4
        assert len(result.comparisons) == 1  # One judging file


# =============================================================================
# ConstraintEvaluator.rejudge() Tests
# =============================================================================


class TestRejudge:
    """Tests for ConstraintEvaluator.rejudge() method."""

    def _create_evaluator_mock(self, tmp_path, traits_path=None):
        """Helper to create a mock evaluator."""
        # Create real config files so file operations work
        config_dir = tmp_path / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        if traits_path is None:
            traits_path = config_dir / "traits.yaml"
            traits_path.write_text("""
dimensions:
  - name: reasoning
    description: "Quality"
    scale: "0-100"
    anchors:
      0: "Bad"
      100: "Good"
""")

        prompts_path = config_dir / "prompts.yaml"
        prompts_path.write_text("""
prompts:
  - id: prompt1
    text: "What is prompt1?"
  - id: prompt2
    text: "What is prompt2?"
""")

        config_path = config_dir / "config.yaml"
        config_path.write_text("name: test\n")

        evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
        evaluator.config = MagicMock()
        evaluator.config.name = "test"
        evaluator.config.models = [{"provider": "anthropic", "name": "sonnet"}]
        evaluator.config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
        evaluator.config.traits_path = traits_path
        evaluator.config.prompts_path = prompts_path

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
        evaluator._config_path = config_path

        return evaluator

    def test_creates_new_timestamped_folder(self, populated_run_dir, tmp_path):
        """Rejudge creates new timestamped folder in same parent directory."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                    judge_model="haiku",
                )

        # Result should have a run_dir in the same parent as source
        assert result.run_dir is not None
        assert result.run_dir != populated_run_dir
        assert result.run_dir.parent == populated_run_dir.parent

    def test_new_folder_has_no_responses_directory(self, populated_run_dir, tmp_path):
        """Rejudge folder does not contain responses/ directory."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                )

        # No responses/ directory in rejudge folder
        assert not (result.run_dir / "responses").exists()

    def test_new_folder_has_standard_judging_directory(self, populated_run_dir, tmp_path):
        """Rejudge folder has standard judging/ directory (not suffixed)."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                )

        # Standard judging/ directory
        assert (result.run_dir / "judging").exists()
        # Has judging files
        judging_files = list((result.run_dir / "judging").glob("*.json"))
        assert len(judging_files) > 0

    def test_new_folder_has_standard_analysis_directory(self, populated_run_dir, tmp_path):
        """Rejudge folder has standard analysis/ directory (not suffixed)."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                )

        # Standard analysis/ directory with summary
        assert (result.run_dir / "analysis").exists()
        assert (result.run_dir / "analysis" / "summary.json").exists()

    def test_manifest_has_rejudge_metadata(self, populated_run_dir, tmp_path):
        """Manifest contains run_type, source_run_id, source_responses_path."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                    judge_model="haiku",
                )

        manifest_path = result.run_dir / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["run_type"] == "rejudge"
        assert manifest["source_run_id"] == populated_run_dir.name
        assert manifest["source_responses_path"] is not None
        assert manifest["judge_model_override"]["name"] == "haiku"

    def test_loads_existing_responses(self, populated_run_dir, tmp_path):
        """Loads responses from existing run directory."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                result = evaluator.rejudge(
                    responses_path=populated_run_dir,
                )

        # Verify responses were loaded
        assert len(result.run_results) == 4  # 2 prompts × 2 constraints

    def test_skips_sdk_calls(self, populated_run_dir, tmp_path):
        """Does not invoke constraint_runner (no SDK calls)."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)
            evaluator.constraint_runner = MagicMock()  # Should NOT be called

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                )

        # Verify constraint_runner was never called
        evaluator.constraint_runner.run_prompt.assert_not_called()
        evaluator.constraint_runner.run_single.assert_not_called()

    def test_judge_model_override_works(self, populated_run_dir, tmp_path):
        """Judge model override is passed to BlindJudge."""
        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    judge_model="haiku",
                    judge_provider="anthropic",
                )

                # Verify BlindJudge was created with override
                MockBlindJudge.assert_called_once()
                call_kwargs = MockBlindJudge.call_args[1]
                assert call_kwargs["judge_model"] == "haiku"
                assert call_kwargs["judge_provider"] == "anthropic"

    def test_original_files_unchanged(self, populated_run_dir, tmp_path):
        """Original responses/, judging/, analysis/ are not modified."""
        # Record original state
        original_responses = sorted(
            [f.name for f in (populated_run_dir / "responses").glob("*.json")]
        )
        original_manifest = (populated_run_dir / "manifest.json").read_text()

        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                )

        # Verify original files unchanged
        current_responses = sorted(
            [f.name for f in (populated_run_dir / "responses").glob("*.json")]
        )
        current_manifest = (populated_run_dir / "manifest.json").read_text()

        assert current_responses == original_responses
        assert current_manifest == original_manifest

    def test_raises_on_missing_responses(self, tmp_path):
        """Raises FileNotFoundError if responses_path has no responses."""
        run_dir = tmp_path / "empty-run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text('{"config_file": "test.yaml", "started_at": ""}')
        (run_dir / "responses").mkdir()  # Empty responses dir

        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = ConstraintEvaluator.__new__(ConstraintEvaluator)
            evaluator.config = MagicMock()
            evaluator.config.judge_models = []

            with pytest.raises(FileNotFoundError) as exc_info:
                evaluator.rejudge(responses_path=run_dir)

            assert "No responses found" in str(exc_info.value)

    def test_progress_callback_reports_rejudging(self, populated_run_dir, tmp_path):
        """Progress callback receives 'rejudging' stage."""
        progress_calls = []

        def track_progress(stage, current, total):
            progress_calls.append((stage, current, total))

        with patch.object(ConstraintEvaluator, "__init__", lambda self, *args, **kwargs: None):
            evaluator = self._create_evaluator_mock(tmp_path)

            mock_judging_result = JudgingResult(
                response_a_scores={"reasoning": TraitScore("reasoning", 70, [])},
                response_b_scores={"reasoning": TraitScore("reasoning", 85, [])},
                raw_response="{}",
            )

            with patch("ctn_testing.runners.evaluation.BlindJudge") as MockBlindJudge:
                mock_judge = MagicMock()
                mock_judge.judge.return_value = mock_judging_result
                MockBlindJudge.return_value = mock_judge

                evaluator.rejudge(
                    responses_path=populated_run_dir,
                    progress_callback=track_progress,
                )

        # Verify progress callbacks were made with "rejudging" stage
        assert len(progress_calls) > 0
        stages = set(call[0] for call in progress_calls)
        assert "rejudging" in stages
