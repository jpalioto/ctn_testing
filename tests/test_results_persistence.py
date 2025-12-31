"""Tests for data integrity of persisted results."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ctn_testing.judging.blind_judge import JudgingResult, TraitScore
from ctn_testing.runners.constraint_runner import RunResult
from ctn_testing.runners.evaluation import (
    EvaluationResult,
    PairedComparison,
)
from ctn_testing.runners.output import RunOutputManager
from ctn_testing.statistics.constraint_analysis import full_analysis

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock EvaluationConfig with 2 prompts, 2 constraints."""
    config = MagicMock()
    config.name = "test_evaluation"
    config.models = [{"provider": "anthropic", "name": "sonnet"}]
    config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]

    # 2 constraints: baseline and analytical
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
def sample_comparisons():
    """Create sample PairedComparison objects for 2 prompts."""
    comparisons = []
    prompts = ["prompt1", "prompt2"]

    for i, prompt_id in enumerate(prompts):
        baseline_was_a = i % 2 == 0  # Alternate for testing

        # Create TraitScore objects
        baseline_scores = {
            "reasoning_depth": TraitScore(
                dimension="reasoning_depth",
                score=70,
                reasons=["Good reasoning"],
            ),
            "conciseness": TraitScore(
                dimension="conciseness",
                score=60,
                reasons=["Average length"],
            ),
        }
        test_scores = {
            "reasoning_depth": TraitScore(
                dimension="reasoning_depth",
                score=85,
                reasons=["Excellent step-by-step"],
            ),
            "conciseness": TraitScore(
                dimension="conciseness",
                score=55,
                reasons=["Slightly verbose"],
            ),
        }

        # Create JudgingResult with correct A/B assignment
        if baseline_was_a:
            response_a_scores = baseline_scores
            response_b_scores = test_scores
        else:
            response_a_scores = test_scores
            response_b_scores = baseline_scores

        judging_result = JudgingResult(
            response_a_scores=response_a_scores,
            response_b_scores=response_b_scores,
            raw_response='{"response_a": {...}, "response_b": {...}}',
            error=None,
        )

        comparisons.append(
            PairedComparison(
                prompt_id=prompt_id,
                prompt_text=f"What is {prompt_id}?",
                baseline_constraint="baseline",
                test_constraint="analytical",
                baseline_response=f"Response for {prompt_id} with baseline",
                test_response=f"Response for {prompt_id} with analytical",
                judging_result=judging_result,
                baseline_was_a=baseline_was_a,
                error=None,
            )
        )

    return comparisons


@pytest.fixture
def populated_run_dir(tmp_path, mock_config, sample_run_results, sample_comparisons):
    """Create a fully populated run directory with all files."""
    # Create output manager
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

    # Save all judging results
    judge_model = {"provider": "anthropic", "name": "sonnet"}
    for comparison in sample_comparisons:
        manager.save_judging(
            comparison=comparison,
            judge_model=judge_model,
            timestamp="2025-01-15T10:35:00",
        )

    # Create EvaluationResult and save analysis
    result = EvaluationResult(
        config_name="test_evaluation",
        timestamp="2025-01-15T10:30:00",
        run_results=sample_run_results,
        comparisons=sample_comparisons,
    )
    analyses = full_analysis(result)
    manager.save_analysis(result=result, analyses=analyses)

    # Finalize
    manager.finalize(errors=[])

    return manager.run_dir


# =============================================================================
# Structure Validation Tests
# =============================================================================


class TestStructureValidation:
    """Tests for validating JSON structure of persisted files."""

    def test_summary_has_required_fields(self, populated_run_dir):
        """summary.json has config_name, generated_at, constraints_tested, results."""
        summary_path = populated_run_dir / "analysis" / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        assert "generated_at" in summary
        assert "config_name" in summary
        assert "constraints_tested" in summary
        assert "results" in summary
        assert "prompts_count" in summary
        assert "baseline_constraint" in summary

    def test_comparisons_has_required_fields(self, populated_run_dir):
        """comparisons.json has generated_at, comparisons array."""
        comparisons_path = populated_run_dir / "analysis" / "comparisons.json"
        with open(comparisons_path) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "comparisons" in data
        assert isinstance(data["comparisons"], list)

    def test_response_files_have_required_fields(self, populated_run_dir):
        """Each response file has prompt_id, output, tokens, etc."""
        responses_dir = populated_run_dir / "responses"
        required_fields = [
            "prompt_id",
            "prompt_text",
            "constraint_name",
            "input_sent",
            "output",
            "provider",
            "model",
            "tokens",
            "timestamp",
        ]

        for response_file in responses_dir.glob("*.json"):
            with open(response_file) as f:
                data = json.load(f)

            for field in required_fields:
                assert field in data, f"Missing {field} in {response_file.name}"

    def test_judging_files_have_required_fields(self, populated_run_dir):
        """Each judging file has scores, baseline_was_a, etc."""
        judging_dir = populated_run_dir / "judging"
        required_fields = [
            "prompt_id",
            "prompt_text",
            "baseline_constraint",
            "test_constraint",
            "baseline_was_a",
            "scores",
            "judge_model",
            "timestamp",
        ]

        for judging_file in judging_dir.glob("*.json"):
            with open(judging_file) as f:
                data = json.load(f)

            for field in required_fields:
                assert field in data, f"Missing {field} in {judging_file.name}"

            # Verify scores structure
            assert "baseline" in data["scores"]
            assert "test" in data["scores"]

    def test_manifest_has_required_fields(self, populated_run_dir):
        """manifest.json has all required metadata fields."""
        manifest_path = populated_run_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        required_fields = [
            "run_id",
            "started_at",
            "completed_at",
            "duration_seconds",
            "config_file",
            "prompts_count",
            "constraints",
            "models",
            "judge_models",
            "total_sdk_calls",
            "total_judge_calls",
            "errors",
        ]

        for field in required_fields:
            assert field in manifest, f"Missing {field} in manifest.json"


# =============================================================================
# Data Completeness Tests
# =============================================================================


class TestDataCompleteness:
    """Tests for validating completeness of persisted data."""

    def test_response_count_matches_prompts_times_constraints(self, populated_run_dir):
        """len(responses/) == prompts × constraints."""
        responses_dir = populated_run_dir / "responses"
        response_files = list(responses_dir.glob("*.json"))

        # 2 prompts × 2 constraints = 4 responses
        assert len(response_files) == 4

    def test_judging_count_matches_prompts_times_test_constraints(self, populated_run_dir):
        """len(judging/) == prompts × (constraints - 1)."""
        judging_dir = populated_run_dir / "judging"
        judging_files = list(judging_dir.glob("*.json"))

        # 2 prompts × 1 test constraint = 2 judging results
        assert len(judging_files) == 2

    def test_all_prompt_ids_present_in_responses(self, populated_run_dir):
        """Every prompt_id from config appears in responses."""
        responses_dir = populated_run_dir / "responses"
        expected_prompts = {"prompt1", "prompt2"}

        found_prompts = set()
        for response_file in responses_dir.glob("*.json"):
            with open(response_file) as f:
                data = json.load(f)
            found_prompts.add(data["prompt_id"])

        assert found_prompts == expected_prompts

    def test_all_constraints_present_in_responses(self, populated_run_dir):
        """Every constraint from config appears in responses."""
        responses_dir = populated_run_dir / "responses"
        expected_constraints = {"baseline", "analytical"}

        found_constraints = set()
        for response_file in responses_dir.glob("*.json"):
            with open(response_file) as f:
                data = json.load(f)
            found_constraints.add(data["constraint_name"])

        assert found_constraints == expected_constraints

    def test_all_prompt_constraint_combinations_present(self, populated_run_dir):
        """Every prompt × constraint combination has a response file."""
        responses_dir = populated_run_dir / "responses"
        expected_combinations = {
            ("prompt1", "baseline"),
            ("prompt1", "analytical"),
            ("prompt2", "baseline"),
            ("prompt2", "analytical"),
        }

        found_combinations = set()
        for response_file in responses_dir.glob("*.json"):
            with open(response_file) as f:
                data = json.load(f)
            found_combinations.add((data["prompt_id"], data["constraint_name"]))

        assert found_combinations == expected_combinations

    def test_comparisons_json_has_all_comparisons(self, populated_run_dir):
        """comparisons.json contains all paired comparisons."""
        comparisons_path = populated_run_dir / "analysis" / "comparisons.json"
        with open(comparisons_path) as f:
            data = json.load(f)

        # 2 prompts, each compared with analytical vs baseline
        assert len(data["comparisons"]) == 2

        prompt_ids = {c["prompt_id"] for c in data["comparisons"]}
        assert prompt_ids == {"prompt1", "prompt2"}


# =============================================================================
# Round-Trip Integrity Tests
# =============================================================================


class TestRoundTripIntegrity:
    """Tests for save → load → compare integrity."""

    def test_save_load_roundtrip(self, populated_run_dir, sample_run_results, sample_comparisons):
        """Save results → Load from disk → Compare to original."""
        loaded = EvaluationResult.load(populated_run_dir)

        assert loaded.config_name == "config"  # From config.yaml filename
        assert len(loaded.run_results) == len(sample_run_results)
        assert len(loaded.comparisons) == len(sample_comparisons)

    def test_scores_match_exactly_after_reload(self, populated_run_dir, sample_comparisons):
        """Numeric scores identical after save/load."""
        loaded = EvaluationResult.load(populated_run_dir)

        # Create lookup for original comparisons
        original_by_prompt = {c.prompt_id: c for c in sample_comparisons}

        for loaded_comp in loaded.comparisons:
            original = original_by_prompt[loaded_comp.prompt_id]

            # Get baseline and test scores from both
            original_baseline = original.get_baseline_scores()
            original_test = original.get_test_scores()
            loaded_baseline = loaded_comp.get_baseline_scores()
            loaded_test = loaded_comp.get_test_scores()

            # Compare scores for each trait
            for trait in original_baseline:
                assert loaded_baseline[trait].score == original_baseline[trait].score
                assert loaded_test[trait].score == original_test[trait].score

    def test_responses_match_exactly_after_reload(self, populated_run_dir, sample_run_results):
        """Response text identical when include_raw_responses=true."""
        loaded = EvaluationResult.load(populated_run_dir)

        # Create lookup for original run results
        original_by_key = {(r.prompt_id, r.constraint_name): r for r in sample_run_results}

        for loaded_result in loaded.run_results:
            key = (loaded_result.prompt_id, loaded_result.constraint_name)
            original = original_by_key[key]

            assert loaded_result.output == original.output
            assert loaded_result.input_sent == original.input_sent
            assert loaded_result.provider == original.provider
            assert loaded_result.model == original.model

    def test_tokens_preserved_after_reload(self, populated_run_dir, sample_run_results):
        """Token counts preserved after save/load."""
        loaded = EvaluationResult.load(populated_run_dir)

        original_by_key = {(r.prompt_id, r.constraint_name): r for r in sample_run_results}

        for loaded_result in loaded.run_results:
            key = (loaded_result.prompt_id, loaded_result.constraint_name)
            original = original_by_key[key]

            assert loaded_result.tokens == original.tokens

    def test_baseline_was_a_preserved(self, populated_run_dir, sample_comparisons):
        """baseline_was_a flag preserved after save/load."""
        loaded = EvaluationResult.load(populated_run_dir)

        original_by_prompt = {c.prompt_id: c for c in sample_comparisons}

        for loaded_comp in loaded.comparisons:
            original = original_by_prompt[loaded_comp.prompt_id]
            assert loaded_comp.baseline_was_a == original.baseline_was_a


# =============================================================================
# Config Honored Tests
# =============================================================================


class TestConfigHonored:
    """Tests for config options being honored in persistence."""

    def test_raw_responses_excluded_when_configured(
        self, tmp_path, sample_run_results, sample_comparisons
    ):
        """include_raw_responses: false → output is null."""
        # Create config with include_raw_responses=false
        config = MagicMock()
        config.name = "test_evaluation"
        config.models = []
        config.judge_models = []
        config.constraints = []
        config.test_constraints = []
        config.baseline_constraint = MagicMock(name="baseline")
        config.baseline_constraint.name = "baseline"
        config.traits_path = Path("/nonexistent/traits.yaml")
        config.prompts_path = Path("/nonexistent/prompts.yaml")
        config.output = {
            "include_raw_responses": False,
            "include_judge_responses": True,
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=config,
            config_path=config_path,
        )
        manager.initialize(prompts_count=2)

        # Save a response
        manager.save_response(
            run_result=sample_run_results[0],
            prompt_text="Test prompt",
        )

        # Verify output is null
        response_files = list(manager.responses_dir.glob("*.json"))
        assert len(response_files) == 1

        with open(response_files[0]) as f:
            data = json.load(f)

        assert data["output"] is None

    def test_judge_responses_excluded_when_configured(self, tmp_path, sample_comparisons):
        """include_judge_responses: false → judge_raw_response is null."""
        # Create config with include_judge_responses=false
        config = MagicMock()
        config.name = "test_evaluation"
        config.models = []
        config.judge_models = []
        config.constraints = []
        config.test_constraints = []
        config.baseline_constraint = MagicMock(name="baseline")
        config.baseline_constraint.name = "baseline"
        config.traits_path = Path("/nonexistent/traits.yaml")
        config.prompts_path = Path("/nonexistent/prompts.yaml")
        config.output = {
            "include_raw_responses": True,
            "include_judge_responses": False,
        }

        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n")

        manager = RunOutputManager(
            base_dir=tmp_path / "results",
            config=config,
            config_path=config_path,
        )
        manager.initialize(prompts_count=2)

        # Save a judging result
        manager.save_judging(
            comparison=sample_comparisons[0],
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-01-15T10:35:00",
        )

        # Verify judge_raw_response is null
        judging_files = list(manager.judging_dir.glob("*.json"))
        assert len(judging_files) == 1

        with open(judging_files[0]) as f:
            data = json.load(f)

        assert data["judge_raw_response"] is None

    def test_raw_responses_included_when_configured(self, populated_run_dir):
        """include_raw_responses: true → output and responses present."""
        responses_dir = populated_run_dir / "responses"

        for response_file in responses_dir.glob("*.json"):
            with open(response_file) as f:
                data = json.load(f)

            assert data["output"] is not None
            assert len(data["output"]) > 0

    def test_judge_responses_included_when_configured(self, populated_run_dir):
        """include_judge_responses: true → judge_raw_response present."""
        judging_dir = populated_run_dir / "judging"

        for judging_file in judging_dir.glob("*.json"):
            with open(judging_file) as f:
                data = json.load(f)

            assert data["judge_raw_response"] is not None


# =============================================================================
# Reload Capability Tests
# =============================================================================


class TestReloadCapability:
    """Tests for reloading EvaluationResult from saved files."""

    def test_can_reload_evaluation_result(self, populated_run_dir):
        """EvaluationResult.load(path) reconstructs from saved files."""
        loaded = EvaluationResult.load(populated_run_dir)

        assert isinstance(loaded, EvaluationResult)
        assert loaded.config_name is not None
        assert len(loaded.run_results) > 0
        assert len(loaded.comparisons) > 0

    def test_analysis_on_reloaded_matches_original(
        self, populated_run_dir, sample_run_results, sample_comparisons
    ):
        """full_analysis() on reloaded data matches original analysis."""
        # Create original result and run analysis
        original = EvaluationResult(
            config_name="test_evaluation",
            timestamp="2025-01-15T10:30:00",
            run_results=sample_run_results,
            comparisons=sample_comparisons,
        )
        original_analysis = full_analysis(original)

        # Load from disk and run analysis
        loaded = EvaluationResult.load(populated_run_dir)
        loaded_analysis = full_analysis(loaded)

        # Compare analysis results
        assert set(original_analysis.keys()) == set(loaded_analysis.keys())

        for constraint in original_analysis:
            orig = original_analysis[constraint]
            load = loaded_analysis[constraint]

            assert orig.n_prompts == load.n_prompts
            assert set(orig.trait_comparisons.keys()) == set(load.trait_comparisons.keys())

            for trait in orig.trait_comparisons:
                orig_comp = orig.trait_comparisons[trait]
                load_comp = load.trait_comparisons[trait]

                # Compare statistical values
                assert abs(orig_comp.baseline_mean - load_comp.baseline_mean) < 0.001
                assert abs(orig_comp.test_mean - load_comp.test_mean) < 0.001
                assert abs(orig_comp.mean_diff - load_comp.mean_diff) < 0.001

    def test_load_raises_on_missing_directory(self, tmp_path):
        """EvaluationResult.load raises FileNotFoundError for missing dir."""
        with pytest.raises(FileNotFoundError) as exc_info:
            EvaluationResult.load(tmp_path / "nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_load_raises_on_missing_manifest(self, tmp_path):
        """EvaluationResult.load raises FileNotFoundError for missing manifest."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            EvaluationResult.load(run_dir)

        assert "manifest" in str(exc_info.value).lower()

    def test_load_handles_empty_responses_dir(self, tmp_path):
        """EvaluationResult.load works with empty responses directory."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "responses").mkdir()

        manifest = {"config_file": "test.yaml", "started_at": "2025-01-15T10:00:00"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        loaded = EvaluationResult.load(run_dir)

        assert len(loaded.run_results) == 0

    def test_load_handles_empty_judging_dir(self, tmp_path):
        """EvaluationResult.load works with empty judging directory."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "judging").mkdir()

        manifest = {"config_file": "test.yaml", "started_at": "2025-01-15T10:00:00"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        loaded = EvaluationResult.load(run_dir)

        assert len(loaded.comparisons) == 0

    def test_reloaded_summary_matches_original(
        self, populated_run_dir, sample_run_results, sample_comparisons
    ):
        """EvaluationResult.summary() matches after reload."""
        # Create original and get summary
        original = EvaluationResult(
            config_name="test_evaluation",
            timestamp="2025-01-15T10:30:00",
            run_results=sample_run_results,
            comparisons=sample_comparisons,
        )
        original_summary = original.summary()

        # Load and get summary
        loaded = EvaluationResult.load(populated_run_dir)
        loaded_summary = loaded.summary()

        # Compare summaries
        assert loaded_summary["total_runs"] == original_summary["total_runs"]
        assert loaded_summary["total_comparisons"] == original_summary["total_comparisons"]
        assert loaded_summary["run_errors"] == original_summary["run_errors"]
        assert loaded_summary["comparison_errors"] == original_summary["comparison_errors"]

        # Compare trait deltas
        for constraint in original_summary["by_constraint"]:
            orig_deltas = original_summary["by_constraint"][constraint]["trait_deltas"]
            load_deltas = loaded_summary["by_constraint"][constraint]["trait_deltas"]

            for trait in orig_deltas:
                assert abs(orig_deltas[trait] - load_deltas[trait]) < 0.001
