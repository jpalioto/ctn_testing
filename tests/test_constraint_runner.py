"""Tests for constraint runner."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ctn_testing.runners.constraint_runner import (
    ConstraintConfig,
    ConstraintRunner,
    PromptConfig,
    RunResult,
    load_constraints,
    load_prompts,
)
from ctn_testing.runners.http_runner import (
    CombinedResponse,
    DryRunInfo,
    SDKError,
    SDKResponse,
    SDKRunner,
)


def make_combined_response(
    output: str = "Response",
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


# Sample prompts YAML
SAMPLE_PROMPTS_YAML = """
version: "1.0"

prompts:
  - id: recursion
    text: "Explain recursion"
    category: technical
    complexity: medium
    notes: "Classic CS concept"

  - id: inflation
    text: "What causes inflation?"
    category: economics
    complexity: medium
"""


@pytest.fixture
def prompts_file():
    """Create a temporary prompts file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_PROMPTS_YAML)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def mock_runner():
    """Create a mock SDK runner."""
    return Mock(spec=SDKRunner)


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        PromptConfig(id="recursion", text="Explain recursion", category="technical"),
        PromptConfig(id="inflation", text="What causes inflation?", category="economics"),
    ]


@pytest.fixture
def sample_constraints():
    """Sample constraints for testing."""
    return [
        ConstraintConfig(name="baseline", input_prefix=""),
        ConstraintConfig(name="analytical", input_prefix="@analytical "),
        ConstraintConfig(name="terse", input_prefix="@terse "),
    ]


class TestLoadPrompts:
    """Tests for loading prompts from YAML."""

    def test_load_prompts_from_yaml(self, prompts_file):
        """Load prompts from valid YAML file."""
        prompts = load_prompts(prompts_file)

        assert len(prompts) == 2
        assert prompts[0].id == "recursion"
        assert prompts[0].text == "Explain recursion"
        assert prompts[0].category == "technical"
        assert prompts[1].id == "inflation"

    def test_load_prompts_preserves_all_fields(self, prompts_file):
        """All prompt fields are loaded."""
        prompts = load_prompts(prompts_file)

        assert prompts[0].complexity == "medium"
        assert prompts[0].notes == "Classic CS concept"

    def test_load_prompts_file_not_found(self):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_prompts(Path("/nonexistent/prompts.yaml"))

    def test_load_prompts_invalid_format(self):
        """Raise ValueError for invalid YAML structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("just a string")
            f.flush()

            with pytest.raises(ValueError, match="expected dict"):
                load_prompts(Path(f.name))

    def test_load_prompts_missing_id(self):
        """Raise ValueError for prompt without id."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
prompts:
  - text: "Missing id"
    category: test
""")
            f.flush()

            with pytest.raises(ValueError, match="missing required 'id'"):
                load_prompts(Path(f.name))

    def test_load_prompts_missing_text(self):
        """Raise ValueError for prompt without text."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
prompts:
  - id: test
    category: test
""")
            f.flush()

            with pytest.raises(ValueError, match="missing required 'text'"):
                load_prompts(Path(f.name))


class TestLoadConstraints:
    """Tests for loading constraints from config."""

    def test_load_constraints_from_config(self):
        """Load constraints from config dict."""
        config = {
            "constraints": [
                {"name": "baseline", "input_prefix": ""},
                {"name": "analytical", "input_prefix": "@analytical "},
            ]
        }

        constraints = load_constraints(config)

        assert len(constraints) == 2
        assert constraints[0].name == "baseline"
        assert constraints[0].input_prefix == ""
        assert constraints[1].name == "analytical"
        assert constraints[1].input_prefix == "@analytical "

    def test_load_constraints_with_description(self):
        """Load constraint description."""
        config = {
            "constraints": [
                {
                    "name": "terse",
                    "input_prefix": "@terse ",
                    "description": "Be concise",
                }
            ]
        }

        constraints = load_constraints(config)

        assert constraints[0].description == "Be concise"

    def test_load_constraints_empty_list(self):
        """Handle empty constraints list."""
        config = {"constraints": []}

        constraints = load_constraints(config)

        assert constraints == []

    def test_load_constraints_missing_name(self):
        """Raise ValueError for constraint without name."""
        config = {"constraints": [{"input_prefix": "@test "}]}

        with pytest.raises(ValueError, match="missing required 'name'"):
            load_constraints(config)

    def test_load_constraints_default_prefix(self):
        """Default to empty prefix if not specified."""
        config = {"constraints": [{"name": "baseline"}]}

        constraints = load_constraints(config)

        assert constraints[0].input_prefix == ""


class TestConstraintRunner:
    """Tests for ConstraintRunner class."""

    def test_run_single_builds_correct_input(self, mock_runner, sample_prompts, sample_constraints):
        """run_single builds correct input string with prefix."""
        mock_runner.send_with_dry_run.return_value = make_combined_response()

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
            provider="anthropic",
        )

        # Test with analytical constraint
        runner.run_single(sample_prompts[0], sample_constraints[1])

        # Check the input sent to SDK
        call_kwargs = mock_runner.send_with_dry_run.call_args[1]
        assert call_kwargs["input"] == "@analytical Explain recursion"

    def test_run_single_baseline_no_prefix(self, mock_runner, sample_prompts, sample_constraints):
        """Baseline constraint has empty prefix."""
        mock_runner.send_with_dry_run.return_value = make_combined_response()

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
        )

        result = runner.run_single(sample_prompts[0], sample_constraints[0])

        call_kwargs = mock_runner.send_with_dry_run.call_args[1]
        assert call_kwargs["input"] == "Explain recursion"
        assert result.input_sent == "Explain recursion"

    def test_run_single_returns_all_fields(self, mock_runner, sample_prompts, sample_constraints):
        """run_single returns RunResult with all fields including dry_run."""
        mock_runner.send_with_dry_run.return_value = make_combined_response(
            output="This is the response",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens={"input": 15, "output": 25},
            kernel="MY_KERNEL",
        )

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        result = runner.run_single(sample_prompts[0], sample_constraints[1])

        assert result.prompt_id == "recursion"
        assert result.constraint_name == "analytical"
        assert result.input_sent == "@analytical Explain recursion"
        assert result.output == "This is the response"
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4"
        assert result.tokens == {"input": 15, "output": 25}
        assert result.timestamp  # Should be set
        assert result.error is None
        # New fields from dry-run
        assert result.dry_run is not None
        assert result.dry_run.kernel == "MY_KERNEL"
        assert result.kernel == "MY_KERNEL"
        assert result.kernel_match is True

    def test_run_single_handles_sdk_error(self, mock_runner, sample_prompts, sample_constraints):
        """run_single stores error in result on SDK failure."""
        mock_runner.send_with_dry_run.side_effect = SDKError("Connection failed")

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
        )

        result = runner.run_single(sample_prompts[0], sample_constraints[0])

        assert result.error == "Connection failed"
        assert result.output == ""
        assert result.prompt_id == "recursion"
        assert result.constraint_name == "baseline"

    def test_run_all_returns_all_combinations(
        self, mock_runner, sample_prompts, sample_constraints
    ):
        """run_all returns len(prompts) × len(constraints) results."""
        mock_runner.send_with_dry_run.return_value = make_combined_response()

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
        )

        results = runner.run_all()

        # 2 prompts × 3 constraints = 6 results
        assert len(results) == 6

        # Check we have all combinations
        combinations = {(r.prompt_id, r.constraint_name) for r in results}
        expected = {
            ("recursion", "baseline"),
            ("recursion", "analytical"),
            ("recursion", "terse"),
            ("inflation", "baseline"),
            ("inflation", "analytical"),
            ("inflation", "terse"),
        }
        assert combinations == expected

    def test_run_prompt_returns_dict_by_constraint(
        self, mock_runner, sample_prompts, sample_constraints
    ):
        """run_prompt returns dict keyed by constraint name."""
        mock_runner.send_with_dry_run.return_value = make_combined_response()

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
        )

        results = runner.run_prompt(sample_prompts[0])

        assert "baseline" in results
        assert "analytical" in results
        assert "terse" in results
        assert results["baseline"].constraint_name == "baseline"
        assert results["analytical"].constraint_name == "analytical"

    def test_run_prompt_same_prompt_all_constraints(
        self, mock_runner, sample_prompts, sample_constraints
    ):
        """run_prompt runs same prompt with all constraints."""
        mock_runner.send_with_dry_run.return_value = make_combined_response()

        runner = ConstraintRunner(
            sdk_runner=mock_runner,
            prompts=sample_prompts,
            constraints=sample_constraints,
        )

        results = runner.run_prompt(sample_prompts[0])

        # All results should be for the same prompt
        for name, result in results.items():
            assert result.prompt_id == "recursion"


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_run_result_to_dict(self):
        """RunResult converts to dict for serialization."""
        result = RunResult(
            prompt_id="test",
            constraint_name="analytical",
            input_sent="@analytical Test",
            output="Response",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens={"input": 10, "output": 20},
            timestamp="2025-01-01T00:00:00",
            error=None,
        )

        d = result.to_dict()

        assert d["prompt_id"] == "test"
        assert d["constraint_name"] == "analytical"
        assert d["input_sent"] == "@analytical Test"
        assert d["output"] == "Response"
        assert d["tokens"] == {"input": 10, "output": 20}

    def test_run_result_with_error(self):
        """RunResult stores error correctly."""
        result = RunResult(
            prompt_id="test",
            constraint_name="baseline",
            input_sent="Test",
            output="",
            provider="anthropic",
            model="",
            tokens={"input": 0, "output": 0},
            timestamp="2025-01-01T00:00:00",
            error="Connection timeout",
        )

        assert result.error == "Connection timeout"
        assert result.to_dict()["error"] == "Connection timeout"


class TestConstraintConfig:
    """Tests for ConstraintConfig dataclass."""

    def test_constraint_config_fields(self):
        """ConstraintConfig stores all fields."""
        config = ConstraintConfig(
            name="analytical",
            input_prefix="@analytical ",
            description="Step-by-step reasoning",
        )

        assert config.name == "analytical"
        assert config.input_prefix == "@analytical "
        assert config.description == "Step-by-step reasoning"

    def test_constraint_config_empty_prefix(self):
        """Baseline constraint has empty prefix."""
        config = ConstraintConfig(name="baseline", input_prefix="")

        assert config.input_prefix == ""


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_prompt_config_fields(self):
        """PromptConfig stores all fields."""
        config = PromptConfig(
            id="recursion",
            text="Explain recursion",
            category="technical",
            complexity="medium",
            notes="Classic example",
        )

        assert config.id == "recursion"
        assert config.text == "Explain recursion"
        assert config.category == "technical"
        assert config.complexity == "medium"
        assert config.notes == "Classic example"

    def test_prompt_config_defaults(self):
        """PromptConfig has sensible defaults."""
        config = PromptConfig(
            id="test",
            text="Test prompt",
            category="test",
        )

        assert config.complexity == "medium"
        assert config.notes == ""
