"""Tests for blind judging infrastructure."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ctn_testing.judging.blind_judge import (
    BlindJudge,
    JudgingError,
    JudgingResult,
    TraitScore,
)
from ctn_testing.judging.traits import TraitDefinitions, TraitDimension, load_traits
from ctn_testing.runners.http_runner import SDKError, SDKResponse, SDKRunner

# Sample traits YAML content
SAMPLE_TRAITS_YAML = """
version: "1.0"

dimensions:
  - name: reasoning_depth
    description: "How much explicit step-by-step reasoning is present"
    scale: 0-100
    anchors:
      0: "No reasoning visible"
      50: "Moderate reasoning"
      100: "Rigorous step-by-step"

  - name: conciseness
    description: "How brief vs verbose the response is"
    scale: 0-100
    anchors:
      0: "Extremely verbose"
      50: "Average length"
      100: "Maximally concise"
"""


@pytest.fixture
def traits_file():
    """Create a temporary traits file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_TRAITS_YAML)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def mock_runner():
    """Create a mock SDK runner."""
    return Mock(spec=SDKRunner)


class TestTraitDimension:
    """Tests for TraitDimension dataclass."""

    def test_format_anchors(self):
        """Format anchors produces readable output."""
        dim = TraitDimension(
            name="test",
            description="Test dimension",
            scale="0-100",
            anchors={0: "Low", 50: "Medium", 100: "High"},
        )

        formatted = dim.format_anchors()

        assert "0: Low" in formatted
        assert "50: Medium" in formatted
        assert "100: High" in formatted

    def test_format_anchors_sorted(self):
        """Anchors are sorted by score."""
        dim = TraitDimension(
            name="test",
            description="Test",
            scale="0-100",
            anchors={100: "High", 0: "Low", 50: "Medium"},
        )

        formatted = dim.format_anchors()
        lines = formatted.split("\n")

        assert "0:" in lines[0]
        assert "50:" in lines[1]
        assert "100:" in lines[2]


class TestTraitDefinitions:
    """Tests for TraitDefinitions dataclass."""

    def test_dimension_names(self):
        """Get list of dimension names."""
        dims = TraitDefinitions(
            version="1.0",
            dimensions=[
                TraitDimension(name="a", description="", scale="", anchors={}),
                TraitDimension(name="b", description="", scale="", anchors={}),
            ],
        )

        assert dims.dimension_names() == ["a", "b"]

    def test_get_dimension(self):
        """Get dimension by name."""
        dim_a = TraitDimension(name="a", description="Desc A", scale="", anchors={})
        dims = TraitDefinitions(version="1.0", dimensions=[dim_a])

        assert dims.get_dimension("a") == dim_a
        assert dims.get_dimension("nonexistent") is None


class TestLoadTraits:
    """Tests for loading traits from YAML."""

    def test_load_traits_from_yaml(self, traits_file):
        """Load traits from valid YAML file."""
        traits = load_traits(traits_file)

        assert traits.version == "1.0"
        assert len(traits.dimensions) == 2
        assert traits.dimensions[0].name == "reasoning_depth"
        assert traits.dimensions[1].name == "conciseness"

    def test_load_traits_anchors_as_ints(self, traits_file):
        """Anchor keys are converted to integers."""
        traits = load_traits(traits_file)

        dim = traits.dimensions[0]
        assert all(isinstance(k, int) for k in dim.anchors.keys())
        assert 0 in dim.anchors
        assert 50 in dim.anchors
        assert 100 in dim.anchors

    def test_load_traits_file_not_found(self):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_traits(Path("/nonexistent/traits.yaml"))

    def test_load_traits_invalid_format(self):
        """Raise ValueError for invalid YAML structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("just a string, not a dict")
            f.flush()

            with pytest.raises(ValueError, match="expected dict"):
                load_traits(Path(f.name))

    def test_load_traits_missing_dimension_name(self):
        """Raise ValueError for dimension without name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: "1.0"
dimensions:
  - description: "Missing name"
    scale: 0-100
""")
            f.flush()

            with pytest.raises(ValueError, match="missing required 'name'"):
                load_traits(Path(f.name))


class TestBlindJudge:
    """Tests for BlindJudge class."""

    def test_build_judge_prompt_includes_dimensions(self, traits_file, mock_runner):
        """Judge prompt includes all trait dimensions."""
        judge = BlindJudge(traits_file, mock_runner)

        prompt = judge._build_judge_prompt(
            response_a="Response A text",
            response_b="Response B text",
            prompt_text="Original question",
        )

        # Check dimensions are included
        assert "reasoning_depth" in prompt
        assert "conciseness" in prompt
        assert "How much explicit step-by-step reasoning" in prompt
        assert "How brief vs verbose" in prompt

    def test_build_judge_prompt_includes_responses(self, traits_file, mock_runner):
        """Judge prompt includes both responses."""
        judge = BlindJudge(traits_file, mock_runner)

        prompt = judge._build_judge_prompt(
            response_a="This is response A",
            response_b="This is response B",
            prompt_text="Test prompt",
        )

        assert "This is response A" in prompt
        assert "This is response B" in prompt
        assert "RESPONSE A:" in prompt
        assert "RESPONSE B:" in prompt

    def test_build_judge_prompt_includes_original_prompt(self, traits_file, mock_runner):
        """Judge prompt includes the original prompt for context."""
        judge = BlindJudge(traits_file, mock_runner)

        prompt = judge._build_judge_prompt(
            response_a="A",
            response_b="B",
            prompt_text="Explain recursion",
        )

        assert "Explain recursion" in prompt
        assert "ORIGINAL PROMPT:" in prompt

    def test_build_judge_prompt_includes_anchors(self, traits_file, mock_runner):
        """Judge prompt includes trait anchors."""
        judge = BlindJudge(traits_file, mock_runner)

        prompt = judge._build_judge_prompt(
            response_a="A",
            response_b="B",
            prompt_text="Test",
        )

        assert "No reasoning visible" in prompt
        assert "Rigorous step-by-step" in prompt
        assert "Extremely verbose" in prompt
        assert "Maximally concise" in prompt

    def test_parse_valid_json_response(self, traits_file, mock_runner):
        """Parse valid JSON response into JudgingResult."""
        judge = BlindJudge(traits_file, mock_runner)

        raw_response = json.dumps(
            {
                "response_a": {
                    "reasoning_depth": {"score": 75, "reasons": ["Good logic", "Clear steps"]},
                    "conciseness": {"score": 60, "reasons": ["Slightly verbose"]},
                },
                "response_b": {
                    "reasoning_depth": {"score": 40, "reasons": ["Weak reasoning"]},
                    "conciseness": {"score": 85, "reasons": ["Very concise"]},
                },
            }
        )

        result = judge._parse_response(raw_response)

        assert result.error is None
        assert result.response_a_scores["reasoning_depth"].score == 75
        assert result.response_a_scores["conciseness"].score == 60
        assert result.response_b_scores["reasoning_depth"].score == 40
        assert result.response_b_scores["conciseness"].score == 85

    def test_parse_json_in_markdown_block(self, traits_file, mock_runner):
        """Parse JSON wrapped in markdown code block."""
        judge = BlindJudge(traits_file, mock_runner)

        raw_response = """Here's my evaluation:

```json
{
  "response_a": {
    "reasoning_depth": {"score": 80, "reasons": ["Strong"]},
    "conciseness": {"score": 70, "reasons": ["Good"]}
  },
  "response_b": {
    "reasoning_depth": {"score": 50, "reasons": ["OK"]},
    "conciseness": {"score": 90, "reasons": ["Brief"]}
  }
}
```
"""

        result = judge._parse_response(raw_response)

        assert result.error is None
        assert result.response_a_scores["reasoning_depth"].score == 80

    def test_parse_malformed_json(self, traits_file, mock_runner):
        """Handle malformed JSON gracefully."""
        judge = BlindJudge(traits_file, mock_runner)

        raw_response = '{"broken": json, not valid}'

        result = judge._parse_response(raw_response)

        assert result.error is not None
        assert "parse error" in result.error.lower()
        assert result.raw_response == raw_response

    def test_parse_missing_dimensions(self, traits_file, mock_runner):
        """Handle missing dimensions in response."""
        judge = BlindJudge(traits_file, mock_runner)

        # Only has reasoning_depth, missing conciseness
        raw_response = json.dumps(
            {
                "response_a": {
                    "reasoning_depth": {"score": 75, "reasons": ["Good"]},
                },
                "response_b": {
                    "reasoning_depth": {"score": 40, "reasons": ["Weak"]},
                },
            }
        )

        result = judge._parse_response(raw_response)

        assert result.error is None
        assert "reasoning_depth" in result.response_a_scores
        assert "conciseness" not in result.response_a_scores

    def test_parse_score_as_number_only(self, traits_file, mock_runner):
        """Handle scores provided as plain numbers."""
        judge = BlindJudge(traits_file, mock_runner)

        raw_response = json.dumps(
            {
                "response_a": {
                    "reasoning_depth": 75,  # Just a number
                    "conciseness": 60,
                },
                "response_b": {
                    "reasoning_depth": 40,
                    "conciseness": 85,
                },
            }
        )

        result = judge._parse_response(raw_response)

        assert result.response_a_scores["reasoning_depth"].score == 75
        assert result.response_a_scores["reasoning_depth"].reasons == []

    def test_parse_clamps_scores_to_range(self, traits_file, mock_runner):
        """Scores are clamped to 0-100 range."""
        judge = BlindJudge(traits_file, mock_runner)

        raw_response = json.dumps(
            {
                "response_a": {
                    "reasoning_depth": {"score": 150, "reasons": []},
                    "conciseness": {"score": -10, "reasons": []},
                },
                "response_b": {
                    "reasoning_depth": {"score": 50, "reasons": []},
                    "conciseness": {"score": 50, "reasons": []},
                },
            }
        )

        result = judge._parse_response(raw_response)

        assert result.response_a_scores["reasoning_depth"].score == 100
        assert result.response_a_scores["conciseness"].score == 0

    def test_parse_no_json_found(self, traits_file, mock_runner):
        """Handle response with no JSON."""
        judge = BlindJudge(traits_file, mock_runner)

        raw_response = "This is just plain text with no JSON at all."

        result = judge._parse_response(raw_response)

        assert result.error is not None
        assert "No valid JSON" in result.error

    def test_judge_calls_sdk_runner(self, traits_file, mock_runner):
        """Judge method calls SDK runner with correct parameters."""
        judge = BlindJudge(
            traits_file,
            mock_runner,
            judge_provider="anthropic",
            judge_model="claude-sonnet-4",
        )

        mock_runner.send.return_value = SDKResponse(
            output=json.dumps(
                {
                    "response_a": {"reasoning_depth": 50, "conciseness": 50},
                    "response_b": {"reasoning_depth": 50, "conciseness": 50},
                }
            ),
            provider="anthropic",
            model="claude-sonnet-4",
            tokens={"input": 100, "output": 50},
        )

        result = judge.judge(
            response_a="Response A",
            response_b="Response B",
            prompt_text="Test prompt",
        )

        mock_runner.send.assert_called_once()
        call_kwargs = mock_runner.send.call_args[1]
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["model"] == "claude-sonnet-4"

    def test_judge_raises_on_sdk_error(self, traits_file, mock_runner):
        """Judge raises JudgingError on SDK failure."""
        judge = BlindJudge(traits_file, mock_runner)

        mock_runner.send.side_effect = SDKError("Connection failed")

        with pytest.raises(JudgingError, match="SDK error"):
            judge.judge(
                response_a="A",
                response_b="B",
                prompt_text="Test",
            )


class TestTraitScore:
    """Tests for TraitScore dataclass."""

    def test_trait_score_fields(self):
        """TraitScore stores all fields correctly."""
        score = TraitScore(
            dimension="reasoning_depth",
            score=85,
            reasons=["Clear logic", "Well structured"],
        )

        assert score.dimension == "reasoning_depth"
        assert score.score == 85
        assert len(score.reasons) == 2


class TestJudgingResult:
    """Tests for JudgingResult dataclass."""

    def test_judging_result_defaults(self):
        """JudgingResult has sensible defaults."""
        result = JudgingResult()

        assert result.response_a_scores == {}
        assert result.response_b_scores == {}
        assert result.raw_response == ""
        assert result.error is None

    def test_judging_result_with_scores(self):
        """JudgingResult stores scores correctly."""
        score_a = TraitScore("test", 80, ["Good"])
        score_b = TraitScore("test", 60, ["OK"])

        result = JudgingResult(
            response_a_scores={"test": score_a},
            response_b_scores={"test": score_b},
            raw_response="raw json",
        )

        assert result.response_a_scores["test"].score == 80
        assert result.response_b_scores["test"].score == 60
