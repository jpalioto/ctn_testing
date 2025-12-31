"""Tests for browser data loading utilities."""

import json
from datetime import datetime

from ctn_testing.browser.data import (
    ResponseData,
    detect_strategy,
    extract_kernel,
    get_unique_constraints,
    get_unique_prompts,
    list_runs,
    load_judgings,
    load_responses,
    parse_run_timestamp,
)


class TestParseRunTimestamp:
    """Tests for parse_run_timestamp."""

    def test_parses_timestamp_with_microseconds(self):
        """Parses timestamp with microseconds format."""
        result = parse_run_timestamp("2025-01-15T10-30-45-123456")
        assert result == datetime(2025, 1, 15, 10, 30, 45, 123456)

    def test_parses_timestamp_without_microseconds(self):
        """Parses timestamp without microseconds format."""
        result = parse_run_timestamp("2025-01-15T10-30-45")
        assert result == datetime(2025, 1, 15, 10, 30, 45)

    def test_returns_none_for_invalid_format(self):
        """Returns None for invalid format."""
        assert parse_run_timestamp("invalid") is None
        assert parse_run_timestamp("2025-01-15") is None


class TestExtractKernel:
    """Tests for kernel extraction from input_sent."""

    def test_extracts_ctn_kernel(self):
        """Extracts CTN kernel schema."""
        input_sent = '''
        CTN_KERNEL_SCHEMA = """
        name: test_kernel
        constraints:
          - be helpful
        """

        User: Hello
        '''
        kernel_type, content = extract_kernel(input_sent)
        assert kernel_type == "ctn"
        assert "name: test_kernel" in content
        assert "be helpful" in content

    def test_extracts_structural_constraints(self):
        """Extracts XML-style structural constraints."""
        input_sent = """
        <constraints>
        <rule>Be concise</rule>
        <rule>Be accurate</rule>
        </constraints>

        User: Hello
        """
        kernel_type, content = extract_kernel(input_sent)
        assert kernel_type == "structural"
        assert "Be concise" in content

    def test_extracts_constraint_prefix(self):
        """Extracts constraint from @prefix."""
        input_sent = "@analytical What is the capital of France?"
        kernel_type, content = extract_kernel(input_sent)
        assert kernel_type == "analytical"
        assert "What is the capital of France?" in content

    def test_returns_raw_for_unknown_format(self):
        """Returns raw input for unknown format."""
        input_sent = "Just a plain question?"
        kernel_type, content = extract_kernel(input_sent)
        assert kernel_type == "raw"
        assert content == input_sent


class TestDetectStrategy:
    """Tests for strategy detection."""

    def test_detects_ctn_from_response_content(self):
        """Detects CTN strategy from response content."""
        responses = [
            ResponseData(
                prompt_id="p1",
                prompt_text="test",
                constraint_name="test",
                input_sent='CTN_KERNEL_SCHEMA = """..."""',
                output="response",
                tokens_in=10,
                tokens_out=20,
                error=None,
            )
        ]

        strategy = detect_strategy({}, responses)
        assert strategy == "ctn"

    def test_detects_structural_from_response_content(self):
        """Detects structural strategy from response content."""
        responses = [
            ResponseData(
                prompt_id="p1",
                prompt_text="test",
                constraint_name="test",
                input_sent="<constraints>...</constraints>",
                output="response",
                tokens_in=10,
                tokens_out=20,
                error=None,
            )
        ]

        strategy = detect_strategy({}, responses)
        assert strategy == "structural"

    def test_returns_none_when_no_responses(self):
        """Returns None when no responses."""
        strategy = detect_strategy({}, [])
        assert strategy is None


class TestListRuns:
    """Tests for list_runs function."""

    def test_lists_runs_from_directory(self, tmp_path):
        """Lists runs from a results directory."""
        # Create two run directories
        run1 = tmp_path / "2025-01-15T10-30-00-000000"
        run1.mkdir()
        (run1 / "manifest.json").write_text(
            json.dumps(
                {
                    "config_file": "test.yaml",
                    "prompts_count": 5,
                    "constraints": ["baseline", "analytical"],
                    "total_sdk_calls": 10,
                    "total_judge_calls": 5,
                    "errors": [],
                }
            )
        )
        (run1 / "responses").mkdir()

        run2 = tmp_path / "2025-01-16T11-00-00-000000"
        run2.mkdir()
        (run2 / "manifest.json").write_text(
            json.dumps(
                {
                    "config_file": "test2.yaml",
                    "prompts_count": 3,
                    "constraints": ["baseline"],
                    "total_sdk_calls": 3,
                    "total_judge_calls": 0,
                    "errors": ["some error"],
                }
            )
        )

        runs = list_runs(tmp_path)

        assert len(runs) == 2
        # Should be sorted by date descending
        assert runs[0].run_id == "2025-01-16T11-00-00-000000"
        assert runs[1].run_id == "2025-01-15T10-30-00-000000"
        assert runs[0].prompts_count == 3
        assert runs[1].prompts_count == 5

    def test_returns_empty_list_for_nonexistent_directory(self, tmp_path):
        """Returns empty list for nonexistent directory."""
        runs = list_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_skips_directories_without_manifest(self, tmp_path):
        """Skips directories without manifest.json."""
        run1 = tmp_path / "2025-01-15T10-30-00-000000"
        run1.mkdir()
        # No manifest.json

        run2 = tmp_path / "2025-01-16T11-00-00-000000"
        run2.mkdir()
        (run2 / "manifest.json").write_text(
            json.dumps(
                {
                    "config_file": "test.yaml",
                    "prompts_count": 5,
                    "constraints": [],
                    "total_sdk_calls": 0,
                    "total_judge_calls": 0,
                    "errors": [],
                }
            )
        )

        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0].run_id == "2025-01-16T11-00-00-000000"


class TestLoadResponses:
    """Tests for load_responses function."""

    def test_loads_responses_from_run(self, tmp_path):
        """Loads responses from a run directory."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        responses_dir = run_dir / "responses"
        responses_dir.mkdir()

        (responses_dir / "p1_baseline.json").write_text(
            json.dumps(
                {
                    "prompt_id": "p1",
                    "prompt_text": "What is 2+2?",
                    "constraint_name": "baseline",
                    "input_sent": "What is 2+2?",
                    "output": "4",
                    "tokens": {"input": 5, "output": 1},
                }
            )
        )

        (responses_dir / "p1_analytical.json").write_text(
            json.dumps(
                {
                    "prompt_id": "p1",
                    "prompt_text": "What is 2+2?",
                    "constraint_name": "analytical",
                    "input_sent": "@analytical What is 2+2?",
                    "output": "The answer is 4",
                    "tokens": {"input": 6, "output": 3},
                }
            )
        )

        responses = load_responses(run_dir)

        assert len(responses) == 2
        assert responses[0].prompt_id == "p1"
        assert responses[0].constraint_name in ["baseline", "analytical"]

    def test_returns_empty_list_for_missing_responses_dir(self, tmp_path):
        """Returns empty list when responses dir doesn't exist."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{}")

        responses = load_responses(run_dir)
        assert responses == []


class TestLoadJudgings:
    """Tests for load_judgings function."""

    def test_loads_judgings_from_run(self, tmp_path):
        """Loads judging results from a run directory."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        judging_dir = run_dir / "judging"
        judging_dir.mkdir()

        (judging_dir / "p1_analytical.json").write_text(
            json.dumps(
                {
                    "prompt_id": "p1",
                    "prompt_text": "What is 2+2?",
                    "test_constraint": "analytical",
                    "baseline_constraint": "baseline",
                    "baseline_response": "4",
                    "test_response": "The answer is 4",
                    "baseline_was_a": True,
                    "response_a_scores": {"clarity": {"score": 70, "reasons": ["clear"]}},
                    "response_b_scores": {"clarity": {"score": 85, "reasons": ["very clear"]}},
                }
            )
        )

        judgings = load_judgings(run_dir)

        assert len(judgings) == 1
        assert judgings[0].prompt_id == "p1"
        assert judgings[0].test_constraint == "analytical"
        assert judgings[0].baseline_scores["clarity"]["score"] == 70
        assert judgings[0].test_scores["clarity"]["score"] == 85


class TestGetUniquePrompts:
    """Tests for get_unique_prompts function."""

    def test_returns_unique_prompts(self):
        """Returns unique prompt id/text pairs."""
        responses = [
            ResponseData("p1", "Question 1", "baseline", "", "", 0, 0, None),
            ResponseData("p1", "Question 1", "analytical", "", "", 0, 0, None),
            ResponseData("p2", "Question 2", "baseline", "", "", 0, 0, None),
        ]

        prompts = get_unique_prompts(responses)

        assert len(prompts) == 2
        assert ("p1", "Question 1") in prompts
        assert ("p2", "Question 2") in prompts


class TestGetUniqueConstraints:
    """Tests for get_unique_constraints function."""

    def test_returns_unique_constraints(self):
        """Returns unique constraint names sorted."""
        responses = [
            ResponseData("p1", "", "baseline", "", "", 0, 0, None),
            ResponseData("p1", "", "analytical", "", "", 0, 0, None),
            ResponseData("p2", "", "baseline", "", "", 0, 0, None),
            ResponseData("p2", "", "structured", "", "", 0, 0, None),
        ]

        constraints = get_unique_constraints(responses)

        assert constraints == ["analytical", "baseline", "structured"]
