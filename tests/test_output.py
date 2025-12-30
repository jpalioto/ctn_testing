"""Tests for evaluation output management."""
import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from ctn_testing.runners.output import (
    PersistenceError,
    RunManifest,
    RunOutputManager,
    NullOutputManager,
)


@pytest.fixture
def mock_config():
    """Create a mock EvaluationConfig."""
    config = MagicMock()
    config.name = "test_evaluation"
    config.models = [{"provider": "anthropic", "name": "sonnet"}]
    config.judge_models = [{"provider": "anthropic", "name": "sonnet"}]
    config.constraints = [
        MagicMock(name="baseline"),
        MagicMock(name="analytical"),
    ]
    # Make constraints iterable and have .name attribute work properly
    config.constraints[0].name = "baseline"
    config.constraints[1].name = "analytical"
    config.test_constraints = [config.constraints[1]]
    config.traits_path = Path("/nonexistent/traits.yaml")
    config.prompts_path = Path("/nonexistent/prompts.yaml")
    return config


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "results"


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("name: test\n")
    return config_path


class TestRunManifest:
    """Tests for RunManifest dataclass."""

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        manifest = RunManifest(
            run_id="2025-01-15T10-30-00",
            started_at="2025-01-15T10:30:00",
            config_file="phase1.yaml",
            prompts_count=5,
            constraints=["baseline", "analytical"],
        )

        d = manifest.to_dict()

        assert d["run_id"] == "2025-01-15T10-30-00"
        assert d["started_at"] == "2025-01-15T10:30:00"
        assert d["config_file"] == "phase1.yaml"
        assert d["prompts_count"] == 5
        assert d["constraints"] == ["baseline", "analytical"]
        assert d["completed_at"] is None
        assert d["errors"] == []

    def test_from_dict(self):
        """Creates from dictionary correctly."""
        data = {
            "run_id": "2025-01-15T10-30-00",
            "started_at": "2025-01-15T10:30:00",
            "completed_at": "2025-01-15T10:35:00",
            "duration_seconds": 300.5,
            "config_file": "phase1.yaml",
            "prompts_count": 5,
            "constraints": ["baseline", "analytical"],
            "models": [{"name": "sonnet"}],
            "judge_models": [{"name": "sonnet"}],
            "total_sdk_calls": 10,
            "total_judge_calls": 5,
            "errors": ["some error"],
        }

        manifest = RunManifest.from_dict(data)

        assert manifest.run_id == "2025-01-15T10-30-00"
        assert manifest.completed_at == "2025-01-15T10:35:00"
        assert manifest.duration_seconds == 300.5
        assert manifest.errors == ["some error"]

    def test_defaults(self):
        """Has correct default values."""
        manifest = RunManifest(
            run_id="test",
            started_at="2025-01-01T00:00:00",
        )

        assert manifest.completed_at is None
        assert manifest.duration_seconds is None
        assert manifest.config_file == ""
        assert manifest.prompts_count == 0
        assert manifest.constraints == []
        assert manifest.models == []
        assert manifest.judge_models == []
        assert manifest.total_sdk_calls == 0
        assert manifest.total_judge_calls == 0
        assert manifest.errors == []


class TestRunOutputManager:
    """Tests for RunOutputManager."""

    def test_creates_timestamped_folder(self, temp_output_dir, mock_config, temp_config_file):
        """Creates run folder with timestamp format."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        # Check run_id format (YYYY-MM-DDTHH-MM-SS)
        assert len(manager.run_id) == 19
        assert manager.run_id[4] == "-"
        assert manager.run_id[7] == "-"
        assert manager.run_id[10] == "T"

        # Check directory was created
        assert manager.run_dir.exists()
        assert manager.run_dir.is_dir()
        assert manager.run_dir.parent == temp_output_dir

    def test_creates_config_subdirectory(self, temp_output_dir, mock_config, temp_config_file):
        """Creates config/ subdirectory."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        assert manager.config_dir.exists()
        assert manager.config_dir == manager.run_dir / "config"

    def test_copies_main_config(self, temp_output_dir, mock_config, temp_config_file):
        """Copies main config file to config/ directory."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        copied_config = manager.config_dir / temp_config_file.name
        assert copied_config.exists()
        assert copied_config.read_text() == "name: test\n"

    def test_copies_traits_file(self, tmp_path, temp_output_dir, temp_config_file):
        """Copies traits definition file if it exists."""
        # Create traits file
        traits_path = tmp_path / "traits.yaml"
        traits_path.write_text("dimensions: []\n")

        config = MagicMock()
        config.constraints = []
        config.test_constraints = []
        config.models = []
        config.judge_models = []
        config.traits_path = traits_path
        config.prompts_path = Path("/nonexistent/prompts.yaml")

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        copied_traits = manager.config_dir / "traits.yaml"
        assert copied_traits.exists()
        assert copied_traits.read_text() == "dimensions: []\n"

    def test_copies_prompts_file(self, tmp_path, temp_output_dir, temp_config_file):
        """Copies prompts file if it exists."""
        # Create prompts file
        prompts_path = tmp_path / "prompts.yaml"
        prompts_path.write_text("prompts: []\n")

        config = MagicMock()
        config.constraints = []
        config.test_constraints = []
        config.models = []
        config.judge_models = []
        config.traits_path = Path("/nonexistent/traits.yaml")
        config.prompts_path = prompts_path

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        copied_prompts = manager.config_dir / "prompts.yaml"
        assert copied_prompts.exists()
        assert copied_prompts.read_text() == "prompts: []\n"

    def test_creates_manifest_json(self, temp_output_dir, mock_config, temp_config_file):
        """Creates manifest.json with required fields."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        assert manager.manifest_path.exists()

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert "run_id" in manifest
        assert "started_at" in manifest
        assert manifest["config_file"] == temp_config_file.name
        assert manifest["prompts_count"] == 5
        assert manifest["constraints"] == ["baseline", "analytical"]
        assert manifest["models"] == mock_config.models
        assert manifest["judge_models"] == mock_config.judge_models

    def test_manifest_has_expected_call_counts(self, temp_output_dir, mock_config, temp_config_file):
        """Manifest contains expected SDK and judge call counts."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        # 5 prompts × 2 constraints = 10 SDK calls
        assert manifest["total_sdk_calls"] == 10
        # 5 prompts × 1 test constraint = 5 judge calls
        assert manifest["total_judge_calls"] == 5

    def test_finalize_updates_completed_at(self, temp_output_dir, mock_config, temp_config_file):
        """finalize() sets completed_at timestamp."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)
        manager.finalize()

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["completed_at"] is not None
        # Should be valid ISO format
        datetime.fromisoformat(manifest["completed_at"])

    def test_finalize_updates_duration(self, temp_output_dir, mock_config, temp_config_file):
        """finalize() calculates duration_seconds."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)
        manager.finalize()

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["duration_seconds"] is not None
        assert manifest["duration_seconds"] >= 0

    def test_finalize_records_errors(self, temp_output_dir, mock_config, temp_config_file):
        """finalize() records error list."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)
        errors = ["Error 1", "Error 2"]
        manager.finalize(errors=errors)

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert manifest["errors"] == ["Error 1", "Error 2"]

    def test_add_error(self, temp_output_dir, mock_config, temp_config_file):
        """add_error() appends to error list."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)
        manager.add_error("First error")
        manager.add_error("Second error")

        # Errors should be in manifest after finalize
        manager.finalize()

        with open(manager.manifest_path) as f:
            manifest = json.load(f)

        assert "First error" in manifest["errors"]
        assert "Second error" in manifest["errors"]

    def test_finalize_without_initialize(self, temp_output_dir, mock_config, temp_config_file):
        """finalize() is no-op if initialize() wasn't called."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        # Should not raise
        manager.finalize()

        # No directories should be created
        assert not temp_output_dir.exists()

    def test_handles_missing_source_files(self, temp_output_dir, temp_config_file):
        """Gracefully handles missing traits/prompts files."""
        config = MagicMock()
        config.constraints = []
        config.test_constraints = []
        config.models = []
        config.judge_models = []
        config.traits_path = Path("/nonexistent/traits.yaml")
        config.prompts_path = Path("/nonexistent/prompts.yaml")

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=config,
            config_path=temp_config_file,
        )

        # Should not raise even though source files don't exist
        manager.initialize(prompts_count=5)

        # Config dir should still be created
        assert manager.config_dir.exists()
        # Main config should be copied
        assert (manager.config_dir / temp_config_file.name).exists()


class TestNullOutputManager:
    """Tests for NullOutputManager (no-op implementation)."""

    def test_has_none_run_id(self):
        """run_id is 'none'."""
        manager = NullOutputManager()
        assert manager.run_id == "none"

    def test_initialize_is_noop(self):
        """initialize() does nothing."""
        manager = NullOutputManager()
        manager.initialize(prompts_count=100)
        # Should not raise, and run_dir should not exist
        assert not manager.run_dir.exists()

    def test_finalize_is_noop(self):
        """finalize() does nothing."""
        manager = NullOutputManager()
        manager.finalize(errors=["error"])
        # Should not raise

    def test_add_error_is_noop(self):
        """add_error() does nothing."""
        manager = NullOutputManager()
        manager.add_error("error")
        # Should not raise


class TestOutputManagerIntegration:
    """Integration tests for output manager with evaluation config."""

    def test_multiple_runs_create_separate_folders(self, temp_output_dir, mock_config, temp_config_file):
        """Multiple runs create separate timestamped folders."""
        # Use mocked timestamps to ensure different run IDs
        with patch("ctn_testing.runners.output.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2025-01-15T10-30-00"
            mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T10:30:00"
            manager1 = RunOutputManager(
                base_dir=temp_output_dir,
                config=mock_config,
                config_path=temp_config_file,
            )
            manager1.initialize(prompts_count=5)

        with patch("ctn_testing.runners.output.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2025-01-15T10-31-00"
            mock_datetime.now.return_value.isoformat.return_value = "2025-01-15T10:31:00"
            manager2 = RunOutputManager(
                base_dir=temp_output_dir,
                config=mock_config,
                config_path=temp_config_file,
            )
            manager2.initialize(prompts_count=5)

        # Different run directories
        assert manager1.run_dir != manager2.run_dir
        assert manager1.run_dir.exists()
        assert manager2.run_dir.exists()

    def test_manifest_roundtrip(self, temp_output_dir, mock_config, temp_config_file):
        """Manifest can be saved and loaded correctly."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        manager.initialize(prompts_count=5)
        manager.add_error("Test error")
        manager.finalize()

        # Load manifest
        with open(manager.manifest_path) as f:
            data = json.load(f)

        loaded = RunManifest.from_dict(data)

        assert loaded.run_id == manager.run_id
        assert loaded.prompts_count == 5
        assert "Test error" in loaded.errors
        assert loaded.completed_at is not None
        assert loaded.duration_seconds is not None


class TestFailFastValidation:
    """Tests for fail-fast persistence validation."""

    def test_raises_on_directory_creation_failure(self, mock_config, temp_config_file):
        """Raises PersistenceError if directory cannot be created."""
        manager = RunOutputManager(
            base_dir=Path("/some/path"),
            config=mock_config,
            config_path=temp_config_file,
        )

        # Mock mkdir to fail
        with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(PersistenceError) as exc_info:
                manager.initialize(prompts_count=5)

            assert "Cannot create" in str(exc_info.value)

    def test_raises_on_write_permission_denied(self, temp_output_dir, mock_config, temp_config_file):
        """Raises PersistenceError if directory is not writable."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        # Mock write_text to simulate permission denied
        with patch.object(Path, "write_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(PersistenceError) as exc_info:
                manager.initialize(prompts_count=5)

            assert "Cannot write" in str(exc_info.value)

    def test_raises_on_disk_full_mock(self, temp_output_dir, mock_config, temp_config_file):
        """Raises PersistenceError when disk is full (mocked)."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        # Mock write_text to simulate disk full
        with patch.object(Path, "write_text", side_effect=OSError("No space left on device")):
            with pytest.raises(PersistenceError) as exc_info:
                manager.initialize(prompts_count=5)

            assert "Cannot write" in str(exc_info.value)

    def test_raises_on_manifest_write_failure(self, temp_output_dir, mock_config, temp_config_file):
        """Raises PersistenceError if manifest cannot be written."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        # Mock json.dump to fail during manifest write
        original_dump = json.dump
        call_count = [0]

        def failing_dump(*args, **kwargs):
            call_count[0] += 1
            raise OSError("Disk full")

        with patch("json.dump", side_effect=failing_dump):
            with pytest.raises(PersistenceError) as exc_info:
                manager.initialize(prompts_count=5)

            assert "Cannot write manifest" in str(exc_info.value)

    def test_error_message_includes_path(self, temp_output_dir, mock_config, temp_config_file):
        """Error message includes the problematic path."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )

        # Mock mkdir to fail with a path-specific error
        def mock_mkdir(*args, **kwargs):
            raise OSError(f"Cannot create {manager.run_dir}")

        with patch.object(Path, "mkdir", side_effect=mock_mkdir):
            with pytest.raises(PersistenceError) as exc_info:
                manager.initialize(prompts_count=5)

            error_msg = str(exc_info.value)
            # Should mention the path or "Cannot"
            assert str(temp_output_dir) in error_msg or "Cannot" in error_msg

    def test_validate_writable_raises_on_failure(self, tmp_path, mock_config, temp_config_file):
        """validate_writable() raises PersistenceError on write failure."""
        manager = RunOutputManager(
            base_dir=tmp_path,
            config=mock_config,
            config_path=temp_config_file,
        )

        # Manually create run_dir so validate_writable can be called
        manager.run_dir.mkdir(parents=True, exist_ok=True)

        # Mock write_text to fail
        with patch.object(Path, "write_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(PersistenceError) as exc_info:
                manager.validate_writable()

            assert "Cannot write to" in str(exc_info.value)

    def test_no_sdk_calls_if_persistence_fails(self, tmp_path, mock_config, temp_config_file):
        """Evaluator run() aborts before SDK calls if persistence fails."""
        from ctn_testing.runners.evaluation import ConstraintEvaluator

        # Create evaluator with mocked dependencies
        with patch("ctn_testing.runners.evaluation.load_config") as mock_load:
            mock_load.return_value = mock_config
            mock_config.runner = {"base_url": "http://localhost:14380"}
            mock_config.prompts_source = "prompts.yaml"
            mock_config.prompts_include_ids = None
            mock_config.prompts_path = temp_config_file
            mock_config.traits_path = tmp_path / "traits.yaml"
            mock_config.output = {"dir": "results/"}
            mock_config.config_dir = temp_config_file.parent

            # Mock load_prompts to return empty list
            with patch("ctn_testing.runners.evaluation.load_prompts", return_value=[]):
                # Mock BlindJudge to avoid loading traits file
                with patch("ctn_testing.runners.evaluation.BlindJudge"):
                    # Mock SDKRunner to track if it's called
                    with patch("ctn_testing.runners.evaluation.SDKRunner") as mock_sdk:
                        evaluator = ConstraintEvaluator(
                            config_path=temp_config_file,
                        )

                        # Now make the output manager fail during run()
                        evaluator._output_manager.initialize = MagicMock(
                            side_effect=PersistenceError("Cannot persist results")
                        )

                        # run() should raise PersistenceError before any SDK calls
                        with pytest.raises(PersistenceError) as exc_info:
                            evaluator.run()

                        assert "Cannot persist" in str(exc_info.value)

                        # ConstraintRunner.run_prompt should never be called
                        # (SDK calls happen through ConstraintRunner, not directly)
                        assert not mock_sdk.return_value.send.called


class TestResponsePersistence:
    """Tests for saving individual response files."""

    @pytest.fixture
    def mock_run_result(self):
        """Create a mock RunResult."""
        from ctn_testing.runners.constraint_runner import RunResult
        return RunResult(
            prompt_id="recursion",
            constraint_name="analytical",
            input_sent="@analytical Explain recursion",
            output="Recursion is a technique where a function calls itself...",
            provider="anthropic",
            model="sonnet",
            tokens={"input": 22, "output": 423},
            timestamp="2025-12-30T12:45:05.000Z",
        )

    def test_save_response_creates_file(self, temp_output_dir, mock_config, temp_config_file, mock_run_result):
        """save_response creates response file in responses/ directory."""
        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_response(mock_run_result, prompt_text="Explain recursion")

        # Check file exists
        expected_path = manager.responses_dir / "recursion_analytical.json"
        assert expected_path.exists()

    def test_save_response_filename_format(self, temp_output_dir, mock_config, temp_config_file):
        """Filename follows {prompt_id}_{constraint_name}.json format."""
        from ctn_testing.runners.constraint_runner import RunResult

        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        run_result = RunResult(
            prompt_id="my_prompt",
            constraint_name="terse",
            input_sent="@terse Test",
            output="Result",
            provider="anthropic",
            model="sonnet",
            tokens={"input": 5, "output": 10},
            timestamp="2025-01-01T00:00:00Z",
        )

        manager.save_response(run_result, prompt_text="Test")

        expected_path = manager.responses_dir / "my_prompt_terse.json"
        assert expected_path.exists()

    def test_save_response_contains_all_fields(self, temp_output_dir, mock_config, temp_config_file, mock_run_result):
        """Response file contains all required fields."""
        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_response(mock_run_result, prompt_text="Explain recursion")

        response_path = manager.responses_dir / "recursion_analytical.json"
        with open(response_path) as f:
            data = json.load(f)

        assert data["prompt_id"] == "recursion"
        assert data["prompt_text"] == "Explain recursion"
        assert data["constraint_name"] == "analytical"
        assert data["input_sent"] == "@analytical Explain recursion"
        assert data["output"] == "Recursion is a technique where a function calls itself..."
        assert data["provider"] == "anthropic"
        assert data["model"] == "sonnet"
        assert data["tokens"] == {"input": 22, "output": 423}
        assert data["timestamp"] == "2025-12-30T12:45:05.000Z"

    def test_save_response_include_raw_false_nullifies_output(self, temp_output_dir, mock_config, temp_config_file, mock_run_result):
        """When include_raw_responses is false, output is null."""
        mock_config.output = {"include_raw_responses": False}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_response(mock_run_result, prompt_text="Explain recursion")

        response_path = manager.responses_dir / "recursion_analytical.json"
        with open(response_path) as f:
            data = json.load(f)

        # Output should be null
        assert data["output"] is None
        # But metadata should still be present
        assert data["prompt_id"] == "recursion"
        assert data["tokens"] == {"input": 22, "output": 423}

    def test_save_response_includes_error_if_present(self, temp_output_dir, mock_config, temp_config_file):
        """Response file includes error field if run had an error."""
        from ctn_testing.runners.constraint_runner import RunResult

        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        run_result = RunResult(
            prompt_id="test",
            constraint_name="baseline",
            input_sent="Test prompt",
            output="",
            provider="anthropic",
            model="sonnet",
            tokens={"input": 0, "output": 0},
            timestamp="2025-01-01T00:00:00Z",
            error="SDK connection failed",
        )

        manager.save_response(run_result, prompt_text="Test prompt")

        response_path = manager.responses_dir / "test_baseline.json"
        with open(response_path) as f:
            data = json.load(f)

        assert data["error"] == "SDK connection failed"

    def test_save_response_raises_on_write_failure(self, temp_output_dir, mock_config, temp_config_file, mock_run_result):
        """save_response raises PersistenceError on write failure."""
        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        # Mock tempfile.mkstemp to fail
        with patch("ctn_testing.runners.output.tempfile.mkstemp", side_effect=OSError("Disk full")):
            with pytest.raises(PersistenceError) as exc_info:
                manager.save_response(mock_run_result, prompt_text="Test")

            assert "Cannot save response" in str(exc_info.value)

    def test_save_response_atomic_no_partial_files(self, temp_output_dir, mock_config, temp_config_file, mock_run_result):
        """On write failure, no partial files are left behind."""
        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        # Mock os.replace to fail after temp file is written
        original_replace = os.replace
        def failing_replace(src, dst):
            raise OSError("Replace failed")

        with patch("ctn_testing.runners.output.os.replace", side_effect=failing_replace):
            with pytest.raises(PersistenceError):
                manager.save_response(mock_run_result, prompt_text="Test")

        # Verify no partial files (including temp files)
        response_files = list(manager.responses_dir.glob("*"))
        # Filter out any directories
        response_files = [f for f in response_files if f.is_file()]
        assert len(response_files) == 0, f"Found unexpected files: {response_files}"

    def test_responses_dir_created_on_initialize(self, temp_output_dir, mock_config, temp_config_file):
        """responses/ directory is created during initialization."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        assert manager.responses_dir.exists()
        assert manager.responses_dir.is_dir()

    def test_null_output_manager_save_response_noop(self, mock_run_result):
        """NullOutputManager.save_response is a no-op."""
        manager = NullOutputManager()

        # Should not raise
        manager.save_response(mock_run_result, prompt_text="Test")

    def test_multiple_responses_saved(self, temp_output_dir, mock_config, temp_config_file):
        """Multiple responses can be saved for different prompt/constraint combos."""
        from ctn_testing.runners.constraint_runner import RunResult

        mock_config.output = {"include_raw_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=2)

        # Save multiple responses
        combos = [
            ("prompt1", "baseline"),
            ("prompt1", "analytical"),
            ("prompt2", "baseline"),
            ("prompt2", "analytical"),
        ]

        for prompt_id, constraint_name in combos:
            run_result = RunResult(
                prompt_id=prompt_id,
                constraint_name=constraint_name,
                input_sent=f"@{constraint_name} Test",
                output="Response",
                provider="anthropic",
                model="sonnet",
                tokens={"input": 5, "output": 10},
                timestamp="2025-01-01T00:00:00Z",
            )
            manager.save_response(run_result, prompt_text="Test")

        # Verify all files created
        for prompt_id, constraint_name in combos:
            path = manager.responses_dir / f"{prompt_id}_{constraint_name}.json"
            assert path.exists(), f"Missing file: {path}"


class TestJudgingPersistence:
    """Tests for saving individual judging result files."""

    @pytest.fixture
    def mock_comparison(self):
        """Create a mock PairedComparison."""
        from ctn_testing.runners.evaluation import PairedComparison
        from ctn_testing.judging.blind_judge import JudgingResult, TraitScore

        judging_result = JudgingResult(
            response_a_scores={
                "reasoning_depth": TraitScore(dimension="reasoning_depth", score=65, reasons=["Good structure"]),
                "conciseness": TraitScore(dimension="conciseness", score=78, reasons=["Well organized"]),
            },
            response_b_scores={
                "reasoning_depth": TraitScore(dimension="reasoning_depth", score=82, reasons=["Excellent depth"]),
                "conciseness": TraitScore(dimension="conciseness", score=71, reasons=["Slightly verbose"]),
            },
            raw_response='{"scores": {...}}',
        )

        return PairedComparison(
            prompt_id="recursion",
            prompt_text="Explain recursion",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="Baseline response text...",
            test_response="Test response with analytical constraint...",
            judging_result=judging_result,
            baseline_was_a=True,
        )

    def test_save_judging_creates_file(self, temp_output_dir, mock_config, temp_config_file, mock_comparison):
        """save_judging creates judging file in judging/ directory."""
        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_judging(
            mock_comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-12-30T12:50:15.000Z",
        )

        expected_path = manager.judging_dir / "recursion_analytical_vs_baseline.json"
        assert expected_path.exists()

    def test_save_judging_filename_format(self, temp_output_dir, mock_config, temp_config_file):
        """Filename follows {prompt_id}_{test}_vs_{baseline}.json format."""
        from ctn_testing.runners.evaluation import PairedComparison
        from ctn_testing.judging.blind_judge import JudgingResult, TraitScore

        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        comparison = PairedComparison(
            prompt_id="my_prompt",
            prompt_text="Test",
            baseline_constraint="control",
            test_constraint="terse",
            baseline_response="baseline",
            test_response="test",
            judging_result=JudgingResult(
                response_a_scores={"trait": TraitScore("trait", 50, [])},
                response_b_scores={"trait": TraitScore("trait", 60, [])},
            ),
            baseline_was_a=True,
        )

        manager.save_judging(
            comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-01-01T00:00:00Z",
        )

        expected_path = manager.judging_dir / "my_prompt_terse_vs_control.json"
        assert expected_path.exists()

    def test_save_judging_contains_all_fields(self, temp_output_dir, mock_config, temp_config_file, mock_comparison):
        """Judging file contains all required fields."""
        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_judging(
            mock_comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-12-30T12:50:15.000Z",
        )

        judging_path = manager.judging_dir / "recursion_analytical_vs_baseline.json"
        with open(judging_path) as f:
            data = json.load(f)

        assert data["prompt_id"] == "recursion"
        assert data["prompt_text"] == "Explain recursion"
        assert data["baseline_constraint"] == "baseline"
        assert data["test_constraint"] == "analytical"
        assert data["baseline_was_a"] is True
        assert data["baseline_response"] == "Baseline response text..."
        assert data["test_response"] == "Test response with analytical constraint..."
        assert data["judge_model"] == {"provider": "anthropic", "name": "sonnet"}
        assert data["judge_raw_response"] == '{"scores": {...}}'
        assert data["timestamp"] == "2025-12-30T12:50:15.000Z"

        # Check scores structure
        assert "baseline" in data["scores"]
        assert "test" in data["scores"]
        assert data["scores"]["baseline"]["reasoning_depth"]["score"] == 65
        assert data["scores"]["baseline"]["reasoning_depth"]["reasons"] == ["Good structure"]
        assert data["scores"]["test"]["reasoning_depth"]["score"] == 82

    def test_save_judging_include_judge_responses_false(self, temp_output_dir, mock_config, temp_config_file, mock_comparison):
        """When include_judge_responses is false, judge_raw_response is null."""
        mock_config.output = {"include_raw_responses": True, "include_judge_responses": False}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_judging(
            mock_comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-01-01T00:00:00Z",
        )

        judging_path = manager.judging_dir / "recursion_analytical_vs_baseline.json"
        with open(judging_path) as f:
            data = json.load(f)

        assert data["judge_raw_response"] is None
        # But scores should still be present
        assert data["scores"]["baseline"]["reasoning_depth"]["score"] == 65

    def test_save_judging_include_raw_responses_false(self, temp_output_dir, mock_config, temp_config_file, mock_comparison):
        """When include_raw_responses is false, baseline_response and test_response are null."""
        mock_config.output = {"include_raw_responses": False, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        manager.save_judging(
            mock_comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-01-01T00:00:00Z",
        )

        judging_path = manager.judging_dir / "recursion_analytical_vs_baseline.json"
        with open(judging_path) as f:
            data = json.load(f)

        assert data["baseline_response"] is None
        assert data["test_response"] is None
        # But scores and judge response should still be present
        assert data["scores"]["baseline"]["reasoning_depth"]["score"] == 65
        assert data["judge_raw_response"] == '{"scores": {...}}'

    def test_save_judging_includes_error_if_present(self, temp_output_dir, mock_config, temp_config_file):
        """Judging file includes error field if comparison had an error."""
        from ctn_testing.runners.evaluation import PairedComparison
        from ctn_testing.judging.blind_judge import JudgingResult, TraitScore

        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        comparison = PairedComparison(
            prompt_id="test",
            prompt_text="Test prompt",
            baseline_constraint="baseline",
            test_constraint="analytical",
            baseline_response="",
            test_response="",
            judging_result=JudgingResult(
                response_a_scores={},
                response_b_scores={},
                error="Parse error",
            ),
            baseline_was_a=True,
            error="Judge failed to parse response",
        )

        manager.save_judging(
            comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-01-01T00:00:00Z",
        )

        judging_path = manager.judging_dir / "test_analytical_vs_baseline.json"
        with open(judging_path) as f:
            data = json.load(f)

        assert data["error"] == "Judge failed to parse response"

    def test_save_judging_raises_on_write_failure(self, temp_output_dir, mock_config, temp_config_file, mock_comparison):
        """save_judging raises PersistenceError on write failure."""
        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        with patch("ctn_testing.runners.output.tempfile.mkstemp", side_effect=OSError("Disk full")):
            with pytest.raises(PersistenceError) as exc_info:
                manager.save_judging(
                    mock_comparison,
                    judge_model={"provider": "anthropic", "name": "sonnet"},
                    timestamp="2025-01-01T00:00:00Z",
                )

            assert "Cannot save judging result" in str(exc_info.value)

    def test_save_judging_atomic_no_partial_files(self, temp_output_dir, mock_config, temp_config_file, mock_comparison):
        """On write failure, no partial files are left behind."""
        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        with patch("ctn_testing.runners.output.os.replace", side_effect=OSError("Replace failed")):
            with pytest.raises(PersistenceError):
                manager.save_judging(
                    mock_comparison,
                    judge_model={"provider": "anthropic", "name": "sonnet"},
                    timestamp="2025-01-01T00:00:00Z",
                )

        # Verify no partial files
        judging_files = list(manager.judging_dir.glob("*"))
        judging_files = [f for f in judging_files if f.is_file()]
        assert len(judging_files) == 0

    def test_judging_dir_created_on_initialize(self, temp_output_dir, mock_config, temp_config_file):
        """judging/ directory is created during initialization."""
        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=1)

        assert manager.judging_dir.exists()
        assert manager.judging_dir.is_dir()

    def test_null_output_manager_save_judging_noop(self, mock_comparison):
        """NullOutputManager.save_judging is a no-op."""
        manager = NullOutputManager()

        # Should not raise
        manager.save_judging(
            mock_comparison,
            judge_model={"provider": "anthropic", "name": "sonnet"},
            timestamp="2025-01-01T00:00:00Z",
        )

    def test_multiple_judging_results_saved(self, temp_output_dir, mock_config, temp_config_file):
        """Multiple judging results can be saved for different comparisons."""
        from ctn_testing.runners.evaluation import PairedComparison
        from ctn_testing.judging.blind_judge import JudgingResult, TraitScore

        mock_config.output = {"include_raw_responses": True, "include_judge_responses": True}

        manager = RunOutputManager(
            base_dir=temp_output_dir,
            config=mock_config,
            config_path=temp_config_file,
        )
        manager.initialize(prompts_count=2)

        # Save multiple judging results
        combos = [
            ("prompt1", "analytical", "baseline"),
            ("prompt1", "terse", "baseline"),
            ("prompt2", "analytical", "baseline"),
            ("prompt2", "terse", "baseline"),
        ]

        for prompt_id, test_constraint, baseline_constraint in combos:
            comparison = PairedComparison(
                prompt_id=prompt_id,
                prompt_text=f"Prompt {prompt_id}",
                baseline_constraint=baseline_constraint,
                test_constraint=test_constraint,
                baseline_response="baseline",
                test_response="test",
                judging_result=JudgingResult(
                    response_a_scores={"trait": TraitScore("trait", 50, [])},
                    response_b_scores={"trait": TraitScore("trait", 60, [])},
                ),
                baseline_was_a=True,
            )
            manager.save_judging(
                comparison,
                judge_model={"provider": "anthropic", "name": "sonnet"},
                timestamp="2025-01-01T00:00:00Z",
            )

        # Verify all files created
        for prompt_id, test_constraint, baseline_constraint in combos:
            path = manager.judging_dir / f"{prompt_id}_{test_constraint}_vs_{baseline_constraint}.json"
            assert path.exists(), f"Missing file: {path}"
