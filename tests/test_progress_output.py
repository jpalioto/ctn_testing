"""Tests for colored progress output."""

from ctn_testing.runners.evaluation import (
    GREEN,
    RED,
    RESET,
    ProgressInfo,
    _call_progress,
    format_status,
)


class TestFormatStatus:
    """Tests for format_status function."""

    def test_success_returns_green_ok(self):
        """Success returns green [ok]."""
        result = format_status(success=True)
        assert result == f"{GREEN}[ok]{RESET}"

    def test_failure_returns_red_error(self):
        """Failure returns red [error]."""
        result = format_status(success=False)
        assert result == f"{RED}[error]{RESET}"

    def test_green_contains_ansi_code(self):
        """GREEN constant contains expected ANSI escape code."""
        assert GREEN == "\033[92m"

    def test_red_contains_ansi_code(self):
        """RED constant contains expected ANSI escape code."""
        assert RED == "\033[91m"

    def test_reset_contains_ansi_code(self):
        """RESET constant contains expected ANSI escape code."""
        assert RESET == "\033[0m"


class TestCallProgress:
    """Tests for _call_progress helper function."""

    def test_calls_new_signature_with_progress_info(self):
        """Calls callback with ProgressInfo object."""
        calls = []

        def new_callback(info: ProgressInfo):
            calls.append(info)

        info = ProgressInfo(stage="running", current=1, total=10, success=True)
        _call_progress(new_callback, info)

        assert len(calls) == 1
        assert calls[0].stage == "running"
        assert calls[0].current == 1
        assert calls[0].total == 10
        assert calls[0].success is True

    def test_passes_error_message(self):
        """Passes error message in ProgressInfo."""
        calls = []

        def new_callback(info: ProgressInfo):
            calls.append(info)

        info = ProgressInfo(
            stage="judging", current=5, total=20, success=False, error_msg="API timeout"
        )
        _call_progress(new_callback, info)

        assert len(calls) == 1
        assert calls[0].error_msg == "API timeout"
        assert calls[0].success is False

    def test_backward_compatible_with_5arg_signature(self):
        """Falls back to 5-arg signature for legacy callbacks."""
        calls = []

        def old_callback(stage, current, total, success, error_msg):
            calls.append((stage, current, total, success, error_msg))

        info = ProgressInfo(stage="running", current=3, total=15, success=True)
        _call_progress(old_callback, info)

        assert len(calls) == 1
        assert calls[0] == ("running", 3, 15, True, None)

    def test_backward_compatible_with_3arg_signature(self):
        """Falls back to 3-arg signature for very old callbacks."""
        calls = []

        def old_callback(stage, current, total):
            calls.append((stage, current, total))

        info = ProgressInfo(
            stage="judging", current=2, total=10, success=False, error_msg="Some error"
        )
        _call_progress(old_callback, info)

        assert len(calls) == 1
        assert calls[0] == ("judging", 2, 10)

    def test_none_callback_does_nothing(self):
        """None callback is handled gracefully."""
        info = ProgressInfo(stage="running", current=1, total=10, success=True)
        # Should not raise
        _call_progress(None, info)

    def test_progress_info_has_constraint_details(self):
        """ProgressInfo can include constraint details."""
        info = ProgressInfo(
            stage="running",
            current=1,
            total=10,
            success=True,
            constraint_name="analytical",
            prompt_id="test_prompt",
            prompt_text="What is recursion?",
            duration_secs=1.5,
        )

        assert info.constraint_name == "analytical"
        assert info.prompt_id == "test_prompt"
        assert info.prompt_text == "What is recursion?"
        assert info.duration_secs == 1.5

    def test_progress_info_has_judging_details(self):
        """ProgressInfo can include judging details."""
        info = ProgressInfo(
            stage="judging",
            current=5,
            total=20,
            success=True,
            baseline_constraint="baseline",
            test_constraint="analytical",
            prompt_text="What is recursion?",
        )

        assert info.baseline_constraint == "baseline"
        assert info.test_constraint == "analytical"


class TestProgressCallbackIntegration:
    """Integration tests for progress callback with format_status."""

    def test_format_output_with_success(self):
        """Can format complete progress output with success."""
        stage = "running"
        current = 1
        total = 57
        success = True

        output = f"{stage}: {current}/{total} {format_status(success)}"

        assert output == f"running: 1/57 {GREEN}[ok]{RESET}"

    def test_format_output_with_error(self):
        """Can format complete progress output with error."""
        stage = "judging"
        current = 5
        total = 38
        success = False

        output = f"{stage}: {current}/{total} {format_status(success)}"

        assert output == f"judging: 5/38 {RED}[error]{RESET}"

    def test_callback_can_print_colored_output(self, capsys):
        """Callback can print colored output to terminal."""

        def print_callback(info: ProgressInfo):
            status = format_status(info.success)
            print(f"{info.stage}: {info.current}/{info.total} {status}")

        info1 = ProgressInfo(stage="running", current=1, total=57, success=True)
        info2 = ProgressInfo(stage="running", current=2, total=57, success=False, error_msg="fail")
        _call_progress(print_callback, info1)
        _call_progress(print_callback, info2)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        assert len(lines) == 2
        assert f"{GREEN}[ok]{RESET}" in lines[0]
        assert f"{RED}[error]{RESET}" in lines[1]
