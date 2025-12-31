"""Tests for colored progress output."""

from ctn_testing.runners.evaluation import (
    GREEN,
    RED,
    RESET,
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

    def test_calls_new_signature_with_all_args(self):
        """Calls callback with new 5-arg signature."""
        calls = []

        def new_callback(stage, current, total, success, error_msg):
            calls.append((stage, current, total, success, error_msg))

        _call_progress(new_callback, "running", 1, 10, success=True, error_msg=None)

        assert len(calls) == 1
        assert calls[0] == ("running", 1, 10, True, None)

    def test_passes_error_message(self):
        """Passes error message to callback."""
        calls = []

        def new_callback(stage, current, total, success, error_msg):
            calls.append((stage, current, total, success, error_msg))

        _call_progress(new_callback, "judging", 5, 20, success=False, error_msg="API timeout")

        assert len(calls) == 1
        assert calls[0] == ("judging", 5, 20, False, "API timeout")

    def test_backward_compatible_with_old_signature(self):
        """Falls back to old 3-arg signature for legacy callbacks."""
        calls = []

        def old_callback(stage, current, total):
            calls.append((stage, current, total))

        _call_progress(old_callback, "running", 3, 15, success=True, error_msg=None)

        assert len(calls) == 1
        assert calls[0] == ("running", 3, 15)

    def test_backward_compatible_ignores_extra_args(self):
        """Old callbacks don't receive success/error_msg."""
        calls = []

        def old_callback(stage, current, total):
            calls.append((stage, current, total))

        _call_progress(old_callback, "judging", 2, 10, success=False, error_msg="Some error")

        # Old callback was still called, just without the extra args
        assert len(calls) == 1
        assert calls[0] == ("judging", 2, 10)

    def test_none_callback_does_nothing(self):
        """None callback is handled gracefully."""
        # Should not raise
        _call_progress(None, "running", 1, 10, success=True, error_msg=None)

    def test_default_success_is_true(self):
        """Default success value is True."""
        calls = []

        def new_callback(stage, current, total, success, error_msg):
            calls.append((stage, current, total, success, error_msg))

        _call_progress(new_callback, "running", 1, 10)

        assert len(calls) == 1
        assert calls[0][3] is True  # success defaults to True

    def test_default_error_msg_is_none(self):
        """Default error_msg value is None."""
        calls = []

        def new_callback(stage, current, total, success, error_msg):
            calls.append((stage, current, total, success, error_msg))

        _call_progress(new_callback, "running", 1, 10)

        assert len(calls) == 1
        assert calls[0][4] is None  # error_msg defaults to None


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

        def print_callback(stage, current, total, success=True, error_msg=None):
            status = format_status(success)
            print(f"{stage}: {current}/{total} {status}")

        _call_progress(print_callback, "running", 1, 57, success=True)
        _call_progress(print_callback, "running", 2, 57, success=False, error_msg="fail")

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        assert len(lines) == 2
        assert f"{GREEN}[ok]{RESET}" in lines[0]
        assert f"{RED}[error]{RESET}" in lines[1]
