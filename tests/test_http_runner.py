"""Tests for HTTP runner (SDK integration)."""
from unittest.mock import Mock, patch

import pytest
import requests

from ctn_testing.runners.http_runner import (
    SDKRunner,
    SDKResponse,
    SDKError,
    DryRunInfo,
    CombinedResponse,
)


class TestSDKResponse:
    """Tests for SDKResponse dataclass."""

    def test_response_fields(self):
        """SDKResponse stores all expected fields."""
        response = SDKResponse(
            output="Hello, world!",
            provider="anthropic",
            model="claude-sonnet-4-5",
            tokens={"input": 10, "output": 5},
        )

        assert response.output == "Hello, world!"
        assert response.provider == "anthropic"
        assert response.model == "claude-sonnet-4-5"
        assert response.tokens == {"input": 10, "output": 5}


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_true_when_server_responds(self):
        """Health check returns True when server responds 200."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = runner.health_check()

            assert result is True
            mock_get.assert_called_once_with(
                "http://localhost:14380/health",
                timeout=30.0,
            )

    def test_health_check_returns_false_when_server_down(self):
        """Health check returns False when connection refused."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()

            result = runner.health_check()

            assert result is False

    def test_health_check_returns_false_on_timeout(self):
        """Health check returns False on timeout."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout()

            result = runner.health_check()

            assert result is False

    def test_health_check_returns_false_on_non_200(self):
        """Health check returns False when server responds non-200."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            result = runner.health_check()

            assert result is False


class TestSend:
    """Tests for send endpoint."""

    def test_send_returns_sdk_response_with_expected_fields(self):
        """Send returns SDKResponse with all expected fields."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "Recursion is a technique...",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 15, "output": 100},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = runner.send(
                input="@analytical Explain recursion",
                provider="anthropic",
                model="claude-sonnet-4-5",
            )

            assert isinstance(result, SDKResponse)
            assert result.output == "Recursion is a technique..."
            assert result.provider == "anthropic"
            assert result.model == "claude-sonnet-4-5"
            assert result.tokens == {"input": 15, "output": 100}

    def test_send_handles_dry_run_mode(self):
        """Send correctly passes dry_run flag and returns DryRunInfo."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "kernel": "CTN_KERNEL_SCHEMA = ...",
                "systemPrompt": "You are a helpful assistant...",
                "userPrompt": "Explain recursion",
                "parameters": {"temperature": 0.7},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = runner.send(
                input="@analytical Explain recursion",
                dry_run=True,
            )

            # Verify dry_run was passed in the payload
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["dry_run"] is True

            # Result should be DryRunInfo, not SDKResponse
            assert isinstance(result, DryRunInfo)
            assert result.kernel == "CTN_KERNEL_SCHEMA = ..."
            assert result.system_prompt == "You are a helpful assistant..."
            assert result.user_prompt == "Explain recursion"
            assert result.parameters == {"temperature": 0.7}

    def test_send_raises_on_connection_error(self):
        """Send raises SDKError when connection refused."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.ConnectionError()

            with pytest.raises(SDKError) as exc_info:
                runner.send(input="test")

            assert "Connection refused" in str(exc_info.value)
            assert exc_info.value.status_code is None

    def test_send_raises_on_timeout(self):
        """Send raises SDKError on timeout."""
        runner = SDKRunner(base_url="http://localhost:14380", timeout=5.0)

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.Timeout()

            with pytest.raises(SDKError) as exc_info:
                runner.send(input="test")

            assert "timed out" in str(exc_info.value)
            assert exc_info.value.status_code is None

    def test_send_raises_on_http_error(self):
        """Send raises SDKError on HTTP error response."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            import requests
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "Invalid provider"}

            http_error = requests.exceptions.HTTPError(response=mock_response)
            mock_response.raise_for_status.side_effect = http_error
            mock_post.return_value = mock_response

            with pytest.raises(SDKError) as exc_info:
                runner.send(input="test", provider="invalid")

            assert "Invalid provider" in str(exc_info.value)
            assert exc_info.value.status_code == 400

    def test_send_uses_default_provider(self):
        """Send uses anthropic as default provider."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="test")

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["provider"] == "anthropic"

    def test_send_omits_model_when_none(self):
        """Send omits model from payload when not specified."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="test", model=None)

            call_kwargs = mock_post.call_args[1]
            assert "model" not in call_kwargs["json"]

    def test_send_includes_model_when_specified(self):
        """Send includes model in payload when specified."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "openai",
                "model": "gpt-4",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="test", provider="openai", model="gpt-4")

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "gpt-4"


class TestTimeoutHandling:
    """Tests for timeout configuration."""

    def test_default_timeout(self):
        """Runner uses default 30s timeout."""
        runner = SDKRunner()
        assert runner.timeout == 30.0

    def test_custom_timeout(self):
        """Runner accepts custom timeout."""
        runner = SDKRunner(timeout=60.0)
        assert runner.timeout == 60.0

    def test_timeout_passed_to_requests(self):
        """Timeout is passed to requests calls."""
        runner = SDKRunner(timeout=15.0)

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="test")

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["timeout"] == 15.0


class TestURLHandling:
    """Tests for URL handling."""

    def test_base_url_trailing_slash_stripped(self):
        """Trailing slash in base_url is stripped."""
        runner = SDKRunner(base_url="http://localhost:14380/")
        assert runner.base_url == "http://localhost:14380"

    def test_default_base_url(self):
        """Default base URL is localhost:14380."""
        runner = SDKRunner()
        assert runner.base_url == "http://localhost:14380"


class TestStats:
    """Tests for stats endpoint."""

    def test_stats_returns_dict(self):
        """Stats returns server statistics dict."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "requests_total": 100,
                "requests_success": 95,
                "uptime_seconds": 3600,
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = runner.stats()

            assert result["requests_total"] == 100
            assert result["requests_success"] == 95

    def test_stats_raises_on_error(self):
        """Stats raises SDKError on failure."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()

            with pytest.raises(SDKError):
                runner.stats()


class TestSDKError:
    """Tests for SDKError exception."""

    def test_error_with_status_code(self):
        """SDKError stores status code."""
        error = SDKError("Bad request", status_code=400)
        assert str(error) == "Bad request"
        assert error.status_code == 400

    def test_error_without_status_code(self):
        """SDKError works without status code."""
        error = SDKError("Connection failed")
        assert str(error) == "Connection failed"
        assert error.status_code is None


class TestResponseParsing:
    """Tests for response parsing edge cases."""

    def test_handles_missing_tokens(self):
        """Response parsing handles missing tokens field."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = runner.send(input="test")

            assert result.tokens == {"input": 0, "output": 0}

    def test_handles_empty_output(self):
        """Response parsing handles empty output."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 5, "output": 0},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = runner.send(input="test")

            assert result.output == ""


class TestStrategyParameter:
    """Tests for strategy parameter handling."""

    def test_default_strategy(self):
        """Runner uses default 'operational' strategy."""
        runner = SDKRunner()
        assert runner.strategy == "operational"

    def test_custom_strategy_in_constructor(self):
        """Runner accepts custom strategy in constructor."""
        runner = SDKRunner(strategy="ctn")
        assert runner.strategy == "ctn"

    def test_strategy_passed_to_request_body(self):
        """Strategy is passed in request body."""
        runner = SDKRunner(strategy="ctn")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="@analytical hello", provider="anthropic")

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["strategy"] == "ctn"

    def test_strategy_can_be_overridden_per_request(self):
        """Strategy can be overridden in send() call."""
        runner = SDKRunner(strategy="operational")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="test", strategy="hybrid")

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["strategy"] == "hybrid"

    def test_none_strategy_uses_default(self):
        """None strategy in send() uses instance default."""
        runner = SDKRunner(strategy="structural")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "test",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 1, "output": 1},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="test", strategy=None)

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["strategy"] == "structural"

    def test_complete_request_body_with_strategy(self):
        """Complete request body includes strategy alongside other fields."""
        runner = SDKRunner(base_url="http://localhost:14380", strategy="ctn")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "response",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 10, "output": 20},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            runner.send(input="@analytical hello", provider="anthropic")

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]

            assert payload["input"] == "@analytical hello"
            assert payload["provider"] == "anthropic"
            assert payload["strategy"] == "ctn"
            assert payload["dry_run"] is False


class TestSendWithDryRun:
    """Tests for combined dry-run + actual response."""

    def test_dry_run_returns_kernel_structure(self):
        """Dry-run returns DryRunInfo with kernel, system_prompt, user_prompt."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "kernel": "CTN_KERNEL v1.0\nΣ(constraints)",
                "systemPrompt": "You are a helpful assistant with τ timing.",
                "userPrompt": "Explain recursion Ψ",
                "parameters": {"temperature": 0.7, "max_tokens": 1000},
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = runner.send(input="@analytical hello", dry_run=True)

            assert isinstance(result, DryRunInfo)
            assert result.kernel == "CTN_KERNEL v1.0\nΣ(constraints)"
            assert result.system_prompt == "You are a helpful assistant with τ timing."
            assert result.user_prompt == "Explain recursion Ψ"
            assert result.parameters == {"temperature": 0.7, "max_tokens": 1000}

    def test_actual_returns_output_and_kernel(self):
        """Actual call returns SDKResponse with output and kernel."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "Recursion is when a function calls itself Ω",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 25, "output": 150},
                "kernel": "CTN_KERNEL v1.0\nΣ(constraints)",
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = runner.send(input="@analytical hello", dry_run=False)

            assert isinstance(result, SDKResponse)
            assert result.output == "Recursion is when a function calls itself Ω"
            assert result.kernel == "CTN_KERNEL v1.0\nΣ(constraints)"
            assert result.provider == "anthropic"
            assert result.model == "claude-sonnet-4-5"
            assert result.tokens == {"input": 25, "output": 150}

    def test_send_with_dry_run_returns_combined_response(self):
        """send_with_dry_run returns CombinedResponse with both results."""
        runner = SDKRunner()

        call_count = [0]

        def mock_post_handler(*args, **kwargs):
            call_count[0] += 1
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()

            # First call is dry-run
            if kwargs["json"]["dry_run"]:
                mock_response.json.return_value = {
                    "kernel": "KERNEL_ABC",
                    "systemPrompt": "System prompt",
                    "userPrompt": "User prompt",
                    "parameters": {"temperature": 0.5},
                }
            else:
                # Second call is actual
                mock_response.json.return_value = {
                    "output": "Response output",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "tokens": {"input": 10, "output": 50},
                    "kernel": "KERNEL_ABC",
                }

            return mock_response

        with patch("ctn_testing.runners.http_runner.requests.post", side_effect=mock_post_handler):
            result = runner.send_with_dry_run(
                input="@analytical hello",
                provider="anthropic",
            )

            assert isinstance(result, CombinedResponse)
            assert call_count[0] == 2

            # Check dry_run info
            assert result.dry_run.kernel == "KERNEL_ABC"
            assert result.dry_run.system_prompt == "System prompt"
            assert result.dry_run.user_prompt == "User prompt"

            # Check actual response
            assert result.response.output == "Response output"
            assert result.response.kernel == "KERNEL_ABC"

    def test_kernel_match_true_when_kernels_identical(self):
        """kernel_match is True when dry-run and actual kernels match."""
        runner = SDKRunner()

        def mock_post_handler(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()

            if kwargs["json"]["dry_run"]:
                mock_response.json.return_value = {
                    "kernel": "IDENTICAL_KERNEL_Σ",
                    "systemPrompt": "Prompt",
                    "userPrompt": "User",
                    "parameters": {},
                }
            else:
                mock_response.json.return_value = {
                    "output": "Output",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "tokens": {"input": 1, "output": 1},
                    "kernel": "IDENTICAL_KERNEL_Σ",
                }

            return mock_response

        with patch("ctn_testing.runners.http_runner.requests.post", side_effect=mock_post_handler):
            result = runner.send_with_dry_run(input="test")

            assert result.kernel_match is True
            assert result.dry_run.kernel == result.response.kernel

    def test_kernel_match_false_when_kernels_differ(self):
        """kernel_match is False when dry-run and actual kernels differ."""
        runner = SDKRunner()

        def mock_post_handler(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()

            if kwargs["json"]["dry_run"]:
                mock_response.json.return_value = {
                    "kernel": "KERNEL_V1",
                    "systemPrompt": "Prompt",
                    "userPrompt": "User",
                    "parameters": {},
                }
            else:
                mock_response.json.return_value = {
                    "output": "Output",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "tokens": {"input": 1, "output": 1},
                    "kernel": "KERNEL_V2_CHANGED",  # Different kernel!
                }

            return mock_response

        with patch("ctn_testing.runners.http_runner.requests.post", side_effect=mock_post_handler):
            result = runner.send_with_dry_run(input="test")

            assert result.kernel_match is False
            assert result.dry_run.kernel != result.response.kernel

    def test_send_with_dry_run_propagates_sdk_error_from_dry_run(self):
        """send_with_dry_run raises SDKError if dry-run fails."""
        runner = SDKRunner()

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()

            with pytest.raises(SDKError) as exc_info:
                runner.send_with_dry_run(input="test")

            assert "Connection refused" in str(exc_info.value)

    def test_send_with_dry_run_propagates_sdk_error_from_actual(self):
        """send_with_dry_run raises SDKError if actual call fails."""
        runner = SDKRunner()

        call_count = [0]

        def mock_post_handler(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (dry-run) succeeds
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {
                    "kernel": "K",
                    "systemPrompt": "S",
                    "userPrompt": "U",
                    "parameters": {},
                }
                return mock_response
            else:
                # Second call (actual) fails
                raise requests.exceptions.Timeout()

        with patch("ctn_testing.runners.http_runner.requests.post", side_effect=mock_post_handler):
            with pytest.raises(SDKError) as exc_info:
                runner.send_with_dry_run(input="test")

            assert "timed out" in str(exc_info.value)

    def test_send_with_dry_run_with_greek_symbols(self):
        """send_with_dry_run handles Greek symbols correctly."""
        runner = SDKRunner()

        greek_kernel = "Σ(constraints) → Ψ(reasoning) → Ω(output) with τ=0.7"
        greek_output = "The sum Σ of the wave function Ψ approaches Ω as τ→∞"

        def mock_post_handler(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()

            if kwargs["json"]["dry_run"]:
                mock_response.json.return_value = {
                    "kernel": greek_kernel,
                    "systemPrompt": "System with Σ",
                    "userPrompt": "User with Ψ",
                    "parameters": {},
                }
            else:
                mock_response.json.return_value = {
                    "output": greek_output,
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "tokens": {"input": 1, "output": 1},
                    "kernel": greek_kernel,
                }

            return mock_response

        with patch("ctn_testing.runners.http_runner.requests.post", side_effect=mock_post_handler):
            result = runner.send_with_dry_run(input="test")

            assert result.dry_run.kernel == greek_kernel
            assert result.response.kernel == greek_kernel
            assert result.response.output == greek_output
            assert "Σ" in result.dry_run.kernel
            assert "Ψ" in result.response.output
            assert "Ω" in result.response.output
            assert "τ" in result.dry_run.kernel
