"""Tests for HTTP runner (SDK integration)."""
from unittest.mock import Mock, patch

import pytest

from ctn_testing.runners.http_runner import SDKRunner, SDKResponse, SDKError


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
        """Send correctly passes dry_run flag."""
        runner = SDKRunner(base_url="http://localhost:14380")

        with patch("ctn_testing.runners.http_runner.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "output": "[DRY RUN] Would send: @analytical Explain recursion",
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": {"input": 0, "output": 0},
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
            assert result.output.startswith("[DRY RUN]")

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
