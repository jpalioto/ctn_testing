"""HTTP runner for CTN SDK server integration."""
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class DryRunInfo:
    """Dry-run response showing what would be sent to the model."""
    kernel: str                           # The kernel/system prompt
    system_prompt: str                    # Full system prompt sent
    user_prompt: str                      # User message sent
    parameters: dict[str, Any] = field(default_factory=dict)  # Model parameters


@dataclass
class SDKResponse:
    """Response from SDK /send endpoint."""
    output: str
    provider: str
    model: str
    tokens: dict[str, int]  # {"input": N, "output": M}
    kernel: str = ""        # Kernel used for this response


@dataclass
class CombinedResponse:
    """Combined dry-run and actual response."""
    dry_run: DryRunInfo
    response: SDKResponse
    kernel_match: bool      # Invariant: dry_run.kernel == response.kernel


class SDKError(Exception):
    """Error from SDK server."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SDKRunner:
    """HTTP client for CTN SDK server.

    A black-box client that sends prompts through the SDK /send endpoint.
    Knows nothing about constraints - just sends strings and parses responses.
    """

    DEFAULT_TIMEOUT = 30.0  # seconds
    DEFAULT_STRATEGY = "operational"

    def __init__(
        self,
        base_url: str = "http://localhost:14380",
        timeout: float | None = None,
        strategy: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.strategy = strategy or self.DEFAULT_STRATEGY

    def health_check(self) -> bool:
        """Check if SDK server is running.

        Returns:
            True if server responds to /health, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def stats(self) -> dict[str, Any]:
        """Get server stats.

        Returns:
            Server statistics dict.

        Raises:
            SDKError: If request fails.
        """
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise SDKError("Request timed out", status_code=None)
        except requests.exceptions.ConnectionError:
            raise SDKError("Connection refused - is the SDK server running?", status_code=None)
        except requests.exceptions.HTTPError as e:
            raise SDKError(str(e), status_code=e.response.status_code if e.response else None)

    def send(
        self,
        input: str,
        provider: str = "anthropic",
        model: str | None = None,
        strategy: str | None = None,
        dry_run: bool = False,
    ) -> SDKResponse | DryRunInfo:
        """Send prompt through SDK.

        Args:
            input: The prompt string (e.g. "@analytical Explain recursion")
            provider: LLM provider (anthropic, openai, google)
            model: Model name (uses provider default if None)
            strategy: Constraint strategy (operational, structural, hybrid).
                      Uses instance default if None.
            dry_run: If True, returns DryRunInfo without calling LLM

        Returns:
            SDKResponse with output and metadata, or DryRunInfo if dry_run=True.

        Raises:
            SDKError: If request fails or server returns error.
        """
        effective_strategy = strategy if strategy is not None else self.strategy

        payload: dict[str, Any] = {
            "input": input,
            "provider": provider,
            "strategy": effective_strategy,
            "dry_run": dry_run,
        }

        if model is not None:
            payload["model"] = model

        try:
            response = requests.post(
                f"{self.base_url}/send",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if dry_run:
                # Parse dry-run response format
                return DryRunInfo(
                    kernel=data.get("kernel", ""),
                    system_prompt=data.get("systemPrompt", data.get("system_prompt", "")),
                    user_prompt=data.get("userPrompt", data.get("user_prompt", "")),
                    parameters=data.get("parameters", {}),
                )

            return SDKResponse(
                output=data.get("output", ""),
                provider=data.get("provider", provider),
                model=data.get("model", model or ""),
                tokens={
                    "input": data.get("tokens", {}).get("input", 0),
                    "output": data.get("tokens", {}).get("output", 0),
                },
                kernel=data.get("kernel", ""),
            )

        except requests.exceptions.Timeout:
            raise SDKError("Request timed out", status_code=None)
        except requests.exceptions.ConnectionError:
            raise SDKError("Connection refused - is the SDK server running?", status_code=None)
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            status_code = e.response.status_code if e.response else None

            # Try to extract error message from response body
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                    elif "message" in error_data:
                        error_msg = error_data["message"]
                except (ValueError, KeyError):
                    pass

            raise SDKError(error_msg, status_code=status_code)

    def send_with_dry_run(
        self,
        input: str,
        provider: str = "anthropic",
        model: str | None = None,
        strategy: str | None = None,
    ) -> CombinedResponse:
        """Send prompt through SDK, capturing both dry-run and actual response.

        This method first calls dry-run to capture what will be sent, then
        makes the actual call. Both results are returned together with an
        invariant check that the kernels match.

        Args:
            input: The prompt string (e.g. "@analytical Explain recursion")
            provider: LLM provider (anthropic, openai, google)
            model: Model name (uses provider default if None)
            strategy: Constraint strategy (operational, structural, hybrid).
                      Uses instance default if None.

        Returns:
            CombinedResponse with dry_run info, actual response, and invariant check.

        Raises:
            SDKError: If either request fails.
        """
        # 1. Dry-run to capture what will be sent
        dry_run_result = self.send(
            input=input,
            provider=provider,
            model=model,
            strategy=strategy,
            dry_run=True,
        )
        assert isinstance(dry_run_result, DryRunInfo)

        # 2. Actual call
        actual_result = self.send(
            input=input,
            provider=provider,
            model=model,
            strategy=strategy,
            dry_run=False,
        )
        assert isinstance(actual_result, SDKResponse)

        # 3. Return combined with invariant check
        return CombinedResponse(
            dry_run=dry_run_result,
            response=actual_result,
            kernel_match=(dry_run_result.kernel == actual_result.kernel),
        )
