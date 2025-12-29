"""HTTP runner for CTN SDK server integration."""
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SDKResponse:
    """Response from SDK /send endpoint."""
    output: str
    provider: str
    model: str
    tokens: dict[str, int]  # {"input": N, "output": M}


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

    def __init__(
        self,
        base_url: str = "http://localhost:14380",
        timeout: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT

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
        strategy: str = "operational",
        dry_run: bool = False,
    ) -> SDKResponse:
        """Send prompt through SDK.

        Args:
            input: The prompt string (e.g. "@analytical Explain recursion")
            provider: LLM provider (anthropic, openai, google)
            model: Model name (uses provider default if None)
            strategy: Constraint strategy (operational, structural, hybrid)
            dry_run: If True, returns processed prompt without calling LLM

        Returns:
            SDKResponse with output and metadata.

        Raises:
            SDKError: If request fails or server returns error.
        """
        payload: dict[str, Any] = {
            "input": input,
            "provider": provider,
            "strategy": strategy,
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

            return SDKResponse(
                output=data.get("output", ""),
                provider=data.get("provider", provider),
                model=data.get("model", model or ""),
                tokens={
                    "input": data.get("tokens", {}).get("input", 0),
                    "output": data.get("tokens", {}).get("output", 0),
                },
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
