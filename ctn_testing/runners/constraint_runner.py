"""Runner for constraint adherence evaluation."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .http_runner import SDKRunner, SDKError


@dataclass
class ConstraintConfig:
    """Configuration for a constraint to test."""
    name: str           # e.g., "analytical"
    input_prefix: str   # e.g., "@analytical "
    description: str = ""


@dataclass
class PromptConfig:
    """Configuration for a test prompt."""
    id: str
    text: str
    category: str
    complexity: str = "medium"
    notes: str = ""


@dataclass
class RunResult:
    """Result of running a single prompt with a constraint."""
    prompt_id: str
    constraint_name: str
    input_sent: str           # Full input with prefix
    output: str
    provider: str
    model: str
    tokens: dict[str, int]
    timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_id": self.prompt_id,
            "constraint_name": self.constraint_name,
            "input_sent": self.input_sent,
            "output": self.output,
            "provider": self.provider,
            "model": self.model,
            "tokens": self.tokens,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class ConstraintRunner:
    """Runner for executing prompt × constraint combinations.

    Executes prompts through the SDK with different constraint prefixes
    and collects responses for subsequent judging.
    """

    def __init__(
        self,
        sdk_runner: SDKRunner,
        prompts: list[PromptConfig],
        constraints: list[ConstraintConfig],
        provider: str = "anthropic",
        model: str | None = None,
    ):
        """Initialize the constraint runner.

        Args:
            sdk_runner: SDK runner for making LLM calls
            prompts: List of prompts to test
            constraints: List of constraints to apply
            provider: LLM provider to use
            model: Model name (uses provider default if None)
        """
        self.sdk_runner = sdk_runner
        self.prompts = prompts
        self.constraints = constraints
        self.provider = provider
        self.model = model

    def run_single(
        self,
        prompt: PromptConfig,
        constraint: ConstraintConfig,
    ) -> RunResult:
        """Run one prompt with one constraint.

        Args:
            prompt: The prompt to run
            constraint: The constraint to apply

        Returns:
            RunResult with response or error
        """
        # Build full input with constraint prefix
        input_sent = self._build_input(prompt.text, constraint.input_prefix)
        timestamp = datetime.now().isoformat()

        try:
            response = self.sdk_runner.send(
                input=input_sent,
                provider=self.provider,
                model=self.model,
            )

            return RunResult(
                prompt_id=prompt.id,
                constraint_name=constraint.name,
                input_sent=input_sent,
                output=response.output,
                provider=response.provider,
                model=response.model,
                tokens=response.tokens,
                timestamp=timestamp,
                error=None,
            )

        except SDKError as e:
            return RunResult(
                prompt_id=prompt.id,
                constraint_name=constraint.name,
                input_sent=input_sent,
                output="",
                provider=self.provider,
                model=self.model or "",
                tokens={"input": 0, "output": 0},
                timestamp=timestamp,
                error=str(e),
            )

    def run_all(self) -> list[RunResult]:
        """Run all prompt × constraint combinations.

        Returns:
            List of RunResults for all combinations
        """
        results = []
        for prompt in self.prompts:
            for constraint in self.constraints:
                result = self.run_single(prompt, constraint)
                results.append(result)
        return results

    def run_prompt(self, prompt: PromptConfig) -> dict[str, RunResult]:
        """Run one prompt with all constraints.

        Useful for paired comparison where you want baseline vs constrained
        for the same prompt.

        Args:
            prompt: The prompt to run

        Returns:
            Dict mapping constraint name to RunResult
        """
        results = {}
        for constraint in self.constraints:
            result = self.run_single(prompt, constraint)
            results[constraint.name] = result
        return results

    def _build_input(self, prompt_text: str, prefix: str) -> str:
        """Build the full input string with constraint prefix.

        Args:
            prompt_text: The base prompt text
            prefix: The constraint prefix (may be empty for baseline)

        Returns:
            Combined input string
        """
        if not prefix:
            return prompt_text
        # Prefix already includes trailing space if needed
        return f"{prefix}{prompt_text}"


def load_prompts(path: Path) -> list[PromptConfig]:
    """Load prompts from YAML file.

    Args:
        path: Path to prompts.yaml

    Returns:
        List of PromptConfig objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid prompts file format: expected dict, got {type(data)}")

    raw_prompts = data.get("prompts", [])
    if not isinstance(raw_prompts, list):
        raise ValueError(f"Invalid prompts format: expected list, got {type(raw_prompts)}")

    prompts = []
    for raw in raw_prompts:
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid prompt format: expected dict, got {type(raw)}")

        prompt_id = raw.get("id")
        if not prompt_id:
            raise ValueError("Prompt missing required 'id' field")

        text = raw.get("text")
        if not text:
            raise ValueError(f"Prompt '{prompt_id}' missing required 'text' field")

        prompts.append(PromptConfig(
            id=prompt_id,
            text=text,
            category=raw.get("category", ""),
            complexity=raw.get("complexity", "medium"),
            notes=raw.get("notes", ""),
        ))

    return prompts


def load_constraints(config: dict) -> list[ConstraintConfig]:
    """Load constraints from config dict.

    Args:
        config: Config dict with 'constraints' key

    Returns:
        List of ConstraintConfig objects

    Raises:
        ValueError: If config format is invalid
    """
    raw_constraints = config.get("constraints", [])
    if not isinstance(raw_constraints, list):
        raise ValueError(f"Invalid constraints format: expected list, got {type(raw_constraints)}")

    constraints = []
    for raw in raw_constraints:
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid constraint format: expected dict, got {type(raw)}")

        name = raw.get("name")
        if not name:
            raise ValueError("Constraint missing required 'name' field")

        constraints.append(ConstraintConfig(
            name=name,
            input_prefix=raw.get("input_prefix", ""),
            description=raw.get("description", ""),
        ))

    return constraints
