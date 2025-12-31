"""Blind judging system for constraint adherence evaluation."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..runners.http_runner import SDKError, SDKRunner
from .traits import load_traits


@dataclass
class TraitScore:
    """Score for a single trait dimension."""

    dimension: str  # e.g., "reasoning_depth"
    score: int  # 0-100
    reasons: list[str]  # Bullet points explaining score


@dataclass
class JudgingResult:
    """Result of blind judging two responses."""

    response_a_scores: dict[str, TraitScore] = field(default_factory=dict)
    response_b_scores: dict[str, TraitScore] = field(default_factory=dict)
    raw_response: str = ""  # For debugging
    error: str | None = None  # Parse error if any


class JudgingError(Exception):
    """Error during judging process."""

    pass


class BlindJudge:
    """Blind judge that scores responses on trait dimensions.

    The judge scores responses WITHOUT knowing which constraint was tested.
    This prevents confirmation bias. The judge sees "Response A" and "Response B"
    (which can be randomized by the caller) and scores each on multiple trait
    dimensions.
    """

    def __init__(
        self,
        traits_path: Path,
        sdk_runner: SDKRunner,
        judge_provider: str = "anthropic",
        judge_model: str | None = None,
    ):
        """Initialize the blind judge.

        Args:
            traits_path: Path to traits.yaml defining scoring dimensions
            sdk_runner: SDK runner for making judge calls
            judge_provider: Provider for judge model
            judge_model: Model name (uses provider default if None)
        """
        self.traits = load_traits(traits_path)
        self.runner = sdk_runner
        self.judge_provider = judge_provider
        self.judge_model = judge_model

    def judge(
        self,
        response_a: str,
        response_b: str,
        prompt_text: str,
    ) -> JudgingResult:
        """Score both responses on all trait dimensions.

        Args:
            response_a: First response to evaluate
            response_b: Second response to evaluate
            prompt_text: Original prompt for context

        Returns:
            JudgingResult with scores for both responses

        Raises:
            JudgingError: If SDK call fails
        """
        judge_prompt = self._build_judge_prompt(response_a, response_b, prompt_text)

        try:
            sdk_response = self.runner.send(
                input=judge_prompt,
                provider=self.judge_provider,
                model=self.judge_model,
            )
        except SDKError as e:
            raise JudgingError(f"SDK error during judging: {e}") from e

        return self._parse_response(sdk_response.output)

    def _build_judge_prompt(
        self,
        response_a: str,
        response_b: str,
        prompt_text: str,
    ) -> str:
        """Build the blind judging prompt.

        The prompt includes:
        - The original question for context
        - Both responses labeled A and B
        - All trait dimensions with descriptions and anchors
        - Instructions for JSON output format
        """
        dimensions_text = self._format_dimensions()
        dimension_names = self.traits.dimension_names()

        # Build expected JSON structure for prompt
        json_structure = self._build_json_template(dimension_names)

        return f'''You are evaluating two responses to the same prompt.
Score each response on the trait dimensions provided.

ORIGINAL PROMPT:
"{prompt_text}"

RESPONSE A:
"""
{response_a}
"""

RESPONSE B:
"""
{response_b}
"""

Score each response on these dimensions (0-100):

{dimensions_text}

For EACH response, provide scores and brief reasoning for each dimension.
Be consistent - the same quality should receive the same score regardless of which response.

Return JSON only (no markdown, no extra text):
{json_structure}'''

    def _format_dimensions(self) -> str:
        """Format all dimensions for the judge prompt."""
        parts = []
        for dim in self.traits.dimensions:
            part = f"{dim.name} ({dim.description})\n"
            part += f"Scale: {dim.scale}\n"
            part += "Anchors:\n"
            part += dim.format_anchors()
            parts.append(part)
        return "\n\n".join(parts)

    def _build_json_template(self, dimension_names: list[str]) -> str:
        """Build JSON template showing expected structure."""
        dims = {name: {"score": "<0-100>", "reasons": ["...", "..."]} for name in dimension_names}
        template = {
            "response_a": dims,
            "response_b": dims.copy(),
        }
        return json.dumps(template, indent=2)

    def _parse_response(self, raw_response: str) -> JudgingResult:
        """Parse judge response into structured result.

        Handles:
        - Valid JSON
        - JSON wrapped in markdown code blocks
        - Malformed JSON (returns error)
        - Missing dimensions (fills with defaults)
        """
        result = JudgingResult(raw_response=raw_response)

        # Try to extract JSON from response
        json_str = self._extract_json(raw_response)
        if json_str is None:
            result.error = "No valid JSON found in response"
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            result.error = f"JSON parse error: {e}"
            return result

        # Parse response_a scores
        if "response_a" in data:
            result.response_a_scores = self._parse_scores(data["response_a"], "response_a")

        # Parse response_b scores
        if "response_b" in data:
            result.response_b_scores = self._parse_scores(data["response_b"], "response_b")

        return result

    def _extract_json(self, text: str) -> str | None:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in markdown code block
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find raw JSON object
        # Look for outermost { } pair
        brace_start = text.find("{")
        if brace_start == -1:
            return None

        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[brace_start:], start=brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start : i + 1]

        return None

    def _parse_scores(self, data: dict, label: str) -> dict[str, TraitScore]:
        """Parse scores for one response from parsed JSON."""
        scores = {}

        for dim_name in self.traits.dimension_names():
            if dim_name not in data:
                # Missing dimension - skip (will be logged via result)
                continue

            dim_data = data[dim_name]

            # Handle various formats
            if isinstance(dim_data, dict):
                score = dim_data.get("score", 0)
                reasons = dim_data.get("reasons", [])
                # Handle case where reasons is a single string
                if isinstance(reasons, str):
                    reasons = [reasons]
            elif isinstance(dim_data, (int, float)):
                # Just a number
                score = dim_data
                reasons = []
            else:
                # Unknown format
                score = 0
                reasons = []

            # Ensure score is int in valid range
            try:
                score = int(score)
                score = max(0, min(100, score))
            except (ValueError, TypeError):
                score = 0

            scores[dim_name] = TraitScore(
                dimension=dim_name,
                score=score,
                reasons=reasons,
            )

        return scores
