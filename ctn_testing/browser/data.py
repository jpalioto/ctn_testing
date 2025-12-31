"""Data loading utilities for CTN Results Browser."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ConstraintInfo:
    """Information about a constraint from config."""

    name: str
    input_prefix: str
    description: str = ""


@dataclass
class RunSummary:
    """Summary of a test run."""

    run_id: str
    path: Path
    timestamp: datetime
    strategy: str | None
    config_name: str
    prompts_count: int
    constraints: list[str]
    total_sdk_calls: int
    total_judge_calls: int
    errors: list[str]
    run_type: str = "evaluation"
    source_run_id: str | None = None
    constraint_configs: list[ConstraintInfo] = field(default_factory=list)


@dataclass
class ResponseData:
    """Data from a single response."""

    prompt_id: str
    prompt_text: str
    constraint_name: str
    input_sent: str
    output: str
    tokens_in: int
    tokens_out: int
    error: str | None


@dataclass
class JudgingData:
    """Data from a single judging comparison."""

    prompt_id: str
    prompt_text: str
    test_constraint: str
    baseline_constraint: str
    baseline_response: str
    test_response: str
    baseline_scores: dict[str, dict]  # {trait: {score, reasons}}
    test_scores: dict[str, dict]
    baseline_was_a: bool
    raw_response: str | None = None


def parse_run_timestamp(run_id: str) -> datetime | None:
    """Parse timestamp from run_id format: YYYY-MM-DDTHH-MM-SS-ffffff"""
    try:
        # Handle format with microseconds
        if len(run_id) == 26:
            return datetime.strptime(run_id, "%Y-%m-%dT%H-%M-%S-%f")
        # Handle format without microseconds
        elif len(run_id) == 19:
            return datetime.strptime(run_id, "%Y-%m-%dT%H-%M-%S")
        else:
            return None
    except ValueError:
        return None


def detect_strategy(manifest: dict, responses: list[ResponseData]) -> str | None:
    """Detect strategy from manifest or response content."""
    # Check manifest for strategy in judge_model_override
    if manifest.get("judge_model_override"):
        return manifest["judge_model_override"].get("strategy")

    # Try to infer from response content
    if responses:
        sample_input = responses[0].input_sent
        if "CTN_KERNEL_SCHEMA" in sample_input:
            return "ctn"
        elif "behavioral_constraints" in sample_input.lower():
            return "operational"
        elif "<constraints>" in sample_input:
            return "structural"

    return None


def get_manifest(run_path: Path) -> dict:
    """Load manifest.json from run."""
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path) as f:
        return json.load(f)


def get_run_config(run_path: Path) -> dict:
    """Load the evaluation config from run directory.

    The config is copied to config/ subdirectory during the run.
    """
    config_dir = run_path / "config"
    if not config_dir.exists():
        return {}

    # Find the yaml config file (not prompts.yaml or traits.yaml)
    for yaml_file in config_dir.glob("*.yaml"):
        if yaml_file.stem not in ("prompts", "traits"):
            try:
                with open(yaml_file) as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError:
                continue

    return {}


def get_strategy_from_config(config: dict) -> str | None:
    """Extract strategy from config dict."""
    runner = config.get("runner", {})
    return runner.get("strategy")


def get_constraint_configs(config: dict) -> list[ConstraintInfo]:
    """Extract constraint configurations from config dict."""
    constraints = config.get("constraints", [])
    result = []
    for c in constraints:
        if isinstance(c, dict):
            result.append(
                ConstraintInfo(
                    name=c.get("name", ""),
                    input_prefix=c.get("input_prefix", ""),
                    description=c.get("description", ""),
                )
            )
    return result


def list_runs(results_dir: Path) -> list[RunSummary]:
    """List all runs in a results directory, sorted by date descending."""
    runs = []

    if not results_dir.exists():
        return runs

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = get_manifest(run_dir)
        timestamp = parse_run_timestamp(run_dir.name)

        # Load config to get strategy and constraint configs
        config = get_run_config(run_dir)
        strategy = get_strategy_from_config(config)
        constraint_configs = get_constraint_configs(config)

        # Fallback: try to detect strategy from response content
        if not strategy:
            responses = load_responses(run_dir, limit=1)
            strategy = detect_strategy(manifest, responses)

        run_summary = RunSummary(
            run_id=run_dir.name,
            path=run_dir,
            timestamp=timestamp or datetime.min,
            strategy=strategy,
            config_name=manifest.get("config_file", "unknown"),
            prompts_count=manifest.get("prompts_count", 0),
            constraints=manifest.get("constraints", []),
            total_sdk_calls=manifest.get("total_sdk_calls", 0),
            total_judge_calls=manifest.get("total_judge_calls", 0),
            errors=manifest.get("errors", []),
            run_type=manifest.get("run_type", "evaluation"),
            source_run_id=manifest.get("source_run_id"),
            constraint_configs=constraint_configs,
        )
        runs.append(run_summary)

    # Sort by timestamp descending
    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return runs


def load_responses(run_path: Path, limit: int | None = None) -> list[ResponseData]:
    """Load all responses from a run."""
    responses = []
    responses_dir = run_path / "responses"

    if not responses_dir.exists():
        # Check if this is a rejudge run - responses are in source
        manifest = get_manifest(run_path)
        source_path = manifest.get("source_responses_path")
        if source_path:
            # Resolve relative path
            responses_dir = run_path / source_path
            if not responses_dir.exists():
                return responses
        else:
            return responses

    response_files = sorted(responses_dir.glob("*.json"))
    if limit:
        response_files = response_files[:limit]

    for response_file in response_files:
        try:
            with open(response_file) as f:
                data = json.load(f)

            responses.append(
                ResponseData(
                    prompt_id=data.get("prompt_id", ""),
                    prompt_text=data.get("prompt_text", ""),
                    constraint_name=data.get("constraint_name", ""),
                    input_sent=data.get("input_sent", ""),
                    output=data.get("output", ""),
                    tokens_in=data.get("tokens", {}).get("input", 0),
                    tokens_out=data.get("tokens", {}).get("output", 0),
                    error=data.get("error"),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return responses


def load_judgings(run_path: Path) -> list[JudgingData]:
    """Load all judging results from a run."""
    judgings = []
    judging_dir = run_path / "judging"

    if not judging_dir.exists():
        return judgings

    for judging_file in sorted(judging_dir.glob("*.json")):
        try:
            with open(judging_file) as f:
                data = json.load(f)

            # Parse scores
            baseline_was_a = data.get("baseline_was_a", True)

            if baseline_was_a:
                baseline_scores = data.get("response_a_scores", {})
                test_scores = data.get("response_b_scores", {})
            else:
                baseline_scores = data.get("response_b_scores", {})
                test_scores = data.get("response_a_scores", {})

            judgings.append(
                JudgingData(
                    prompt_id=data.get("prompt_id", ""),
                    prompt_text=data.get("prompt_text", ""),
                    test_constraint=data.get("test_constraint", ""),
                    baseline_constraint=data.get("baseline_constraint", "baseline"),
                    baseline_response=data.get("baseline_response", ""),
                    test_response=data.get("test_response", ""),
                    baseline_scores=baseline_scores,
                    test_scores=test_scores,
                    baseline_was_a=baseline_was_a,
                    raw_response=data.get("raw_judge_response"),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return judgings


def extract_kernel(input_sent: str) -> tuple[str, str]:
    """Extract kernel/system prompt from input_sent.

    Returns:
        Tuple of (kernel_type, kernel_content)
    """
    # Check for CTN kernel
    ctn_match = re.search(r'CTN_KERNEL_SCHEMA\s*=\s*"""(.*?)"""', input_sent, re.DOTALL)
    if ctn_match:
        return ("ctn", ctn_match.group(1).strip())

    # Check for XML-style structural constraints
    structural_match = re.search(
        r"<constraints>(.*?)</constraints>", input_sent, re.DOTALL | re.IGNORECASE
    )
    if structural_match:
        return ("structural", structural_match.group(1).strip())

    # Check for operational/behavioral constraints
    behavioral_match = re.search(
        r"(?:behavioral_constraints|system\s*prompt)[:\s]*(.*?)(?:\n\n|\Z)",
        input_sent,
        re.DOTALL | re.IGNORECASE,
    )
    if behavioral_match:
        return ("operational", behavioral_match.group(1).strip())

    # Check for any system message pattern
    system_match = re.search(r"\[System\](.*?)\[/System\]", input_sent, re.DOTALL | re.IGNORECASE)
    if system_match:
        return ("system", system_match.group(1).strip())

    # Check for @constraint prefix
    constraint_match = re.match(r"^@(\w+)\s+(.*)", input_sent, re.DOTALL)
    if constraint_match:
        return (constraint_match.group(1), constraint_match.group(2).strip())

    # Return raw input if no pattern matched
    return ("raw", input_sent)


def get_unique_prompts(responses: list[ResponseData]) -> list[tuple[str, str]]:
    """Get unique (prompt_id, prompt_text) pairs from responses."""
    seen = set()
    prompts = []
    for r in responses:
        if r.prompt_id not in seen:
            seen.add(r.prompt_id)
            prompts.append((r.prompt_id, r.prompt_text))
    return prompts


def get_unique_constraints(responses: list[ResponseData]) -> list[str]:
    """Get unique constraint names from responses."""
    return sorted(set(r.constraint_name for r in responses))
