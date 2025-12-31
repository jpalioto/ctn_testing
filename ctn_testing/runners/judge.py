"""Judge module for evaluating extractions."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from ..core import GroundTruth, ModelConfig
from ..utils.logger import get_logger
from ..utils.network import make_client
from .kernel import RenderedPrompt
from .results import FieldResult


@dataclass
class JudgeResult:
    fields: list[FieldResult]
    raw_response: str
    outcome: str  # "OK" or "ERROR: <msg>"


class JudgeVerdict(BaseModel):
    field: str
    extracted: Any = None
    exact_match: bool = False
    semantic_match: bool = False
    usable: bool = False
    complete: bool = False


class JudgeResponse(BaseModel):
    verdicts: list[JudgeVerdict]

    @field_validator("verdicts", mode="before")
    @classmethod
    def normalize_input(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            for key in ["evaluations", "results", "fields", "verdicts"]:
                if key in v and isinstance(v[key], list):
                    return v[key]
            first_val = next(iter(v.values()), None) if v else None
            if isinstance(first_val, dict) and "value_correct" in first_val:
                return [{"field": k, **val} for k, val in v.items()]
            return [v]
        return []


def load_judge_prompt(path: Path) -> str:
    """Load judge prompt template from file."""
    if not path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def judge_extraction(
    judge_model: ModelConfig,
    judge_prompt_path: Path,
    ground_truth: dict[str, GroundTruth],
    raw_response: str,
) -> JudgeResult:
    """Have judge model evaluate extraction quality."""
    log = get_logger()

    # Load system and user prompts
    base_path = judge_prompt_path.with_suffix("")
    system_path = Path(str(base_path) + "_system.txt")
    user_path = Path(str(base_path) + "_user.txt")

    system_prompt = load_judge_prompt(system_path)
    user_template = load_judge_prompt(user_path)

    # Format ground truth
    gt_formatted = {}
    for name, gt in ground_truth.items():
        gt_formatted[name] = {
            "expected_value": gt.value,
            "acceptable_values": gt.acceptable_values if gt.acceptable_values else None,
            "candidate_values": gt.candidate_values if gt.candidate_values else None,
            "exists": gt.exists_in_document,
            "ambiguous": gt.is_ambiguous,
        }

    user_prompt = user_template.format(
        ground_truth=json.dumps(gt_formatted, indent=2),
        candidate=raw_response,
    )

    log.debug("Judge prompt constructed", data={"prompt_length": len(user_prompt)})

    prompt = RenderedPrompt(
        system=system_prompt,
        user=user_prompt,
        kernel_name="judge",
        document=None,
    )

    # Call judge
    complete = make_client(judge_model)
    result = complete(prompt)
    raw_judge_response = result.text

    log.debug(
        "Judge raw response",
        data=raw_judge_response[:500] if len(raw_judge_response) > 500 else raw_judge_response,
    )

    # Strip markdown fences
    text = raw_judge_response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start_idx = 1
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])
        log.debug("Stripped code fences", data={"start": start_idx, "end": end_idx})

    # Parse response
    try:
        raw = json.loads(text)
        response = JudgeResponse(verdicts=raw)
        verdicts = response.verdicts
    except (json.JSONDecodeError, Exception) as e:
        log.error(
            "Judge response parse failed",
            exc=e,
            context={
                "raw_text_preview": text[:500],
                "raw_text_length": len(text),
            },
        )
        return JudgeResult(fields=[], raw_response=raw_judge_response, outcome=f"ERROR: {e}")

    log.debug("Parsed verdicts", data={"count": len(verdicts)})

    # Convert to FieldResults
    field_results = []
    for v in verdicts:
        field_name = v.field
        if not field_name:
            log.warn("Verdict missing field name", data=v)
            continue

        gt = ground_truth.get(field_name)

        scores = {
            "exact": 1.0 if v.exact_match else 0.0,
            "semantic": 1.0 if v.semantic_match else 0.0,
            "usable": 1.0 if v.usable else 0.0,
            "complete": 1.0 if v.complete else 0.0,
        }

        # Composite: semantic-heavy weighting
        composite = (
            0.10 * scores["exact"]
            + 0.40 * scores["semantic"]
            + 0.30 * scores["usable"]
            + 0.20 * scores["complete"]
        )

        field_results.append(
            FieldResult(
                field_name=field_name,
                extracted_value=v.extracted,
                expected_value=gt.value if gt else None,
                composite_score=composite,
                scores=scores,
            )
        )

    log.debug("Built field results", data={"count": len(field_results)})

    return JudgeResult(fields=field_results, raw_response=raw_judge_response, outcome="OK")
