#!/usr/bin/env python3
"""Create Tier 3 test prompts from downloaded datasets."""

import csv
import json
import random
import re
from pathlib import Path

import pandas as pd
import yaml


def create_tier3a_hallucination(data_dir: Path, output_dir: Path, sample_size: int = 50) -> None:
    """Create tier3a_hallucination.yaml from TruthfulQA dataset.

    Focuses on Misconceptions category - questions that test common false beliefs.
    """
    csv_path = data_dir / "TruthfulQA" / "TruthfulQA.csv"

    prompts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to Misconceptions category and sample
    misconceptions = [r for r in rows if r.get("Category") == "Misconceptions"]

    # Also include some from other interesting categories
    health = [r for r in rows if "Health" in r.get("Category", "")]
    science = [r for r in rows if "Science" in r.get("Category", "")]

    # Combine and dedupe
    all_candidates = misconceptions + health[:20] + science[:20]
    seen_questions = set()
    unique_candidates = []
    for r in all_candidates:
        q = r.get("Question", "")
        if q and q not in seen_questions:
            seen_questions.add(q)
            unique_candidates.append(r)

    # Sample
    random.seed(42)
    if len(unique_candidates) > sample_size:
        selected = random.sample(unique_candidates, sample_size)
    else:
        selected = unique_candidates

    for i, row in enumerate(selected, 1):
        question = row.get("Question", "").strip()
        best_answer = row.get("Best Answer", "").strip()
        category = row.get("Category", "Unknown").strip()

        # Clean up question
        if question and best_answer:
            prompts.append(
                {
                    "id": f"3a-{i:02d}",
                    "text": question,
                    "category": category,
                    "correct_answer": best_answer,
                    "source": "TruthfulQA",
                }
            )

    output_data = {
        "name": "tier3a_hallucination",
        "description": "Tier 3A: Hallucination resistance - common misconceptions from TruthfulQA",
        "version": "1.0",
        "prompts": prompts,
    }

    output_path = output_dir / "tier3a_hallucination.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Created {output_path} with {len(prompts)} prompts")


def create_tier3b_reasoning(data_dir: Path, output_dir: Path, sample_size: int = 30) -> None:
    """Create tier3b_reasoning.yaml from GSM8K dataset.

    Multi-step math problems for chain-of-thought coherence testing.
    """
    parquet_path = data_dir / "OpenAI" / "main" / "test-00000-of-00001.parquet"

    df = pd.read_parquet(parquet_path)

    prompts = []

    # Sample problems
    random.seed(42)
    indices = random.sample(range(len(df)), min(sample_size, len(df)))

    for i, idx in enumerate(indices, 1):
        row = df.iloc[idx]
        question = row["question"]
        answer_text = row["answer"]

        # Clean up encoding issues
        question = question.replace("�", "'").replace("â€™", "'").strip()

        # Extract final answer (format: "#### <number>")
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        if match:
            final_answer = match.group(1)
        else:
            final_answer = "unknown"

        # Count steps (approximate by counting "<<...>>" patterns)
        steps = len(re.findall(r"<<[^>]+>>", answer_text))
        if steps == 0:
            steps = answer_text.count("\n") + 1

        prompts.append(
            {
                "id": f"3b-{i:02d}",
                "text": question,
                "correct_answer": final_answer,
                "steps": steps,
                "source": "GSM8K",
            }
        )

    output_data = {
        "name": "tier3b_reasoning",
        "description": "Tier 3B: Chain coherence - multi-step math problems from GSM8K",
        "version": "1.0",
        "prompts": prompts,
    }

    output_path = output_dir / "tier3b_reasoning.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Created {output_path} with {len(prompts)} prompts")


def create_tier3d_sycophancy(data_dir: Path, output_dir: Path, sample_size: int = 30) -> None:
    """Create tier3d_sycophancy.yaml from Anthropic sycophancy evals.

    Prompts where user expresses a view to test if model agrees/disagrees appropriately.
    """
    jsonl_path = (
        data_dir / "Anthropic" / "evals-main" / "sycophancy" / "sycophancy_on_philpapers2020.jsonl"
    )

    prompts = []

    with open(jsonl_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    # Sample
    random.seed(42)
    if len(lines) > sample_size:
        selected = random.sample(lines, sample_size)
    else:
        selected = lines

    for i, item in enumerate(selected, 1):
        question = item.get("question", "")
        matching_behavior = item.get("answer_matching_behavior", "").strip()

        # Extract the user's stated view from the question
        # The format typically has "My view on X is Y" or "I believe X"
        user_bias = "unknown"
        if "I believe" in question:
            match = re.search(r"I believe[^.]+", question)
            if match:
                user_bias = match.group(0).strip()
        elif "My view" in question:
            match = re.search(r'My view[^.]+is "[^"]+"', question)
            if match:
                user_bias = match.group(0).strip()
        elif "I do not believe" in question:
            match = re.search(r"I do not believe[^.]+", question)
            if match:
                user_bias = match.group(0).strip()

        # Extract the topic from "What is your view on the following topic: X"
        topic_match = re.search(r"What is your view on the following topic:\s*([^\n]+)", question)
        topic = topic_match.group(1).strip() if topic_match else "philosophy"

        # Clean up the question - remove the explicit answer choices for more natural evaluation
        # Keep the persona and setup, but remove "(A) no (B) yes Answer:"
        clean_question = re.sub(r"\s*\(A\)[^\n]*\(B\)[^\n]*\n*Answer:\s*$", "", question).strip()

        prompts.append(
            {
                "id": f"3d-{i:02d}",
                "text": clean_question,
                "topic": topic,
                "user_bias": user_bias,
                "sycophantic_answer": matching_behavior,
                "source": "Anthropic sycophancy",
            }
        )

    output_data = {
        "name": "tier3d_sycophancy",
        "description": "Tier 3D: Sycophancy resistance - philosophical questions with stated user views from Anthropic evals",
        "version": "1.0",
        "prompts": prompts,
    }

    output_path = output_dir / "tier3d_sycophancy.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Created {output_path} with {len(prompts)} prompts")


def create_tier3_config(output_dir: Path, configs_dir: Path) -> None:
    """Create tier3_complex.yaml config combining all Tier 3 prompts."""

    config = {
        "name": "tier3_complex",
        "version": "1.0",
        "description": "Tier 3 Complex Reasoning Tests: Hallucination, Chain Coherence, Sycophancy",
        "runner": {"type": "sdk_http", "base_url": "http://localhost:14380", "timeout": 60},
        "prompts": {
            "sources": [
                "prompts/tier3a_hallucination.yaml",
                "prompts/tier3b_reasoning.yaml",
                "prompts/tier3d_sycophancy.yaml",
            ]
        },
        "models": [{"name": "claude-sonnet-4", "provider": "anthropic"}],
        "constraints": [
            {"name": "baseline"},
            {"name": "analytical"},
            {"name": "precise"},
            {"name": "strict"},
        ],
        "strategy": "ctn-v2",
        "judging": {"enabled": True, "provider": "anthropic", "model": "claude-sonnet-4"},
        "output": {"dir": "results", "include_raw_responses": True},
    }

    output_path = configs_dir / "tier3_complex.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Created {output_path}")


def main():
    base_dir = Path(__file__).parent.parent / "domains" / "constraint_adherence"
    data_dir = base_dir / "data"
    prompts_dir = base_dir / "prompts"
    configs_dir = base_dir / "configs"

    # Ensure output directories exist
    prompts_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    print("Creating Tier 3 test prompts from datasets...")
    print()

    # Create each prompt file
    create_tier3a_hallucination(data_dir, prompts_dir, sample_size=50)
    create_tier3b_reasoning(data_dir, prompts_dir, sample_size=30)
    create_tier3d_sycophancy(data_dir, prompts_dir, sample_size=30)

    # Create combined config
    create_tier3_config(prompts_dir, configs_dir)

    print()
    print("Done! Created:")
    print(f"  - {prompts_dir}/tier3a_hallucination.yaml (50 misconception prompts)")
    print(f"  - {prompts_dir}/tier3b_reasoning.yaml (30 multi-step math problems)")
    print(f"  - {prompts_dir}/tier3d_sycophancy.yaml (30 sycophancy test prompts)")
    print(f"  - {configs_dir}/tier3_complex.yaml (combined config using ctn-v2)")


if __name__ == "__main__":
    main()
