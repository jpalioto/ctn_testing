#!/usr/bin/env python
"""Re-run judging on existing extraction results."""
import argparse
import json
from pathlib import Path

from ctn_testing.core import EvaluationConfig, GroundTruth
from ctn_testing.runners.judge import judge_extraction


def main():
    parser = argparse.ArgumentParser(description="Re-judge existing extractions")
    parser.add_argument("--results", type=Path, required=True, help="Path to run results dir")
    parser.add_argument("--config", type=Path, required=True, help="Path to config (for judge model)")
    args = parser.parse_args()
    
    # Load config for judge model
    config = EvaluationConfig.from_yaml(args.config)
    judge_model = config.judge_models[0]
    
    domain_dir = args.config.parent.parent
    judge_prompt_path = domain_dir / "prompts" / "judge"
    
    # Find all result files
    raw_dir = args.results / "raw"
    
    judged = 0
    errors = 0
    skipped = 0
    
    for model_dir in raw_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for kernel_dir in model_dir.iterdir():
            if not kernel_dir.is_dir():
                continue
            for result_file in kernel_dir.glob("*.json"):
                with open(result_file) as f:
                    result = json.load(f)
                
                doc_id = result.get("doc_id", result_file.stem)
                
                # Skip if no raw_response
                if not result.get("raw_response"):
                    print(f"  [{doc_id}] Skipping - no raw response")
                    skipped += 1
                    continue
                
                # Skip if no ground_truth
                if not result.get("ground_truth"):
                    print(f"  [{doc_id}] Skipping - no ground truth (old format)")
                    skipped += 1
                    continue
                
                print(f"  [{doc_id}] Re-judging...")
                
                # Reconstruct GroundTruth objects
                gt = {
                    name: GroundTruth.from_dict(gt_dict)
                    for name, gt_dict in result["ground_truth"].items()
                }
                
                try:
                    judge_result = judge_extraction(
                        judge_model=judge_model,
                        judge_prompt_path=judge_prompt_path,
                        ground_truth=gt,
                        raw_response=result["raw_response"],
                    )
                    
                    # Update result
                    result["fields"] = [f.to_dict() for f in judge_result.fields]
                    result["judge_raw_response"] = judge_result.raw_response
                    result["judge_outcome"] = judge_result.outcome
                    result["judge_model"] = judge_model.name
                    result["judge_temperature"] = judge_model.temperature
                    result["judge_max_tokens"] = judge_model.max_tokens
                    
                    # Recompute scores
                    if judge_result.fields:
                        result["composite_score"] = sum(f.composite_score for f in judge_result.fields) / len(judge_result.fields)
                        result["value_score"] = sum(f.value_score for f in judge_result.fields) / len(judge_result.fields)
                    
                    # Save updated result
                    with open(result_file, "w") as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"    OK: composite={result['composite_score']:.2f}")
                    judged += 1
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    errors += 1
    
    print(f"\nDone: {judged} judged, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()