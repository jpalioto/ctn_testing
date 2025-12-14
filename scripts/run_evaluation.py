#!/usr/bin/env python
"""
Run evaluation.

Usage:
    python scripts/run_evaluation.py --config domains/extraction/configs/phase1.yaml --data dev
    python scripts/run_evaluation.py --config domains/extraction/configs/phase1.yaml --data test --final
"""
import argparse
from pathlib import Path
import sys

from ctn_testing.core import DocumentSchema, FieldSchema, EvaluationConfig
from ctn_testing.runners import run_evaluation


# Default schema - will be loaded from config in future
DEFAULT_SCHEMA = DocumentSchema(
    name="invoice",
    fields=[
        FieldSchema(name="invoice_number", description="Unique invoice identifier"),
        FieldSchema(name="total_amount", description="Total amount due including tax"),
        FieldSchema(name="due_date", description="Payment due date"),
        FieldSchema(name="vendor_name", description="Company issuing the invoice"),
    ]
)


def main():
    parser = argparse.ArgumentParser(description="Run CTN evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--data",
        choices=["dev", "test"],
        default="dev",
        help="Which dataset to use (default: dev)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Final TEST run (requires confirmation)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    domain_dir = config_path.parent.parent
    data_dir = domain_dir / "data" / args.data
    output_dir = domain_dir / "results"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print(f"Expected structure:")
        print(f"  {data_dir}/documents/*.json")
        print(f"  {data_dir}/ground_truth/*.yaml")
        sys.exit(1)
    
    # Final run confirmation
    if args.data == "test":
        if not args.final:
            print("TEST set requires --final flag.")
            print("This is a one-time evaluation. Are you sure?")
            sys.exit(1)
        
        response = input("Running FINAL TEST evaluation. Type 'yes' to confirm: ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    # Load config for cost estimate
    config = EvaluationConfig.from_yaml(config_path)
    
    # Estimate
    from ctn_testing.runners import load_document_set
    try:
        docs = load_document_set(data_dir)
        n_docs = len(docs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    matrix_size = len(config.run_matrix())
    total_calls = matrix_size * n_docs
    
    print(f"\nEvaluation: {config.name}")
    print(f"Data: {args.data} ({n_docs} documents)")
    print(f"Matrix: {len(config.models)} models × {len(config.enabled_kernels())} kernels = {matrix_size}")
    print(f"Total API calls: {total_calls}")
    print()
    
    # Run
    results = run_evaluation(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        schema=DEFAULT_SCHEMA,
        verbose=not args.quiet,
    )
    
    print(f"\nResults saved to: {output_dir / f'run_{results.run_id}'}")


if __name__ == "__main__":
    main()
