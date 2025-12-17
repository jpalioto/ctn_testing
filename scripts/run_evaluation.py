#!/usr/bin/env python
"""
Run evaluation.

Usage:
    python scripts/run_evaluation.py --config domains/extraction/configs/quick.yaml --data dev_quick
    python scripts/run_evaluation.py --config domains/extraction/configs/docile/docile_quick.yaml
    python scripts/run_evaluation.py --config domains/extraction/configs/phase1.yaml --data test --final
"""
import argparse
from pathlib import Path
import sys

from ctn_testing.core import DocumentSchema, FieldSchema, EvaluationConfig


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
        choices=["dev", "dev_quick", "dev_full", "test"],
        default=None,
        help="Which dataset to use (optional if config has dataset section)",
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
    
    # Resolve config path
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = EvaluationConfig.from_yaml(config_path)
    
    # Resolve output directory
    domain_dir = config_path.parent.parent
    output_dir = domain_dir / "results"
    
    # Load documents based on source
    data_dir: Path | None = None
    dataset_name: str
    
    if config.dataset:
        # External dataset specified in config
        if config.dataset.type == "docile":
            from ctn_testing.core.loaders import load_docile
            docs = load_docile(
                config.dataset.path,
                split=config.dataset.split,
                n=config.dataset.n,
            )
        else:
            print(f"Unknown dataset type: {config.dataset.type}")
            sys.exit(1)
        dataset_name = f"{config.dataset.type}:{config.dataset.split}"
        if config.dataset.n:
            dataset_name += f":{config.dataset.n}"
    elif args.data:
        # Local directory
        data_dir = domain_dir / "data" / args.data
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            print(f"Expected structure:")
            print(f"  {data_dir}/documents/*.json")
            print(f"  {data_dir}/ground_truth/*.yaml")
            sys.exit(1)
        
        from ctn_testing.runners import load_document_set
        try:
            docs = load_document_set(data_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        dataset_name = args.data
    else:
        print("Error: Either --data or config dataset section required")
        sys.exit(1)
    
    n_docs = len(docs)
    
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
    
    # Summary
    matrix_size = len(config.run_matrix())
    total_calls = matrix_size * n_docs
    
    print(f"\nEvaluation: {config.name}")
    print(f"Data: {dataset_name} ({n_docs} documents)")
    print(f"Matrix: {len(config.models)} models × {len(config.enabled_kernels())} kernels = {matrix_size}")
    print(f"Total API calls: {total_calls}")
    print()
    
    # Run
    from ctn_testing.runners import run_evaluation
    results = run_evaluation(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        schema=DEFAULT_SCHEMA,
        documents=docs,
        verbose=not args.quiet,
    )
    
    print(f"\nResults saved to: {output_dir / f'run_{results.run_id}'}")


if __name__ == "__main__":
    main()