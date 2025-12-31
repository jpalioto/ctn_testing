# scripts/smoke_test.py
"""
Vertical slice: one doc, one kernel, one model.
Verifies the entire pipeline wires up.

Usage: python scripts/smoke_test.py
"""

import json

from ctn_testing.core import (
    DocumentSchema,
    Extraction,
    FieldSchema,
    GroundTruth,
    ModelConfig,
    get_client,
)
from ctn_testing.metrics import composite_score

# Hardcoded test document
DOCUMENT = """
INVOICE

Invoice Number: INV-2024-0042
Date: December 13, 2024
Due Date: January 13, 2025

Bill To:
Acme Corporation
123 Main Street

From:
TechSupply Co.

Items:
- Widget A (qty 10) - $100.00
- Widget B (qty 5) - $75.00
- Shipping - $25.00

Subtotal: $200.00
Tax (10%): $20.00
Total Amount Due: $220.00

Payment Terms: Net 30
"""

PAGES = [DOCUMENT]  # Single page

# Schema
SCHEMA = DocumentSchema(
    name="invoice",
    fields=[
        FieldSchema(name="invoice_number", description="Unique invoice identifier"),
        FieldSchema(name="total_amount", description="Total amount due including tax"),
        FieldSchema(name="due_date", description="Payment due date"),
        FieldSchema(name="vendor_name", description="Company issuing the invoice"),
    ],
)

# Ground truth
GROUND_TRUTH = {
    "invoice_number": GroundTruth(
        field_name="invoice_number",
        exists_in_document=True,
        value="INV-2024-0042",
    ),
    "total_amount": GroundTruth(
        field_name="total_amount",
        exists_in_document=True,
        value="$220.00",
        acceptable_values=["$220.00", "220.00", "220"],
    ),
    "due_date": GroundTruth(
        field_name="due_date",
        exists_in_document=True,
        value="January 13, 2025",
        acceptable_values=["January 13, 2025", "2025-01-13", "01/13/2025"],
    ),
    "vendor_name": GroundTruth(
        field_name="vendor_name",
        exists_in_document=True,
        value="TechSupply Co.",
        acceptable_values=["TechSupply Co.", "TechSupply Co", "TechSupply"],
    ),
}

# Output format for kernel
OUTPUT_FORMAT = {
    "extractions": [
        {
            "field": "<field_name>",
            "value": "<extracted_value_or_null>",
            "evidence": {"quote": "<verbatim_quote_from_document>", "page": 1},
            "status": "ok | missing | ambiguous",
            "confidence": "high | medium | low",
            "candidates": [],
        }
    ]
}

# Kernel template (inline for smoke test)
KERNEL_TEMPLATE = """You are a document extraction assistant. Extract the requested fields from the document.

{schema}

For each field:
- Extract the value if present
- Provide the exact quote from the document that contains the value
- Note the page number where you found it
- If the field is not in the document, mark it as missing

Return your response as JSON:
{output_format}

DOCUMENT:
{document}
"""


def run_smoke_test():
    print("=" * 60)
    print("CTN TESTING - SMOKE TEST")
    print("=" * 60)

    # Check for API key
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[ERROR] ANTHROPIC_API_KEY not set.")
        print("Run: . .\\scripts\\activate.ps1")
        return

    # Build prompt
    prompt = KERNEL_TEMPLATE.format(
        schema=SCHEMA.to_prompt(),
        output_format=json.dumps(OUTPUT_FORMAT, indent=2),
        document=DOCUMENT,
    )

    print(f"\n[1/4] Prompt built ({len(prompt.split())} tokens approx)")

    # Call model
    config = ModelConfig(
        provider="anthropic",
        name="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.0,
        max_tokens=1024,
    )

    print(f"[2/4] Calling {config.name}...")
    client = get_client(config)
    result = client.complete(prompt)

    print(f"      Tokens: {result.input_tokens} in, {result.output_tokens} out")

    # Parse response
    print("[3/4] Parsing response...")
    try:
        text = result.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        data = json.loads(text)
        extractions = [Extraction.from_dict(e) for e in data["extractions"]]
        print(f"      Parsed {len(extractions)} extractions")
    except Exception as e:
        print(f"      [ERROR] Parse failed: {e}")
        print(f"      Raw response:\n{result.text[:500]}")
        return

    # Score
    print("[4/4] Scoring...")
    print()

    total_composite = 0.0
    for ext in extractions:
        gt = GROUND_TRUTH.get(ext.field_name)
        if not gt:
            print(f"  {ext.field_name}: [SKIP] No ground truth")
            continue

        score = composite_score(ext, gt, DOCUMENT, PAGES)
        total_composite += score.composite

        status = "✓" if score.value == 1.0 else "✗"
        print(f"  {ext.field_name}:")
        print(f"    Extracted: {ext.value}")
        print(f"    Expected:  {gt.value}")
        print(
            f"    Value: {score.value:.2f} | Evidence: {score.evidence:.2f} | Page: {score.page:.2f}"
        )
        print(f"    Composite: {score.composite:.2f} {status}")
        print()

    avg = total_composite / len(GROUND_TRUTH)
    print("=" * 60)
    print(f"AVERAGE COMPOSITE: {avg:.2f}")
    print("=" * 60)

    if avg > 0.8:
        print("\n[PASS] Pipeline works.")
    else:
        print("\n[WARN] Low score. Check extractions above.")


if __name__ == "__main__":
    run_smoke_test()
