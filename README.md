# CTN Testing Framework

Evaluation framework for Cognitive Tensor Network kernels.

## Structure

```
ctn-testing/
├── ctn_testing/           # Core library
│   ├── core/              # Types, kernel abstraction, config
│   ├── metrics/           # Scorer with correctness gating
│   ├── statistics/        # Paired comparisons, effect sizes
│   ├── runners/           # Evaluation execution
│   └── utils/             # Normalization, helpers
├── domains/               # Domain-specific tests
│   └── extraction/        # Document extraction
│       ├── kernels/       # CTN, idiomatic, etc.
│       ├── schemas/       # Field definitions
│       ├── configs/       # Phase configs
│       └── data/          # DEV/TEST splits
├── scripts/               # CLI entry points
└── tests/                 # Unit tests
```

## Quick Start

```bash
# Install
pip install -e .

# Run scorer validation (REQUIRED before any evaluation)
pytest tests/test_scorer_adversarial.py -v

# Generate synthetic data
python scripts/generate.py --domain extraction --config domains/extraction/configs/phase1.yaml

# Run evaluation on DEV set
python scripts/evaluate.py --domain extraction --set dev

# Run final TEST (single run, pre-registered)
python scripts/evaluate.py --domain extraction --set test --final
```

## Scorer Validation

Before any real evaluation, the scorer must pass 10 adversarial cases:

1. Wrong value + real quote → composite ≤ 0.20
2. Correct value + fabricated quote → evidence=0
3. Ambiguous + random candidates → low score
4. Ambiguous GT but status=ok → penalized
5. Missing GT + hallucinated value → near 0
6. Missing + non-null quote → evidence=0
7. Duplicate fields → schema failure
8. Omitted field → miss
9. Quote doesn't contain value → evidence=0
10. Page off-by-one → page=0

## Key Design Decisions

**Correctness Gating:** Evidence, page, and status scores are ZEROED if value is wrong. This prevents "cite anything plausible" from scoring high.

**DEV/TEST Split:** 100 DEV (iterate freely) + 100 TEST (locked, single run). Prevents p-hacking.

**Boundary Typing:** Type hints on public interfaces, trust internals.

## License

MIT
