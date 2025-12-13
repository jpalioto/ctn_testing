"""Statistical analysis for kernel comparisons."""
from .comparison import (
    ComparisonResult,
    paired_comparison,
    document_level_aggregate,
    cohens_d_paired,
)

__all__ = [
    "ComparisonResult",
    "paired_comparison",
    "document_level_aggregate",
    "cohens_d_paired",
]
