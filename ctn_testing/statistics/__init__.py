"""Statistical analysis for kernel comparisons."""

from .comparison import (
    ComparisonResult,
    cohens_d_paired,
    document_level_aggregate,
    paired_comparison,
)
from .constraint_analysis import (
    ConstraintAnalysis,
    TraitComparison,
    analyze_constraint,
    format_report,
    full_analysis,
)

__all__ = [
    # Core comparison
    "ComparisonResult",
    "paired_comparison",
    "document_level_aggregate",
    "cohens_d_paired",
    # Constraint analysis
    "TraitComparison",
    "ConstraintAnalysis",
    "analyze_constraint",
    "full_analysis",
    "format_report",
]
