"""Statistical analysis for kernel comparisons."""
from .comparison import (
    ComparisonResult,
    paired_comparison,
    document_level_aggregate,
    cohens_d_paired,
)
from .constraint_analysis import (
    TraitComparison,
    ConstraintAnalysis,
    analyze_constraint,
    full_analysis,
    format_report,
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
