"""Statistical analysis for constraint adherence comparisons."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .comparison import paired_comparison, ComparisonResult

if TYPE_CHECKING:
    from ..runners.evaluation import PairedComparison, EvaluationResult


@dataclass
class TraitComparison:
    """Statistical comparison for a single trait."""
    trait: str
    baseline_mean: float
    test_mean: float
    mean_diff: float
    t_stat: float
    p_value: float
    effect_size: float        # Cohen's d
    n: int
    significant: bool         # p < 0.05

    @property
    def effect_interpretation(self) -> str:
        """Interpret effect size magnitude."""
        d = abs(self.effect_size)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        return "large"

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        return ""


@dataclass
class ConstraintAnalysis:
    """Analysis of a single constraint vs baseline."""
    constraint: str
    n_prompts: int
    trait_comparisons: dict[str, TraitComparison] = field(default_factory=dict)

    def significant_traits(self) -> list[str]:
        """Traits where p < 0.05."""
        return [
            name for name, comp in self.trait_comparisons.items()
            if comp.significant
        ]

    def best_trait(self) -> str | None:
        """Trait with largest positive effect size.

        Returns None if no traits have positive effect.
        """
        best_name = None
        best_effect = 0.0

        for name, comp in self.trait_comparisons.items():
            if comp.effect_size > best_effect:
                best_effect = comp.effect_size
                best_name = name

        return best_name

    def improved_traits(self) -> list[str]:
        """Traits with significant positive effect (p < 0.05, mean_diff > 0)."""
        return [
            name for name, comp in self.trait_comparisons.items()
            if comp.significant and comp.mean_diff > 0
        ]

    def degraded_traits(self) -> list[str]:
        """Traits with significant negative effect (p < 0.05, mean_diff < 0)."""
        return [
            name for name, comp in self.trait_comparisons.items()
            if comp.significant and comp.mean_diff < 0
        ]


def analyze_constraint(
    comparisons: list["PairedComparison"],
    constraint: str,
) -> ConstraintAnalysis:
    """Analyze all comparisons for one constraint.

    For each trait:
    - Extract baseline and test scores across all prompts
    - Run paired t-test
    - Compute effect size

    Args:
        comparisons: List of paired comparisons for this constraint
        constraint: Name of the constraint being analyzed

    Returns:
        ConstraintAnalysis with trait-level statistics
    """
    # Filter to valid comparisons (no errors, with scores)
    valid_comparisons = [
        c for c in comparisons
        if not c.error
        and not c.judging_result.error
        and c.get_baseline_scores()
        and c.get_test_scores()
    ]

    if not valid_comparisons:
        return ConstraintAnalysis(
            constraint=constraint,
            n_prompts=0,
            trait_comparisons={},
        )

    # Collect all trait names across comparisons
    all_traits: set[str] = set()
    for comp in valid_comparisons:
        all_traits.update(comp.get_baseline_scores().keys())
        all_traits.update(comp.get_test_scores().keys())

    # Analyze each trait
    trait_comparisons: dict[str, TraitComparison] = {}

    for trait in sorted(all_traits):
        baseline_scores: list[float] = []
        test_scores: list[float] = []

        for comp in valid_comparisons:
            baseline = comp.get_baseline_scores().get(trait)
            test = comp.get_test_scores().get(trait)

            if baseline is not None and test is not None:
                baseline_scores.append(baseline.score)
                test_scores.append(test.score)

        if len(baseline_scores) < 2:
            # Need at least 2 pairs for t-test
            if len(baseline_scores) == 1:
                # Single comparison - report descriptive stats only
                trait_comparisons[trait] = TraitComparison(
                    trait=trait,
                    baseline_mean=baseline_scores[0],
                    test_mean=test_scores[0],
                    mean_diff=test_scores[0] - baseline_scores[0],
                    t_stat=0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    n=1,
                    significant=False,
                )
            continue

        # Run paired comparison (test - baseline)
        result = paired_comparison(test_scores, baseline_scores)

        trait_comparisons[trait] = TraitComparison(
            trait=trait,
            baseline_mean=sum(baseline_scores) / len(baseline_scores),
            test_mean=sum(test_scores) / len(test_scores),
            mean_diff=result.mean_diff,
            t_stat=result.t_stat,
            p_value=result.p_value,
            effect_size=result.effect_size,
            n=result.n,
            significant=result.significant_at_05,
        )

    return ConstraintAnalysis(
        constraint=constraint,
        n_prompts=len(valid_comparisons),
        trait_comparisons=trait_comparisons,
    )


def full_analysis(result: "EvaluationResult") -> dict[str, ConstraintAnalysis]:
    """Analyze all constraints in evaluation result.

    Args:
        result: EvaluationResult from ConstraintEvaluator.run()

    Returns:
        Dict mapping constraint name to ConstraintAnalysis
    """
    analyses: dict[str, ConstraintAnalysis] = {}

    # Get unique test constraints
    constraints = set(c.test_constraint for c in result.comparisons)

    for constraint in sorted(constraints):
        comparisons = result.get_comparisons_for(constraint)
        analyses[constraint] = analyze_constraint(comparisons, constraint)

    return analyses


def format_report(analyses: dict[str, ConstraintAnalysis]) -> str:
    """Format human-readable report.

    Args:
        analyses: Dict from full_analysis()

    Returns:
        Formatted string report
    """
    lines = [
        "Constraint Adherence Analysis",
        "=" * 29,
        "",
    ]

    for constraint, analysis in sorted(analyses.items()):
        lines.append(f"@{constraint} (n={analysis.n_prompts} prompts)")

        if not analysis.trait_comparisons:
            lines.append("  No valid comparisons")
            lines.append("")
            continue

        # Sort traits by absolute effect size (largest first)
        sorted_traits = sorted(
            analysis.trait_comparisons.items(),
            key=lambda x: abs(x[1].effect_size),
            reverse=True,
        )

        for trait, comp in sorted_traits:
            # Format: trait_name: +/-X.X (p=0.XXX*, d=X.XX interpretation)
            sign = "+" if comp.mean_diff >= 0 else ""
            p_str = _format_p_value(comp.p_value)

            line = (
                f"  {trait:20s} {sign}{comp.mean_diff:5.1f} "
                f"(p={p_str}{comp.significance_stars}, "
                f"d={comp.effect_size:.2f} {comp.effect_interpretation})"
            )
            lines.append(line)

        # Summary of significant effects
        improved = analysis.improved_traits()
        degraded = analysis.degraded_traits()

        if improved:
            lines.append(f"  Significant improvements: {', '.join(improved)}")
        if degraded:
            lines.append(f"  Significant degradations: {', '.join(degraded)}")
        if not improved and not degraded:
            lines.append("  No significant effects")

        lines.append("")

    return "\n".join(lines)


def _format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"
