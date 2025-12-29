"""Tests for constraint adherence statistical analysis."""
import pytest

from ctn_testing.statistics.constraint_analysis import (
    TraitComparison,
    ConstraintAnalysis,
    analyze_constraint,
    full_analysis,
    format_report,
)
from ctn_testing.judging.blind_judge import JudgingResult, TraitScore

# Import evaluation types after other imports to avoid circular import
from ctn_testing.runners.evaluation import PairedComparison, EvaluationResult


def make_trait_score(trait: str, score: int) -> TraitScore:
    """Helper to create TraitScore."""
    return TraitScore(dimension=trait, score=score, reasons=[])


def make_judging_result(
    baseline_scores: dict[str, int],
    test_scores: dict[str, int],
    baseline_was_a: bool = True,
) -> JudgingResult:
    """Helper to create JudgingResult with scores."""
    if baseline_was_a:
        a_scores = {k: make_trait_score(k, v) for k, v in baseline_scores.items()}
        b_scores = {k: make_trait_score(k, v) for k, v in test_scores.items()}
    else:
        a_scores = {k: make_trait_score(k, v) for k, v in test_scores.items()}
        b_scores = {k: make_trait_score(k, v) for k, v in baseline_scores.items()}

    return JudgingResult(
        response_a_scores=a_scores,
        response_b_scores=b_scores,
    )


def make_comparison(
    prompt_id: str,
    baseline_scores: dict[str, int],
    test_scores: dict[str, int],
    baseline_was_a: bool = True,
    constraint: str = "analytical",
) -> PairedComparison:
    """Helper to create PairedComparison."""
    return PairedComparison(
        prompt_id=prompt_id,
        prompt_text=f"Prompt {prompt_id}",
        baseline_constraint="baseline",
        test_constraint=constraint,
        baseline_response="baseline response",
        test_response="test response",
        judging_result=make_judging_result(baseline_scores, test_scores, baseline_was_a),
        baseline_was_a=baseline_was_a,
    )


class TestTraitComparison:
    """Tests for TraitComparison dataclass."""

    def test_effect_interpretation_negligible(self):
        """Effect size < 0.2 is negligible."""
        comp = TraitComparison(
            trait="test",
            baseline_mean=50,
            test_mean=52,
            mean_diff=2,
            t_stat=0.5,
            p_value=0.6,
            effect_size=0.1,
            n=10,
            significant=False,
        )
        assert comp.effect_interpretation == "negligible"

    def test_effect_interpretation_small(self):
        """Effect size 0.2-0.5 is small."""
        comp = TraitComparison(
            trait="test",
            baseline_mean=50,
            test_mean=55,
            mean_diff=5,
            t_stat=1.5,
            p_value=0.15,
            effect_size=0.35,
            n=10,
            significant=False,
        )
        assert comp.effect_interpretation == "small"

    def test_effect_interpretation_medium(self):
        """Effect size 0.5-0.8 is medium."""
        comp = TraitComparison(
            trait="test",
            baseline_mean=50,
            test_mean=60,
            mean_diff=10,
            t_stat=2.5,
            p_value=0.03,
            effect_size=0.65,
            n=10,
            significant=True,
        )
        assert comp.effect_interpretation == "medium"

    def test_effect_interpretation_large(self):
        """Effect size >= 0.8 is large."""
        comp = TraitComparison(
            trait="test",
            baseline_mean=50,
            test_mean=70,
            mean_diff=20,
            t_stat=4.0,
            p_value=0.001,
            effect_size=1.2,
            n=10,
            significant=True,
        )
        assert comp.effect_interpretation == "large"

    def test_significance_stars(self):
        """Correct significance stars for p-values."""
        # p >= 0.05: no stars
        comp = TraitComparison(
            trait="t", baseline_mean=0, test_mean=0, mean_diff=0,
            t_stat=0, p_value=0.1, effect_size=0, n=10, significant=False,
        )
        assert comp.significance_stars == ""

        # p < 0.05: one star
        comp = TraitComparison(
            trait="t", baseline_mean=0, test_mean=0, mean_diff=0,
            t_stat=0, p_value=0.03, effect_size=0, n=10, significant=True,
        )
        assert comp.significance_stars == "*"

        # p < 0.01: two stars
        comp = TraitComparison(
            trait="t", baseline_mean=0, test_mean=0, mean_diff=0,
            t_stat=0, p_value=0.005, effect_size=0, n=10, significant=True,
        )
        assert comp.significance_stars == "**"

        # p < 0.001: three stars
        comp = TraitComparison(
            trait="t", baseline_mean=0, test_mean=0, mean_diff=0,
            t_stat=0, p_value=0.0005, effect_size=0, n=10, significant=True,
        )
        assert comp.significance_stars == "***"


class TestConstraintAnalysis:
    """Tests for ConstraintAnalysis dataclass."""

    def test_significant_traits(self):
        """Filter traits by significance."""
        analysis = ConstraintAnalysis(
            constraint="analytical",
            n_prompts=10,
            trait_comparisons={
                "reasoning": TraitComparison(
                    trait="reasoning", baseline_mean=50, test_mean=70,
                    mean_diff=20, t_stat=3.0, p_value=0.01, effect_size=0.9,
                    n=10, significant=True,
                ),
                "conciseness": TraitComparison(
                    trait="conciseness", baseline_mean=60, test_mean=55,
                    mean_diff=-5, t_stat=-1.0, p_value=0.3, effect_size=-0.2,
                    n=10, significant=False,
                ),
            },
        )

        assert analysis.significant_traits() == ["reasoning"]

    def test_best_trait(self):
        """Find trait with largest positive effect."""
        analysis = ConstraintAnalysis(
            constraint="analytical",
            n_prompts=10,
            trait_comparisons={
                "reasoning": TraitComparison(
                    trait="reasoning", baseline_mean=50, test_mean=70,
                    mean_diff=20, t_stat=3.0, p_value=0.01, effect_size=0.9,
                    n=10, significant=True,
                ),
                "structure": TraitComparison(
                    trait="structure", baseline_mean=50, test_mean=65,
                    mean_diff=15, t_stat=2.5, p_value=0.02, effect_size=0.7,
                    n=10, significant=True,
                ),
            },
        )

        assert analysis.best_trait() == "reasoning"

    def test_best_trait_no_positive(self):
        """Return None if no positive effects."""
        analysis = ConstraintAnalysis(
            constraint="terse",
            n_prompts=10,
            trait_comparisons={
                "completeness": TraitComparison(
                    trait="completeness", baseline_mean=70, test_mean=60,
                    mean_diff=-10, t_stat=-2.0, p_value=0.05, effect_size=-0.5,
                    n=10, significant=True,
                ),
            },
        )

        assert analysis.best_trait() is None

    def test_improved_traits(self):
        """Filter traits with significant positive effect."""
        analysis = ConstraintAnalysis(
            constraint="analytical",
            n_prompts=10,
            trait_comparisons={
                "reasoning": TraitComparison(
                    trait="reasoning", baseline_mean=50, test_mean=70,
                    mean_diff=20, t_stat=3.0, p_value=0.01, effect_size=0.9,
                    n=10, significant=True,
                ),
                "conciseness": TraitComparison(
                    trait="conciseness", baseline_mean=60, test_mean=50,
                    mean_diff=-10, t_stat=-2.0, p_value=0.04, effect_size=-0.5,
                    n=10, significant=True,
                ),
            },
        )

        assert analysis.improved_traits() == ["reasoning"]

    def test_degraded_traits(self):
        """Filter traits with significant negative effect."""
        analysis = ConstraintAnalysis(
            constraint="analytical",
            n_prompts=10,
            trait_comparisons={
                "reasoning": TraitComparison(
                    trait="reasoning", baseline_mean=50, test_mean=70,
                    mean_diff=20, t_stat=3.0, p_value=0.01, effect_size=0.9,
                    n=10, significant=True,
                ),
                "conciseness": TraitComparison(
                    trait="conciseness", baseline_mean=60, test_mean=50,
                    mean_diff=-10, t_stat=-2.0, p_value=0.04, effect_size=-0.5,
                    n=10, significant=True,
                ),
            },
        )

        assert analysis.degraded_traits() == ["conciseness"]


class TestAnalyzeConstraint:
    """Tests for analyze_constraint function."""

    def test_computes_correct_means(self):
        """Means are computed correctly across comparisons."""
        comparisons = [
            make_comparison("p1", {"reasoning": 50}, {"reasoning": 70}),
            make_comparison("p2", {"reasoning": 60}, {"reasoning": 80}),
        ]

        analysis = analyze_constraint(comparisons, "analytical")

        assert analysis.trait_comparisons["reasoning"].baseline_mean == 55.0
        assert analysis.trait_comparisons["reasoning"].test_mean == 75.0
        assert analysis.trait_comparisons["reasoning"].mean_diff == 20.0

    def test_runs_paired_ttest(self):
        """Runs paired t-test on scores."""
        # Create comparisons with consistent improvement but some variability
        comparisons = [
            make_comparison("p1", {"reasoning": 40}, {"reasoning": 58}),
            make_comparison("p2", {"reasoning": 45}, {"reasoning": 67}),
            make_comparison("p3", {"reasoning": 50}, {"reasoning": 72}),
            make_comparison("p4", {"reasoning": 55}, {"reasoning": 73}),
        ]

        analysis = analyze_constraint(comparisons, "analytical")

        comp = analysis.trait_comparisons["reasoning"]
        assert comp.n == 4
        assert comp.t_stat != 0  # Should have non-zero t-stat
        assert comp.p_value < 0.05  # Should be significant

    def test_effect_size_computed(self):
        """Effect size (Cohen's d) is computed."""
        # Need variability in differences for Cohen's d to be meaningful
        comparisons = [
            make_comparison("p1", {"reasoning": 50}, {"reasoning": 78}),
            make_comparison("p2", {"reasoning": 52}, {"reasoning": 82}),
            make_comparison("p3", {"reasoning": 48}, {"reasoning": 80}),
            make_comparison("p4", {"reasoning": 51}, {"reasoning": 79}),
        ]

        analysis = analyze_constraint(comparisons, "analytical")

        # Large consistent effect should have large Cohen's d
        assert analysis.trait_comparisons["reasoning"].effect_size > 0.8

    def test_handles_single_comparison(self):
        """Handles n=1 gracefully (no t-test possible)."""
        comparisons = [
            make_comparison("p1", {"reasoning": 50}, {"reasoning": 70}),
        ]

        analysis = analyze_constraint(comparisons, "analytical")

        comp = analysis.trait_comparisons["reasoning"]
        assert comp.n == 1
        assert comp.mean_diff == 20.0
        assert comp.p_value == 1.0  # Cannot determine significance
        assert comp.significant is False

    def test_handles_missing_traits(self):
        """Handles traits missing in some comparisons."""
        comparisons = [
            make_comparison("p1", {"reasoning": 50, "conciseness": 60}, {"reasoning": 70, "conciseness": 50}),
            make_comparison("p2", {"reasoning": 55}, {"reasoning": 75}),  # Missing conciseness
        ]

        analysis = analyze_constraint(comparisons, "analytical")

        # Reasoning should have n=2
        assert analysis.trait_comparisons["reasoning"].n == 2
        # Conciseness should have n=1
        assert analysis.trait_comparisons["conciseness"].n == 1

    def test_handles_randomized_order(self):
        """Correctly unscrambles randomized A/B order."""
        # baseline_was_a=False means baseline is B
        comparisons = [
            make_comparison("p1", {"reasoning": 50}, {"reasoning": 70}, baseline_was_a=False),
            make_comparison("p2", {"reasoning": 60}, {"reasoning": 80}, baseline_was_a=True),
        ]

        analysis = analyze_constraint(comparisons, "analytical")

        # Should correctly compute baseline mean as 55, test mean as 75
        assert analysis.trait_comparisons["reasoning"].baseline_mean == 55.0
        assert analysis.trait_comparisons["reasoning"].test_mean == 75.0

    def test_handles_empty_comparisons(self):
        """Handles empty comparison list."""
        analysis = analyze_constraint([], "analytical")

        assert analysis.n_prompts == 0
        assert analysis.trait_comparisons == {}

    def test_handles_error_comparisons(self):
        """Filters out comparisons with errors."""
        valid = make_comparison("p1", {"reasoning": 50}, {"reasoning": 70})
        with_error = make_comparison("p2", {"reasoning": 55}, {"reasoning": 75})
        with_error.error = "Judge failed"

        analysis = analyze_constraint([valid, with_error], "analytical")

        # Only valid comparison should be counted
        assert analysis.n_prompts == 1


class TestFullAnalysis:
    """Tests for full_analysis function."""

    def test_analyzes_all_constraints(self):
        """Analyzes all constraints in result."""
        comparisons = [
            make_comparison("p1", {"r": 50}, {"r": 70}, constraint="analytical"),
            make_comparison("p1", {"r": 50}, {"r": 40}, constraint="terse"),
        ]

        result = EvaluationResult(
            config_name="test",
            timestamp="2025-01-01",
            comparisons=comparisons,
        )

        analyses = full_analysis(result)

        assert "analytical" in analyses
        assert "terse" in analyses


class TestFormatReport:
    """Tests for format_report function."""

    def test_produces_readable_output(self):
        """Report contains expected sections."""
        analyses = {
            "analytical": ConstraintAnalysis(
                constraint="analytical",
                n_prompts=10,
                trait_comparisons={
                    "reasoning_depth": TraitComparison(
                        trait="reasoning_depth",
                        baseline_mean=50,
                        test_mean=68.3,
                        mean_diff=18.3,
                        t_stat=3.5,
                        p_value=0.002,
                        effect_size=0.89,
                        n=10,
                        significant=True,
                    ),
                },
            ),
        }

        report = format_report(analyses)

        assert "Constraint Adherence Analysis" in report
        assert "@analytical" in report
        assert "n=10 prompts" in report
        assert "reasoning_depth" in report
        assert "18.3" in report  # Mean diff value (sign may have spacing)
        assert "0.002" in report
        assert "large" in report

    def test_shows_significant_improvements(self):
        """Report lists significant improvements."""
        analyses = {
            "terse": ConstraintAnalysis(
                constraint="terse",
                n_prompts=5,
                trait_comparisons={
                    "conciseness": TraitComparison(
                        trait="conciseness",
                        baseline_mean=50,
                        test_mean=81,
                        mean_diff=31,
                        t_stat=5.0,
                        p_value=0.0005,
                        effect_size=1.4,
                        n=5,
                        significant=True,
                    ),
                },
            ),
        }

        report = format_report(analyses)

        assert "Significant improvements: conciseness" in report

    def test_shows_no_significant_effects(self):
        """Report indicates when no significant effects."""
        analyses = {
            "casual": ConstraintAnalysis(
                constraint="casual",
                n_prompts=5,
                trait_comparisons={
                    "formality": TraitComparison(
                        trait="formality",
                        baseline_mean=50,
                        test_mean=48,
                        mean_diff=-2,
                        t_stat=-0.5,
                        p_value=0.6,
                        effect_size=-0.1,
                        n=5,
                        significant=False,
                    ),
                },
            ),
        }

        report = format_report(analyses)

        assert "No significant effects" in report

    def test_handles_empty_analyses(self):
        """Report handles constraint with no valid data."""
        analyses = {
            "broken": ConstraintAnalysis(
                constraint="broken",
                n_prompts=0,
                trait_comparisons={},
            ),
        }

        report = format_report(analyses)

        assert "No valid comparisons" in report
