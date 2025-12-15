"""Statistical analysis for kernel comparisons."""
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    mean_diff: float
    t_stat: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    n: int
    
    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05
    
    @property
    def effect_interpretation(self) -> str:
        d = abs(self.effect_size)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        return "large"


def cohens_d_paired(x: list[float], y: list[float]) -> float:
    """Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0


def paired_comparison(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Paired t-test at document level.
    
    Args:
        scores_a: Scores from kernel A (one per document)
        scores_b: Scores from kernel B (one per document)
        alpha: Significance level for CI
    
    Returns:
        ComparisonResult with all statistics
    """
    assert len(scores_a) == len(scores_b), "Must have same number of documents"
    
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    n = len(diff)

    if diff.std(ddof=1) == 0:
        return ComparisonResult(
            mean_diff=float(diff.mean()),
            t_stat=0.0,
            p_value=1.0,  # No evidence of difference
            effect_size=0.0,
            ci_lower=float(diff.mean()),
            ci_upper=float(diff.mean()),
            n=n
        )
    
    t_stat, p_value = stats.ttest_rel(a, b)

    
    t_stat, p_value = stats.ttest_rel(a, b)
    effect_size = cohens_d_paired(scores_a, scores_b)
    
    # 95% CI for mean difference
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    ci_lower = diff.mean() - t_crit * se
    ci_upper = diff.mean() + t_crit * se
    
    return ComparisonResult(
        mean_diff=float(diff.mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
        effect_size=float(effect_size),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n=n
    )


def document_level_aggregate(field_scores: dict[str, float]) -> float:
    """Average score across fields within a document."""
    if not field_scores:
        return 0.0
    return sum(field_scores.values()) / len(field_scores)
