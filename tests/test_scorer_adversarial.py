"""
10-case adversarial scorer validation.

GATE: All 10 must pass before any real evaluation.
"""
import pytest
from ctn_testing.core.types import (
    Extraction, 
    GroundTruth, 
    Evidence, 
    ExtractionStatus,
    Candidate
)
from ctn_testing.metrics.scorer import composite_score


def make_ext(
    field_name="total",
    value=None,
    quote=None,
    page=None,
    status="ok",
    candidates=None
) -> Extraction:
    return Extraction(
        field_name=field_name,
        value=value,
        evidence=Evidence(quote=quote, page=page),
        status=ExtractionStatus(status),
        candidates=candidates or []
    )


def make_gt(
    field_name="total",
    exists=True,
    ambiguous=False,
    value=None,
    candidate_values=None
) -> GroundTruth:
    return GroundTruth(
        field_name=field_name,
        exists_in_document=exists,
        is_ambiguous=ambiguous,
        value=value,
        acceptable_values=[value] if value else [],
        candidate_values=candidate_values or []
    )


class TestScorerAdversarial:
    """10 adversarial cases the scorer must handle correctly."""
    
    def test_case_1_wrong_value_real_quote(self):
        """Wrong value + real quote + correct page → composite ≤ 0.20"""
        doc = "Invoice\nTotal: 9999.99\nActual Amount: 1234.56"
        pages = [doc]
        
        ext = make_ext(value="9999.99", quote="Total: 9999.99", page=1, status="ok")
        gt = make_gt(exists=True, value="1234.56")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.composite <= 0.20, f"Wrong value scored {result.composite}"
        assert result.value == 0.0
        assert result.evidence == 0.0
        assert result.page == 0.0
        assert result.status == 0.0
    
    def test_case_2_correct_value_fabricated_quote(self):
        """Correct value + fabricated quote → evidence=0"""
        doc = "Invoice\nTotal: 1234.56"
        pages = [doc]
        
        ext = make_ext(value="1234.56", quote="This quote does not exist", page=1, status="ok")
        gt = make_gt(exists=True, value="1234.56")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 1.0
        assert result.evidence == 0.0
    
    def test_case_3_ambiguous_random_candidates(self):
        """Ambiguous GT + random candidates → low score"""
        doc = "Date: 2024-01-15\nEffective: 2024-02-01\nRandom: apple\nOther: banana"
        pages = [doc]
        
        gt = make_gt(
            field_name="effective_date",
            exists=True,
            ambiguous=True,
            candidate_values=["2024-01-15", "2024-02-01"]
        )
        
        ext = make_ext(
            field_name="effective_date",
            value=None,
            status="ambiguous",
            candidates=[
                Candidate(value="apple", quote="Random: apple", page=1),
                Candidate(value="banana", quote="Other: banana", page=1),
            ]
        )
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 0.0
        assert result.composite < 0.30
    
    def test_case_4_ambiguous_gt_but_status_ok(self):
        """Ambiguous GT but model outputs ok → penalized"""
        doc = "Date: 2024-01-15\nEffective: 2024-02-01"
        pages = [doc]
        
        gt = make_gt(
            field_name="date",
            exists=True,
            ambiguous=True,
            candidate_values=["2024-01-15", "2024-02-01"]
        )
        
        ext = make_ext(value="2024-01-15", quote="Date: 2024-01-15", page=1, status="ok")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 0.0
        assert result.status == 0.0
    
    def test_case_5_missing_gt_hallucinated_value(self):
        """Missing GT + hallucinated value → near 0"""
        doc = "Invoice\nNo total field here"
        pages = [doc]
        
        gt = make_gt(exists=False)
        ext = make_ext(value="100.00", quote="No total field here", page=1, status="ok")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 0.0
        assert result.composite < 0.20
    
    def test_case_6_missing_with_nonnull_quote(self):
        """Missing GT + correct status but includes quote → evidence=0"""
        doc = "Invoice\nSome text"
        pages = [doc]
        
        gt = make_gt(exists=False)
        ext = make_ext(value=None, quote="Some text", page=1, status="missing")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 1.0
        assert result.evidence == 0.0
    
    def test_case_7_duplicate_fields(self):
        """Single extraction with valid schema scores correctly."""
        doc = "Total: 100.00"
        pages = [doc]
        
        ext = make_ext(value="100.00", quote="Total: 100.00", page=1, status="ok")
        gt = make_gt(exists=True, value="100.00")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.schema == 1.0
    
    def test_case_8_field_omitted(self):
        """Omitted field for existing GT → miss"""
        doc = "Invoice text"
        pages = [doc]
        
        ext = make_ext(value=None, quote=None, page=None, status="missing")
        gt = make_gt(exists=True, value="expected_value")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 0.0
    
    def test_case_9_quote_doesnt_contain_value(self):
        """Quote doesn't contain value substring → evidence=0"""
        doc = "Total amount: 1234.56\nSummary: Payment received"
        pages = [doc]
        
        ext = make_ext(value="1234.56", quote="Summary: Payment received", page=1, status="ok")
        gt = make_gt(exists=True, value="1234.56")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 1.0
        assert result.evidence == 0.0
    
    def test_case_10_page_off_by_one(self):
        """Page number off-by-one → page=0"""
        page1 = "Page 1 content"
        page2 = "Total: 1234.56"
        pages = [page1, page2]
        doc = "\n".join(pages)
        
        ext = make_ext(value="1234.56", quote="Total: 1234.56", page=1, status="ok")
        gt = make_gt(exists=True, value="1234.56")
        
        result = composite_score(ext, gt, doc, pages)
        
        assert result.value == 1.0
        assert result.evidence > 0
        assert result.page == 0.0


def test_all_adversarial_cases():
    """Meta-test: all 10 cases exist and pass."""
    t = TestScorerAdversarial()
    
    cases = [
        t.test_case_1_wrong_value_real_quote,
        t.test_case_2_correct_value_fabricated_quote,
        t.test_case_3_ambiguous_random_candidates,
        t.test_case_4_ambiguous_gt_but_status_ok,
        t.test_case_5_missing_gt_hallucinated_value,
        t.test_case_6_missing_with_nonnull_quote,
        t.test_case_7_duplicate_fields,
        t.test_case_8_field_omitted,
        t.test_case_9_quote_doesnt_contain_value,
        t.test_case_10_page_off_by_one,
    ]
    
    assert len(cases) == 10
    
    for case in cases:
        case()
    
    print("✓ All 10 adversarial cases passed. Scorer is valid.")


if __name__ == "__main__":
    test_all_adversarial_cases()
