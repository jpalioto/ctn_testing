"""
Extraction scoring with correctness gating.

Evidence/page/status scores are ZEROED if value is wrong.
This prevents "cite anything plausible" from scoring high.
"""
import unicodedata

from ..core.types import (
    Extraction, 
    GroundTruth, 
    ExtractionStatus, 
    CompositeResult,
    Candidate
)


def normalize_minimal(text: str) -> str:
    """Unicode NFKC + whitespace collapse. NO case folding."""
    if text is None:
        return ""
    text = unicodedata.normalize('NFKC', text)
    return ' '.join(text.split())


def normalize_value(value) -> str:
    """Normalize value for comparison (includes case fold)."""
    if value is None:
        return ""
    return normalize_minimal(str(value)).lower()


def values_match(extracted, ground_truth_values: list) -> bool:
    """Check if extracted value matches any acceptable ground truth value."""
    if extracted is None:
        return False
    ext_norm = normalize_value(extracted)
    for gt in ground_truth_values:
        if normalize_value(gt) == ext_norm:
            return True
    return False


def value_accuracy(ext: Extraction, gt: GroundTruth) -> float:
    """
    Compute value accuracy score.
    
    Branches:
    - Ambiguous: F1 of predicted vs GT candidates
    - Missing: 1.0 if correctly identified, 0.0 if hallucinated
    - Present: 1.0 if correct, 0.0 if wrong
    """
    if gt.is_ambiguous:
        if ext.status != ExtractionStatus.AMBIGUOUS or not ext.candidates:
            return 0.0
        
        gt_values = set(normalize_value(v) for v in gt.candidate_values)
        pred_values = set(normalize_value(c.value) for c in ext.candidates)
        
        if not gt_values:
            return 1.0 if not pred_values else 0.5
        
        intersection = len(gt_values & pred_values)
        recall = intersection / len(gt_values) if gt_values else 0
        precision = intersection / len(pred_values) if pred_values else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    if not gt.exists_in_document:
        if ext.value is None and ext.status == ExtractionStatus.MISSING:
            return 1.0
        return 0.0
    
    if ext.value is None:
        return 0.0
    
    acceptable = gt.acceptable_values or [gt.value]
    if values_match(ext.value, acceptable):
        return 1.0
    
    return 0.0


def quote_exists_in_document(quote: str | None, document_text: str) -> bool:
    """Check if quote exists verbatim in document."""
    if quote is None:
        return False
    return normalize_minimal(quote) in normalize_minimal(document_text)


def quote_contains_value(quote: str | None, value) -> bool:
    """Check if quote contains the extracted value."""
    if quote is None or value is None:
        return False
    return normalize_minimal(str(value)) in normalize_minimal(quote)


def evidence_validity(
    ext: Extraction,
    gt: GroundTruth,
    document_text: str,
    judge_fn=None
) -> float:
    """
    Compute evidence validity score.
    
    Gates:
    1. Quote exists in document
    2. Quote contains value
    Then: token efficiency penalty + optional judge
    """
    if not gt.exists_in_document:
        return 1.0 if ext.evidence.quote is None else 0.0
    
    if gt.is_ambiguous:
        return candidate_evidence_score(ext.candidates, document_text, judge_fn)
    
    if ext.value is None:
        return 0.0
    
    quote = ext.evidence.quote
    value = str(ext.value)
    
    if quote is None:
        return 0.0
    
    if not quote_exists_in_document(quote, document_text):
        return 0.0
    
    if not quote_contains_value(quote, value):
        return 0.0
    
    token_ratio = len(quote.split()) / max(len(value.split()), 1)
    token_score = max(0.0, min(1.0, 1.0 - (token_ratio - 5) * 0.1))
    
    if judge_fn is not None:
        judge_score = judge_fn(quote, ext.field, value)
        return 0.3 * token_score + 0.7 * judge_score
    
    return token_score


def candidate_evidence_score(
    candidates: list[Candidate],
    document_text: str,
    judge_fn=None
) -> float:
    """Score evidence quality across candidates for ambiguous fields."""
    if not candidates:
        return 0.0
    
    scores = []
    for c in candidates:
        if not c.quote:
            scores.append(0.0)
            continue
        
        if not quote_exists_in_document(c.quote, document_text):
            scores.append(0.0)
            continue
        
        if not quote_contains_value(c.quote, c.value):
            scores.append(0.0)
            continue
        
        if judge_fn is not None:
            scores.append(judge_fn(c.quote, "candidate", str(c.value)))
        else:
            scores.append(1.0)
    
    return sum(scores) / len(scores) if scores else 0.0


def find_page_containing_quote(quote: str, pages: list[str]) -> int | None:
    """Find which page contains the quote (1-indexed)."""
    if quote is None:
        return None
    norm_quote = normalize_minimal(quote)
    for i, page in enumerate(pages):
        if norm_quote in normalize_minimal(page):
            return i + 1
    return None


def page_accuracy(ext: Extraction, gt: GroundTruth, pages: list[str]) -> float:
    """Verify evidence.page matches the page containing the quote."""
    if gt.is_ambiguous:
        if not ext.candidates:
            return 0.0
        correct = 0
        for c in ext.candidates:
            if c.quote and c.page:
                actual = find_page_containing_quote(c.quote, pages)
                if actual == c.page:
                    correct += 1
        return correct / len(ext.candidates)
    
    if not gt.exists_in_document:
        return 1.0 if ext.evidence.page is None else 0.0
    
    if ext.evidence.quote is None or ext.evidence.page is None:
        return 0.0
    
    actual = find_page_containing_quote(ext.evidence.quote, pages)
    return 1.0 if actual == ext.evidence.page else 0.0


def is_valid_schema(ext: Extraction) -> bool:
    """Check if extraction has valid schema structure."""
    if ext.field is None:
        return False
    if ext.status == ExtractionStatus.OK:
        return ext.evidence.quote is not None and ext.evidence.page is not None
    if ext.status == ExtractionStatus.MISSING:
        return ext.value is None
    if ext.status == ExtractionStatus.AMBIGUOUS:
        return ext.value is None and len(ext.candidates) >= 2
    return True


def composite_score(
    ext: Extraction,
    gt: GroundTruth,
    document_text: str,
    pages: list[str],
    judge_fn=None
) -> CompositeResult:
    """
    Compute weighted composite with correctness gating.
    
    Evidence/page/status are ZEROED if value is wrong.
    """
    weights = {
        'value': 0.30,
        'evidence': 0.30,
        'page': 0.10,
        'status': 0.15,
        'schema': 0.15
    }
    
    val = value_accuracy(ext, gt)
    schema = 1.0 if is_valid_schema(ext) else 0.0
    
    if gt.exists_in_document and not gt.is_ambiguous:
        if val == 0.0:
            evidence = 0.0
            page = 0.0
            status = 0.0
        else:
            evidence = evidence_validity(ext, gt, document_text, judge_fn)
            page = page_accuracy(ext, gt, pages)
            status = 1.0 if ext.status == ExtractionStatus.OK else 0.0
    
    elif gt.is_ambiguous:
        cand_f1 = val
        if cand_f1 == 0.0:
            evidence = 0.0
            page = 0.0
            status = 0.0
        else:
            raw_evidence = candidate_evidence_score(ext.candidates, document_text, judge_fn)
            raw_page = page_accuracy(ext, gt, pages)
            evidence = cand_f1 * raw_evidence
            page = cand_f1 * raw_page
            status = 1.0 if ext.status == ExtractionStatus.AMBIGUOUS else 0.0
    
    else:
        if val == 1.0:
            evidence = evidence_validity(ext, gt, document_text, judge_fn)
            page = page_accuracy(ext, gt, pages)
            status = 1.0 if ext.status == ExtractionStatus.MISSING else 0.0
        else:
            evidence = 0.0
            page = 0.0
            status = 0.0
    
    composite = (
        weights['value'] * val +
        weights['evidence'] * evidence +
        weights['page'] * page +
        weights['status'] * status +
        weights['schema'] * schema
    )
    
    return CompositeResult(
        composite=composite,
        value=val,
        evidence=evidence,
        page=page,
        status=status,
        schema=schema
    )
