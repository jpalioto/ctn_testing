# CTN Structured Extraction Evaluation Plan v2

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Previous | Initial plan |
| 2.0 | Current | Phased approach (Practical → Applied → Academic), architecture/execution split, red team feedback incorporated |

---

## Executive Summary

This document describes a **phased experimental methodology** for evaluating CTN (Cognitive Tensor Network) kernels for structured document extraction.

**Design Philosophy:** Build general, test specific.
- Architecture supports academic-level rigor
- Execution proceeds in phases, with gates
- No phase introduces technical debt that blocks later phases
- Stop early if no signal; extend only on positive results

**Three Phases:**

| Phase | Question | Rigor Level | Time | Gate |
|-------|----------|-------------|------|------|
| **1: Practical** | Does CTN work better? | Engineering | 3-4 days | >5% improvement |
| **2: Applied Research** | When and where does it work? | Replication | +1 week | Generalizes to real data |
| **3: Academic** | Why does it work? | Causal isolation | +2-3 weeks | Mechanism identified |

Each phase builds on the infrastructure of previous phases. No rewrites required.

---

## 1. Research Questions by Phase

### Phase 1: Practical
> **"Should engineers use CTN kernels for extraction?"**

- Does CTN outperform well-written English prompts?
- By how much?
- Is the improvement worth the complexity?

**Acceptable unknowns:** Why it works, which component helps, whether it's notation or length.

### Phase 2: Applied Research
> **"When and where does CTN help?"**

- Does the effect replicate on real documents?
- Which document types benefit most?
- Does it generalize to unseen domains (blind validation)?
- What are the boundary conditions?

**Acceptable unknowns:** The specific mechanism, cross-model generalization.

### Phase 3: Academic
> **"What is the mechanism by which CTN improves extraction?"**

- Is it the notation specifically, or the additional content?
- Is it prompt length/attention effects?
- Which CTN components (v₁-v₇) drive the effect?
- Does it generalize across model families?

**Acceptable unknowns:** None at this level. All confounds must be addressed.

---

## 2. Architecture (Built Once)

The following infrastructure is built in Phase 1 and used unchanged in all phases.

### 2.1 Directory Structure

```
ctn_extraction/
├── README.md
├── requirements.txt
├── run_evaluation.py           # Main entry point
│
├── core/
│   ├── __init__.py
│   ├── kernel.py               # Abstract kernel loader
│   ├── document.py             # Abstract document (text/PDF/image ready)
│   ├── schema.py               # Schema definitions
│   ├── extraction.py           # Run extraction against API
│   └── output.py               # Parse and validate model output
│
├── metrics/
│   ├── __init__.py
│   ├── base.py                 # Abstract metric interface
│   ├── value_accuracy.py       # Type-aware value comparison
│   ├── evidence_validity.py    # Quote verification
│   ├── null_handling.py        # Missing field detection
│   ├── ambiguity.py            # Ambiguity detection scoring
│   ├── schema_compliance.py    # JSON schema validation
│   ├── hallucination.py        # Invented value detection
│   └── composite.py            # Weighted combination
│
├── statistics/
│   ├── __init__.py
│   ├── descriptive.py          # Means, CIs, effect sizes
│   ├── paired_tests.py         # Doc-level paired comparisons
│   ├── clustered.py            # Cluster bootstrap (Phase 2+)
│   └── mixed_effects.py        # Full hierarchical models (Phase 3)
│
├── generators/
│   ├── __init__.py
│   ├── base.py                 # Abstract generator interface
│   ├── synthetic.py            # Template + LLM content generation
│   ├── loader.py               # Load documents from directory
│   └── templates/              # HTML templates per doc type
│       ├── invoice/
│       ├── contract/
│       └── ...
│
├── kernels/                    # Drop-in kernel text files
│   ├── ctn_extraction.txt
│   ├── idiomatic.txt
│   ├── literal.txt             # Added Phase 2
│   ├── idiomatic_long.txt      # Added Phase 3
│   └── placebo_math.txt        # Added Phase 3
│
├── baselines/                  # Added Phase 3
│   ├── null_baseline.py        # Always outputs missing
│   └── heuristic.py            # Regex-based extraction
│
├── schemas/                    # Document type definitions
│   ├── invoice.json
│   ├── contract.json
│   ├── medical.json
│   ├── receipt.json
│   └── resume.json
│
├── data/
│   ├── synthetic/              # Phase 1 generated
│   ├── real/                   # Phase 2 collected
│   └── external/               # Phase 3 blind validation
│
├── configs/
│   ├── phase1.yaml
│   ├── phase2.yaml
│   └── phase3.yaml
│
└── results/
    ├── phase1/
    ├── phase2/
    └── phase3/
```

### 2.2 Key Abstractions

**Kernel Interface:**
```python
class Kernel:
    """Abstract kernel - load any .txt file."""
    
    def __init__(self, path: str):
        self.text = load_file(path)
        self.name = extract_name(path)
    
    def render(self, schema: Schema, output_format: str) -> str:
        """Inject schema and output format into kernel template."""
        return self.text.replace("{{SCHEMA}}", schema.to_prompt())
                        .replace("{{OUTPUT_FORMAT}}", output_format)
```
Adding a new kernel = dropping a .txt file in `kernels/`.

**Document Interface:**
```python
class Document:
    """Abstract document - supports text, PDF, image."""
    
    def __init__(self, source: str, doc_type: str):
        self.source = source          # Path or content
        self.doc_type = doc_type      # text | pdf | image
        self.text = None              # Extracted text (for scoring)
        self.pages = []               # Page-separated text
    
    def to_api_format(self) -> dict:
        """Format for API call (text block or attachment)."""
        pass
    
    def get_scoring_text(self) -> str:
        """Text used for evidence verification."""
        pass
```
Phase 1 uses text-only. Phase 3 can add PDF without changing interface.

**Metric Interface:**
```python
class Metric:
    """Abstract metric - all metrics share this interface."""
    
    name: str
    
    def score(self, 
              extracted: Extraction, 
              ground_truth: GroundTruth,
              document: Document) -> float:
        """Return score in [0, 1]."""
        pass
    
    def aggregate(self, scores: List[float]) -> AggregateResult:
        """Aggregate across fields/documents."""
        pass
```
All metrics implemented in Phase 1, selectively reported per phase.

**Statistics Interface:**
```python
class StatisticalTest:
    """Abstract test - supports escalating complexity."""
    
    def compare(self, 
                scores_a: List[float], 
                scores_b: List[float],
                clustering: Optional[List[int]] = None) -> TestResult:
        """Compare two conditions, optionally with clustering."""
        pass
```
Phase 1 uses simple paired test. Phase 3 uses mixed-effects. Same interface.

### 2.3 Configuration Schema

```yaml
# All phases use same schema, different values
name: str                       # Phase name
version: str                    # Config version

documents:
  source: str                   # synthetic | directory | mixed
  path: str                     # For directory source
  synthetic:                    # For synthetic source
    types: List[DocTypeConfig]
    
kernels:
  - name: str
    path: str
    enabled: bool               # Can disable without removing
    
comparisons:                    # Which comparisons to report
  - a: str
    b: str
    primary: bool               # Is this a primary endpoint?
    
metrics:
  primary: List[str]            # Metrics to report as primary
  secondary: List[str]          # Metrics to compute but report separately
  composite_weights: Dict       # Weights for composite score
  
statistics:
  method: str                   # simple | clustered | mixed_effects
  alpha: float                  # Significance threshold
  correction: str               # none | bonferroni | holm
  cluster_by: str               # document | doc_type
  
output:
  results_dir: str
  save_raw: bool
  generate_report: bool
  
model:
  name: str
  temperature: float
  max_tokens: int

execution:
  delay: float
  retries: int
  timeout: int
```

---

## 3. Phase 1: Practical

### 3.1 Objective

Answer: **"Does CTN work better than good English prompts for extraction?"**

This is a head-to-head comparison. We test the CTN package (notation + content + structure) against an idiomatic English prompt. We do not isolate which component helps.

### 3.2 Test Arms

| Arm | Description | Purpose |
|-----|-------------|---------|
| **CTN** | Full CTN kernel with cognitive tensors | Test condition |
| **Idiomatic** | Best-effort concise English | Control |

**Why only 2 arms:**
- Minimal comparison needed to answer practical question
- Infrastructure supports more arms; we add them in later phases
- Reduces API cost and time for initial signal

### 3.3 Input Format

**Text-only with page markers.**

```
---PAGE 1---
INVOICE

Vendor: Acme Corporation
Invoice Number: INV-2024-0892
Date: March 15, 2024
...

---PAGE 2---
Line Items:
...
```

**Rationale:**
- Eliminates PDF representation mismatch (red team issue #1)
- Evidence scoring is well-defined (substring of exact text model sees)
- Infrastructure supports PDF; we defer to Phase 3 if needed

**Document class implementation:**
```python
def get_scoring_text(self) -> str:
    """Same text sent to model and used for scoring."""
    return self.text  # No transformation

def to_api_format(self) -> dict:
    """Send as text content."""
    return {"type": "text", "text": self.text}
```

### 3.4 Kernels

#### CTN Extraction Kernel

```
CTN_EXTRACTION_KERNEL(Σ_extract) ← {
    SYS_KERNEL_INIT(Ψ_global),
    COGNITIVE_TENSORS(U_extract),
    STRATEGIC_SOLVER(Ω_extract),
    DECODER_MANIFOLD(D_extract)
}

SYS_KERNEL_INIT(Ψ_global) ←
    { Auth: P_spec,
      Domain: structured_extraction,
      Constraint: extract_with_evidence }

COGNITIVE_TENSORS(U_extract):
    v₁ = { ε_hid → 0⁺ , Atomic_Clarity }
        # Parse each field independently. No blending.
    
    v₂ = { κ(f) → min , Extraction_Accuracy }
        # Extract exactly what appears. No invention.
    
    v₃ = { Φ:W→I , Context_Isolation }
        # Each field extraction is independent.
    
    v₄ = { π_gl ≫ π_loc , Schema_Over_Fluency }
        # Output must match schema exactly.
    
    v₅ = { δA = 0 , Reality_Anchor }
        # If not in document: value=null, status=missing.
        # If ambiguous: value=null, status=ambiguous, candidates required.
        # NEVER invent. NEVER fabricate quotes.
    
    v₆ = { U \ S , Minimal_Extraction }
        # Extract only requested fields.
    
    v₇ = { OutputSchema_Lock }
        # Valid JSON only. No prose.

STRATEGIC_SOLVER(Ω_extract):
    For each field f in schema:
        search(f) → {evidence_set}
        
        |evidence_set| = 0  → value=null, status=missing
        |evidence_set| = 1  → value=extract(evidence), status=ok
        |evidence_set| > 1  → value=null, status=ambiguous, 
                              candidates=all options with quotes

    CONSTRAINT: Every non-null value MUST have verbatim quote from document.
    CONSTRAINT: Every candidate MUST have verbatim quote from document.
    CONSTRAINT: Quotes must exist exactly in document text.

DECODER_MANIFOLD(D_extract):
    ℓ* = argmax_ℓ [
        SchemaValidity(ℓ)
      + EvidenceIntegrity(ℓ)
      - FabricatedContent(ℓ)
    ]

{{SCHEMA}}

{{OUTPUT_FORMAT}}
```

#### Idiomatic Extraction Kernel

```
Extract structured data from the document.

For each field in the schema:

1. Find the value in the document
2. Copy the EXACT quote where you found it (verbatim, character-for-character)
3. Record the page number

If a field is not in the document:
- value: null
- status: "missing"
- Do NOT make anything up

If multiple values could match:
- value: null  
- status: "ambiguous"
- List ALL candidates with their exact quotes
- Do NOT guess

Critical rules:
- Every value MUST have a quote that appears exactly in the document
- Every candidate MUST have a quote that appears exactly in the document
- If you cannot find an exact quote, the value must be null
- No paraphrasing quotes—copy exactly

{{SCHEMA}}

{{OUTPUT_FORMAT}}
```

#### Shared Output Format

Both kernels receive the same output format:

```
Output valid JSON only. No other text before or after.

{
  "extractions": [
    {
      "field": "<field_name>",
      "value": "<extracted_value or null>",
      "evidence": {
        "quote": "<verbatim_quote_from_document or null>",
        "page": <page_number or null>
      },
      "status": "<ok|missing|ambiguous>",
      "confidence": "<high|medium|low>",
      "candidates": [
        {"value": "<alternative>", "quote": "<exact_quote>"}
      ]
    }
  ]
}

Rules:
- If status is "missing": value must be null, evidence must be null
- If status is "ambiguous": value must be null, candidates must have ≥2 entries
- If status is "ok": value must be non-null, evidence.quote must be non-null
- Every quote (in evidence or candidates) must appear verbatim in the document
```

#### Consistent Policies Across Arms

| Policy | CTN | Idiomatic | Same? |
|--------|-----|-----------|-------|
| Missing → null | Yes | Yes | ✓ |
| Ambiguous → null + candidates | Yes | Yes | ✓ |
| No guessing | Yes | Yes | ✓ |
| Evidence required for value | Yes | Yes | ✓ |
| Candidates need quotes | Yes | Yes | ✓ |

This addresses red team issue #3 (ambiguity policy mismatch).

### 3.5 Metrics Specification

All metrics are implemented. Phase 1 reports primary metrics; secondary metrics are computed for exploratory analysis.

#### Scoring vs Analysis Separation

| Purpose | Method | Where | Stakes |
|---------|--------|-------|--------|
| **Scoring** | Gates + penalties + small model judge | Runtime | High (determines winner) |
| **Analysis** | Cosine similarity, clustering, patterns | BigQuery (post-hoc) | Low (explains why) |

Scoring metrics are leak-free, deterministic, and minimal. Analysis metrics are computed on all embeddings post-hoc for exploratory investigation. Analysis metrics are never used for gate decisions.

#### Output Schema Standardization

**Evidence format (standardized across all kernels):**
```json
{
  "evidence": {
    "quote": "<verbatim_quote_or_null>",
    "page": <int_or_null>
  }
}
```

**Rules (no contradictions):**

| Status | evidence.quote | evidence.page | candidates |
|--------|----------------|---------------|------------|
| `ok` | Required (string) | Required (int) | Optional |
| `missing` | `null` | `null` | Empty `[]` |
| `ambiguous` | `null` | `null` | Required (≥2) |

**Type constraints:**
- `evidence.page`: Always `int` or `null`. Never string.
- `evidence.quote`: Always `string` or `null`. Never omitted.
- `candidates[].page`: Always `int` or `null`.

**Object always present:** The `evidence` key is always an object, never `null` itself. For missing/ambiguous fields, the object contains null values.

```json
// CORRECT for status="missing"
"evidence": {"quote": null, "page": null}

// WRONG - do not use
"evidence": null
```

**Parser behavior:** The scorer normalizes `"evidence": null` to `{"quote": null, "page": null}` for robustness, but kernels should output the object form.

**Kernel instruction (verbatim in both kernels):**
```
Output format:
- evidence.quote: verbatim substring from document, or null if missing/ambiguous
- evidence.page: integer page number (1-indexed), or null if missing/ambiguous
- For status="ok": quote and page are REQUIRED
- For status="missing" or "ambiguous": quote=null, page=null
```

#### Output Completeness Requirements

| Condition | Handling |
|-----------|----------|
| Field omitted entirely | Automatic miss (value=0, evidence=0, status=0) |
| Field appears twice | Schema failure (score=0 for that field) |
| Extra fields not in schema | Ignored (no penalty, no credit) |
| Output truncated | Schema compliance penalty |

#### Primary Metrics

**Value Accuracy:**
```python
def value_accuracy(extracted: Extraction, ground_truth: GroundTruth) -> float:
    # BRANCH 1: Ambiguous field (GT says multiple valid answers)
    if ground_truth.is_ambiguous:
        if extracted.status != "ambiguous" or not extracted.candidates:
            return 0.0  # Should have flagged ambiguous with candidates
        
        # Score = coverage of GT candidates by predicted candidates
        gt_values = set(normalize(v) for v in ground_truth.candidate_values)
        pred_values = set(normalize(c.value) for c in extracted.candidates)
        
        if not gt_values:
            return 1.0 if not pred_values else 0.5
        
        # F1-style: reward recall, penalize excess
        recall = len(gt_values & pred_values) / len(gt_values)
        precision = len(gt_values & pred_values) / len(pred_values) if pred_values else 0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # BRANCH 2: Missing field (GT says not in document)
    if not ground_truth.exists_in_document:
        if extracted.value is None and extracted.status == "missing":
            return 1.0
        else:
            return 0.0  # Hallucination
    
    # BRANCH 3: Present field (GT says should extract single value)
    if extracted.value is None:
        return 0.0  # Miss
    if matches(extracted.value, ground_truth.acceptable_values):
        return 1.0
    if fuzzy_match(extracted.value, ground_truth.correct_value) > 0.9:
        return 0.5  # Type-aware partial credit
    return 0.0
```

**Evidence Validity (Layered Approach):**
```python
def evidence_validity(extracted: Extraction, 
                      ground_truth: GroundTruth,
                      document_text: str,
                      field_name: str) -> float:
    
    # BRANCH 1: Ambiguous field - score candidate quotes
    if ground_truth.is_ambiguous:
        if not extracted.candidates:
            return 0.0
        return candidate_evidence_score(extracted.candidates, document_text, field_name)
    
    # BRANCH 2: Missing field - no evidence expected
    if not ground_truth.exists_in_document:
        if extracted.evidence.quote is None:
            return 1.0
        else:
            return 0.0  # Fabricated evidence for non-existent field
    
    # BRANCH 3: Present field - evidence required
    if extracted.value is None:
        return 0.0  # Can't get evidence credit for missing a field
    
    quote = extracted.evidence.quote
    value = str(extracted.value)
    
    if quote is None:
        return 0.0  # Value without evidence
    
    # Gate 1: Quote exists in document (minimal normalization)
    normalized_doc = normalize_minimal(document_text)  # Unicode + whitespace only
    normalized_quote = normalize_minimal(quote)
    
    if normalized_quote not in normalized_doc:
        return 0.0  # Fabricated quote
    
    # Gate 2: Quote contains the value
    if normalize_minimal(value) not in normalized_quote:
        return 0.0  # Quote doesn't support this value
    
    # Layer 1: Token efficiency (penalize padded quotes)
    token_ratio = len(quote.split()) / max(len(value.split()), 1)
    token_penalty = max(0.0, min(1.0, 1.0 - (token_ratio - 5) * 0.1))
    
    # Layer 2: Small model judge (can quote support this extraction?)
    judge_score = small_model_judge(quote, field_name, value)
    
    # Combine
    return 0.3 * token_penalty + 0.7 * judge_score


def normalize_minimal(text: str) -> str:
    """Minimal normalization: Unicode NFKC + whitespace collapse. NO case folding."""
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    text = ' '.join(text.split())  # Collapse whitespace
    return text


def small_model_judge(quote: str, field_name: str, claimed_value: str) -> float:
    """Ask small model to independently extract. Compare to claimed value.
    
    Leak-free: Only uses field_name (which both kernels have).
    Does not use schema descriptions.
    """
    response = flash_lite.complete(
        f"Extract the {field_name} from this text. "
        f"Return only the value, nothing else.\n\n"
        f"Text: {quote}",
        temperature=0
    )
    
    extracted = normalize_minimal(response.strip())
    
    if extracted == normalize_minimal(claimed_value):
        return 1.0  # Quote supports this extraction
    else:
        return 0.0  # Quote doesn't support this extraction
```

**Candidate Evidence Score:**
```python
def candidate_evidence_score(candidates: List[Candidate], 
                             document_text: str,
                             field_name: str) -> float:
    """Score evidence quality across all candidates."""
    if not candidates:
        return 0.0
    
    scores = []
    for candidate in candidates:
        if not candidate.quote:
            scores.append(0.0)
            continue
        
        # Gate: Quote exists
        if normalize_minimal(candidate.quote) not in normalize_minimal(document_text):
            scores.append(0.0)  # Fabricated
            continue
        
        # Gate: Quote contains value
        if normalize_minimal(str(candidate.value)) not in normalize_minimal(candidate.quote):
            scores.append(0.0)  # Misaligned
            continue
        
        # Small model judge
        judge_score = small_model_judge(candidate.quote, field_name, str(candidate.value))
        scores.append(judge_score)
    
    return sum(scores) / len(scores) if scores else 0.0
```

**Page Number Scoring:**
```python
def page_accuracy(extracted: Extraction, 
                  ground_truth: GroundTruth,
                  document_pages: List[str]) -> float:
    """Verify evidence.page matches the page containing the quote."""
    
    if ground_truth.is_ambiguous:
        # Score candidate pages
        if not extracted.candidates:
            return 0.0
        correct = 0
        for candidate in extracted.candidates:
            if candidate.quote and candidate.page:
                actual_page = find_page_containing(candidate.quote, document_pages)
                if actual_page == candidate.page:
                    correct += 1
        return correct / len(extracted.candidates)
    
    if not ground_truth.exists_in_document:
        return 1.0 if extracted.evidence.page is None else 0.0
    
    if extracted.evidence.quote is None or extracted.evidence.page is None:
        return 0.0
    
    actual_page = find_page_containing(extracted.evidence.quote, document_pages)
    return 1.0 if actual_page == extracted.evidence.page else 0.0
```

**Hallucination Rate:**
```python
def hallucination_rate(results: List[Tuple[Extraction, GroundTruth]]) -> float:
    """% of extractions where value≠null but field doesn't exist."""
    hallucinations = sum(
        1 for ext, gt in results 
        if not gt.exists_in_document and ext.value is not None
    )
    applicable = sum(1 for _, gt in results if not gt.exists_in_document)
    return hallucinations / applicable if applicable > 0 else 0.0
```

**Fabrication Rate (Fixed Denominator):**
```python
def fabrication_rate(results: List[Tuple[Extraction, GroundTruth, str]]) -> float:
    """% of required quotes that are fabricated.
    
    Denominator = all quotes that should exist:
    - All fields with status="ok"
    - All candidates for fields with status="ambiguous"
    """
    required_quotes = []
    fabricated = 0
    
    for ext, gt, doc_text in results:
        if ext.status == "ok" and ext.evidence.quote:
            required_quotes.append(ext.evidence.quote)
            if normalize_minimal(ext.evidence.quote) not in normalize_minimal(doc_text):
                fabricated += 1
        
        if ext.status == "ambiguous" and ext.candidates:
            for candidate in ext.candidates:
                if candidate.quote:
                    required_quotes.append(candidate.quote)
                    if normalize_minimal(candidate.quote) not in normalize_minimal(doc_text):
                        fabricated += 1
    
    return fabricated / len(required_quotes) if required_quotes else 0.0
```

**Quote Coverage:**
```python
def quote_coverage(results: List[Tuple[Extraction, GroundTruth]]) -> dict:
    """Report quote presence rates."""
    ok_with_quote = sum(1 for ext, gt in results 
                        if ext.status == "ok" and ext.evidence.quote)
    ok_total = sum(1 for ext, gt in results if ext.status == "ok")
    
    ambig_with_candidates = sum(1 for ext, gt in results
                                 if ext.status == "ambiguous" and len(ext.candidates) >= 2)
    ambig_total = sum(1 for ext, gt in results if ext.status == "ambiguous")
    
    return {
        "ok_coverage": ok_with_quote / ok_total if ok_total else 0,
        "ambiguous_coverage": ambig_with_candidates / ambig_total if ambig_total else 0
    }
```

#### Composite Score

**CRITICAL: Evidence/page/status are conditional on value correctness.**

A wrong value with a real quote should not score 0.70. It should score near 0.

```python
def composite_score(extracted: Extraction, 
                    ground_truth: GroundTruth,
                    document_text: str,
                    document_pages: List[str],
                    field_name: str) -> CompositeResult:
    """
    Compute weighted composite with correctness gating.
    
    Evidence, page, and status scores are ZEROED if value is wrong.
    This prevents "cite anything plausible" from scoring high.
    """
    weights = {
        'value': 0.30,
        'evidence': 0.30,
        'page': 0.10,
        'status': 0.15,
        'schema': 0.15
    }
    
    # Always compute value accuracy first
    val = value_accuracy(extracted, ground_truth)
    schema = 1.0 if is_valid_schema(extracted) else 0.0
    
    # GATING LOGIC: wrong value zeroes dependent scores
    if ground_truth.exists_in_document and not ground_truth.is_ambiguous:
        # Unambiguous present field
        if val == 0.0:
            # Wrong value → no credit for evidence, page, status
            evidence = 0.0
            page = 0.0
            status = 0.0
        else:
            # Correct value → compute normally
            evidence = evidence_validity(extracted, ground_truth, document_text, field_name)
            page = page_accuracy(extracted, ground_truth, document_pages)
            status = 1.0 if extracted.status == "ok" else 0.0
    
    elif ground_truth.is_ambiguous:
        # Ambiguous field: scale evidence/page by candidate correctness
        cand_f1 = val  # value_accuracy already returns F1 for ambiguous
        
        if cand_f1 == 0.0:
            evidence = 0.0
            page = 0.0
            status = 0.0
        else:
            raw_evidence = candidate_evidence_score(extracted.candidates, document_text, field_name)
            raw_page = page_accuracy(extracted, ground_truth, document_pages)
            
            # Scale by correctness to prevent "dump random quotes" exploit
            evidence = cand_f1 * raw_evidence
            page = cand_f1 * raw_page
            status = 1.0 if extracted.status == "ambiguous" else 0.0
    
    else:
        # Missing field
        if val == 1.0:
            # Correctly identified as missing
            evidence = evidence_validity(extracted, ground_truth, document_text, field_name)
            page = page_accuracy(extracted, ground_truth, document_pages)
            status = 1.0 if extracted.status == "missing" else 0.0
        else:
            # Hallucinated value for missing field
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
```

**Why this matters:**

| Scenario | Before Fix | After Fix |
|----------|------------|-----------|
| Wrong value + real quote + correct page | 0.70 | 0.15 (schema only) |
| Correct value + fabricated quote | 0.55 | 0.45 (value + status + schema) |
| Ambiguous GT + random real candidates | ~0.70 | F1 * raw (near 0 if candidates wrong) |
| Correct extraction | 1.0 | 1.0 |

The gating ensures we measure extraction correctness, not "ability to quote anything plausible."
```

#### Null Baseline Implementation

```python
class NullBaseline:
    """Baseline that always outputs missing. Implemented for comparison."""
    
    def extract(self, document: str, schema: Schema) -> dict:
        return {
            "extractions": [
                {
                    "field": field.name,
                    "value": None,
                    "evidence": {"quote": None, "page": None},
                    "status": "missing",
                    "confidence": "high",
                    "candidates": []
                }
                for field in schema.fields
            ]
        }
    
    def expected_score(self, ground_truth_distribution: dict) -> float:
        """Calculate expected composite score given GT distribution.
        
        For missing field (exists=False): value=1, evidence=1, page=1, status=1, schema=1 → 1.0
        For present field (exists=True): value=0, evidence=0, page=0, status=0, schema=1 → 0.15
        For ambiguous field: value=0, evidence=0, page=0, status=0, schema=1 → 0.15
        """
        pct_missing = ground_truth_distribution['missing']  # e.g., 0.25
        pct_present = ground_truth_distribution['present']  # e.g., 0.60
        pct_ambiguous = ground_truth_distribution['ambiguous']  # e.g., 0.15
        
        return (pct_missing * 1.0 + 
                pct_present * 0.15 + 
                pct_ambiguous * 0.15)
```

**Null baseline is computed and reported in Phase 1. CTN must beat it significantly.**

#### Analysis Metrics (Post-Hoc, BigQuery)

The following are computed for all extractions and stored for analysis, but NOT used in scoring:

```python
# Computed once, stored in BigQuery
analysis_record = {
    "doc_id": str,
    "field_name": str,
    "kernel": str,
    "quote": str,
    "value": str,
    "quote_embedding": List[float],  # For cosine similarity analysis
    "value_embedding": List[float],
    "field_description_embedding": List[float],  # Note: NOT used in scoring (leak)
    "quote_length": int,
    "value_length": int,
    "token_ratio": float,
    "judge_score": float,
    "cosine_quote_to_value": float,  # Computed post-hoc
    "cosine_quote_to_field_desc": float,  # Computed post-hoc (analysis only)
    "extra_fields": List[str],  # Track any extra fields output
}
```

Use cases:
- Failure mode clustering
- "Why did CTN win on these cases?"
- Cosine similarity distributions
- Quote tightness analysis
- Extra field rate tracking

#### Known Limitations (Tier-2 Risks)

**1. Small-model judge can introduce false negatives**

If quote is minimal (value only, no label), judge may fail even for valid quotes. This creates tension with token_penalty.

*Mitigation:* 
- Calibrate judge on DEV set first
- Run judge on GT quotes; if accuracy < 95%, downweight or disable
- Consider: `evidence = 0.8 * deterministic + 0.2 * judge`

**2. "Evidence set" definition is implicit**

What constitutes a single piece of evidence is defined by model behavior, not explicitly in the kernel. We're evaluating emergent strategies.

*Mitigation:*
- Document as Phase-1 limitation
- Phase-3 can add explicit evidence boundary rules if needed

**3. Extra fields are ignored (not penalized)**

A kernel could emit junk fields that don't affect scoring but make downstream use worse.

*Mitigation:*
- Track `extra_field_rate` in analysis metrics
- Report in Phase-1 results
- Add penalty in Phase-2 if it's a problem

### 3.6 Statistical Analysis

**Unit of Analysis:** Document (not field)

Each document has multiple fields. Fields within a document are correlated. We aggregate to document-level before statistical tests.

```python
def document_level_score(doc_id: str, field_scores: Dict[str, float]) -> float:
    """Average score across fields within a document."""
    return mean(field_scores.values())
```

**DEV/TEST Discipline:**

| Set | Statistical Use | Allowed Operations |
|-----|-----------------|-------------------|
| **DEV** | Exploratory only | Iterate kernels, tune parameters, debug, look at outputs, run any analysis |
| **TEST** | Confirmatory | Single pre-registered test, no peeking before final run |

**DEV Set Analysis (Investigation Cycles):**
- No p-values reported from DEV
- Use for: "Is this kernel variant better?" → iterate
- Use for: "Why are these cases failing?" → debug
- Use for: "What does the score distribution look like?" → calibrate

**TEST Set Analysis (Gate Decision):**
- Single run after investigation complete
- Pre-registered test: paired t-test on document-level composite scores
- Significance threshold: p < 0.05 (single comparison in Phase 1)
- Effect size computed: Cohen's d

```python
def compare_kernels(scores_ctn: List[float], 
                    scores_idiomatic: List[float]) -> ComparisonResult:
    """
    Paired t-test at document level on TEST set only.
    
    Both lists have same length (one score per document).
    Documents are matched (same doc tested with both kernels).
    """
    assert len(scores_ctn) == 100, "Must be TEST set"
    
    t_stat, p_value = ttest_rel(scores_ctn, scores_idiomatic)
    effect_size = cohens_d_paired(scores_ctn, scores_idiomatic)
    mean_diff = mean(scores_ctn) - mean(scores_idiomatic)
    
    return ComparisonResult(
        mean_diff=mean_diff,
        mean_diff_absolute=mean_diff,  # This is our ">5%" threshold basis
        t_stat=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        ci_95=confidence_interval(scores_ctn, scores_idiomatic)
    )
```

**Gate Thresholds (on TEST set, absolute scale 0-1):**

| Threshold | Value | Meaning |
|-----------|-------|---------|
| Minimum improvement | +0.05 absolute | "5% better" on composite |
| Strong improvement | +0.10 absolute | "10% better" on composite |
| Significance | p < 0.05 | Not due to chance |
| Effect size (small) | d > 0.2 | Practically meaningful |
| Effect size (medium) | d > 0.5 | Clearly meaningful |

**Effect Size Interpretation:**

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

**What We Report:**
- DEV set: descriptive statistics only (means, distributions, failure analysis)
- TEST set: formal comparison with p-value, CI, effect size
- Null baseline: computed score on TEST set for reference

### 3.7 Test Data: Synthetic Documents

**DEV/TEST Split (Critical for Valid Gates)**

| Set | Documents | Purpose | Rules |
|-----|-----------|---------|-------|
| **DEV** | 100 | Investigation, iteration, debugging | Iterate freely, no p-values |
| **TEST** | 100 | Gate decision | Locked. Single run. Pre-registered test. |

Both sets are generated identically and stratified by difficulty. They are separated before any experimentation begins. Results on TEST determine the Phase 1 gate. Results on DEV inform iteration but have no bearing on decisions.

**Document Types:**

| Type | DEV | TEST | Fields | Purpose |
|------|-----|------|--------|---------|
| Invoice | 25 | 25 | 5 | Numbers, dates, tables |
| Contract | 20 | 20 | 6 | Long text, legal |
| Medical | 20 | 20 | 7 | Abbreviations, sensitive |
| Receipt | 20 | 20 | 6 | Compact, OCR-like |
| Resume | 15 | 15 | 5 | Semi-structured |
| **Total** | **100** | **100** | | |

**Difficulty Distribution (per set):**

| Difficulty | % | DEV | TEST | Characteristics |
|------------|---|-----|------|-----------------|
| Easy | 40% | 40 | 40 | All fields present, clear |
| Medium | 30% | 30 | 30 | Some missing, formatting variation |
| Hard | 20% | 20 | 20 | Ambiguous, decoys, multi-page |
| Adversarial | 10% | 10 | 10 | Contradictions, edge cases |

**Required Case Distribution (per set):**

| Case Type | DEV | TEST | Purpose |
|-----------|-----|------|---------|
| Field exists, unambiguous | 60 | 60 | Standard extraction |
| Field missing | 25 | 25 | Test null handling |
| Field ambiguous | 15 | 15 | Test ambiguity detection |

**Ambiguous Case Requirements:**

For all ambiguous cases, ground truth includes:
- `is_ambiguous: true`
- `candidate_values: [list of valid interpretations]`
- Explicit note on why ambiguous (e.g., "two dates appear, unclear which is effective date")

No ambiguous case relies on implicit precedence rules. If contradictory values exist, GT marks it ambiguous—we are not testing semantic reasoning, we are testing extraction.

**Stratified Verification:**
- Minimum 2 documents per (type × difficulty) cell per set
- = 5 types × 4 difficulties × 2 docs × 2 sets = 80 documents manually verified
- Verification done by vendor (see Buy vs Build section)
- Verification checklist:
  - [ ] Ground truth value is correct
  - [ ] Evidence quote exists verbatim in document text
  - [ ] Missing fields are actually missing
  - [ ] Ambiguous fields have valid alternative interpretations
  - [ ] Page numbers are correct

### 3.8 Generation Pipeline

```python
def generate_synthetic_document(doc_type: str, difficulty: str) -> Document:
    """
    Generate one synthetic document with ground truth.
    
    Uses different LLM (or structured generation) from evaluation model
    to avoid style leakage.
    """
    # 1. Generate structured content
    schema = load_schema(doc_type)
    content = generate_content(schema, difficulty)  # Returns dict
    
    # 2. Inject difficulty
    if difficulty in ["hard", "adversarial"]:
        content = inject_difficulty(content, difficulty)
    
    # 3. Render to text (with page markers)
    template = random.choice(get_templates(doc_type))
    text = render_template(template, content)
    
    # 4. Build ground truth from content
    ground_truth = build_ground_truth(content, text, schema)
    
    # 5. Verify evidence quotes exist
    for field_gt in ground_truth:
        if field_gt.exists_in_document:
            assert field_gt.evidence_quote in text, f"Quote not found: {field_gt}"
    
    return Document(text=text, ground_truth=ground_truth, doc_type=doc_type)
```

**Difficulty Injection:**
```python
def inject_difficulty(content: dict, difficulty: str) -> dict:
    if difficulty == "hard":
        # Add decoy values (similar but wrong)
        # Add ambiguous dates/amounts
        # Split across pages
        pass
    
    if difficulty == "adversarial":
        # Remove some field labels (value without label)
        # Add contradictory values
        # Missing required fields
        pass
    
    return content
```

### 3.9 Execution

**Configuration (phase1.yaml):**
```yaml
name: "Phase 1: Practical"
version: "1.0"

documents:
  source: synthetic
  dev_test_split: true  # Generate DEV and TEST sets
  synthetic:
    per_set: 100  # 100 DEV + 100 TEST
    types:
      - name: invoice
        count_per_set: 25
        difficulties: [easy, medium, hard, adversarial]
      - name: contract
        count_per_set: 20
        difficulties: [easy, medium, hard, adversarial]
      - name: medical
        count_per_set: 20
        difficulties: [easy, medium, hard, adversarial]
      - name: receipt
        count_per_set: 20
        difficulties: [easy, medium, hard, adversarial]
      - name: resume
        count_per_set: 15
        difficulties: [easy, medium, hard, adversarial]

kernels:
  - name: ctn
    path: kernels/ctn_extraction.txt
    enabled: true
  - name: idiomatic
    path: kernels/idiomatic.txt
    enabled: true
  - name: null_baseline
    path: baselines/null_baseline.py
    enabled: true
  - name: literal
    path: kernels/literal.txt
    enabled: false  # Added in Phase 2

comparisons:
  - a: ctn
    b: idiomatic
    primary: true
  - a: ctn
    b: null_baseline
    primary: false  # Reference only

metrics:
  primary: [value_accuracy, evidence_validity, hallucination_rate, fabrication_rate]
  secondary: [page_accuracy, status_accuracy, schema_compliance, quote_coverage]
  composite_weights:
    value: 0.30
    evidence: 0.30
    page: 0.10
    status: 0.15
    schema: 0.15

statistics:
  method: paired_ttest
  alpha: 0.05
  correction: none  # Single comparison
  cluster_by: document
  test_set_only: true  # Formal stats on TEST only

model:
  name: claude-sonnet-4-20250514
  temperature: 0
  max_tokens: 2048

judge_model:
  name: gemini-flash-lite  # For evidence scoring
  temperature: 0

execution:
  delay: 0.3
  retries: 3
  timeout: 60

output:
  results_dir: results/phase1
  save_raw: true
  generate_report: true
  bigquery_export: true  # For analysis metrics
```

**Run Commands:**
```bash
# Investigation (DEV set)
python run_evaluation.py --config configs/phase1.yaml --set dev

# Final gate decision (TEST set) - run once only
python run_evaluation.py --config configs/phase1.yaml --set test
```

**Time Estimate (Buy Path):**
- Build infrastructure: Day 1-2 (16 hrs)
- Document generation (200 total): Day 3 morning (2 hrs machine)
- Verification (parallel with build): Vendor delivers Day 3
- DEV iteration: Day 3-4 (as needed)
- TEST evaluation run: Day 4 (~1.5 hrs)
- Analysis: Day 4 (4 hrs)
- **Total: 4 days, 22 hours your time, $850**

### 3.10 Decision Gate

**Philosophy:** Try like a founder, be smart like a founder. We persist through difficulty but recognize when to change direction.

#### Phase 1 Budget

| Resource | Budget | Hard Limit |
|----------|--------|------------|
| Time | 4 days initial + 6 days investigation | 10 days total |
| Cost | ~$20 initial + ~$40 investigation | $100 |
| Revision cycles | 3 | 3 |

#### Outcome Tree (TEST Set Results)

```
Phase 1 TEST Set Run Complete
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESULT CLASSIFICATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  A. STRONG SIGNAL                                               │
│     CTN > Idiomatic by ≥0.10 absolute, p<0.05, d>0.5           │
│     → Proceed to Phase 2                                        │
│                                                                 │
│  B. MODERATE SIGNAL                                             │
│     CTN > Idiomatic by 0.05-0.10 absolute, p<0.05              │
│     → Proceed to Phase 2                                        │
│                                                                 │
│  C. WEAK SIGNAL                                                 │
│     CTN > Idiomatic by 0.03-0.05, p<0.10                       │
│     → Review DEV analysis, possibly proceed with caveats        │
│                                                                 │
│  D. NO SIGNAL                                                   │
│     Difference <0.03 or wrong direction                         │
│     → Review DEV findings, consider pivot                       │
│                                                                 │
│  E. INFRASTRUCTURE FAILURE                                      │
│     Both kernels <50%, parsing failures >20%                    │
│     → Fix infrastructure, re-run DEV (TEST still locked)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** Outcomes C and D do NOT trigger more iteration on TEST. The DEV set analysis should have already revealed these issues. If TEST shows weak/no signal after DEV looked promising, this is a real finding (DEV ≠ TEST generalization issue), not a cue to iterate more.

#### Investigation Branch (DEV Set — Before TEST Run)

**Investigation happens BEFORE the TEST run, not after.**

If DEV results show no signal, investigate on DEV:

```
DEV Set Shows No/Weak Signal
      │
      ▼
DIAGNOSTIC CHECKLIST (DEV analysis)
      │
      ├─── Q1: Ceiling Effect?
      │    Both kernels >85% accuracy on DEV
      │    └── YES → Action: Increase difficulty
      │              - Add more adversarial cases to DEV
      │              - Regenerate harder DEV set
      │              → Re-run DEV (Revision Cycle 1)
      │
      ├─── Q2: Floor Effect?
      │    Both kernels <60% accuracy on DEV
      │    └── YES → Action: Decrease difficulty OR fix task
      │              - Simplify schema
      │              - Clearer field definitions
      │              - Better ground truth
      │              → Re-run DEV (Revision Cycle 1)
      │
      ├─── Q3: Kernel Instability?
      │    CTN variance >> Idiomatic variance on DEV
      │    └── YES → Action: Revise CTN kernel
      │              - Tighten constraints
      │              - Clearer instructions
      │              - Test kernel variants on DEV
      │              → Re-run DEV (Revision Cycle 1)
      │
      ├─── Q4: Metric Mismatch?
      │    Signal in some metrics but not composite on DEV
      │    └── YES → Action: Investigate metric
      │              - Which metric shows signal?
      │              - Is composite weighting wrong?
      │              - Should we focus on that metric?
      │              → Re-analyze DEV, possibly adjust weights
      │
      ├─── Q5: Document Type Effect?
      │    Signal in subset of document types on DEV
      │    └── YES → Action: Narrow scope
      │              - CTN helps for [X] not [Y]
      │              - Focus Phase 2 on [X]
      │              → Proceed to TEST with hypothesis
      │
      └─── Q6: All Diagnostics Negative?
           No explanation found in DEV
           └── → Revision Cycle (modify and re-run DEV)
           
After 3 revision cycles on DEV with no signal:
      │
      ▼
Decision: Run TEST anyway (maybe DEV is harder than TEST)
          OR Pivot (DEV is representative, CTN doesn't help)
```

**Key principle:** TEST is never used for iteration. If you want to try something, try it on DEV. TEST is the final exam.

#### Revision Cycles

| Cycle | Focus | Time Budget | What Changes |
|-------|-------|-------------|--------------|
| **1** | Difficulty calibration | 2 days | Test cases, ground truth |
| **2** | Kernel refinement | 2 days | CTN kernel structure, wording |
| **3** | Task/metric pivot | 2 days | Different metrics, different emphasis |

**Revision Cycle Protocol (DEV SET ONLY):**
```
1. Run both kernels on DEV set
2. Analyze: What's failing? Why?
3. Form hypothesis: "We saw X because Y, so we'll change Z"
4. Make ONE major change (not multiple)
5. Re-run on DEV set
6. Compare to previous DEV run
7. Update hypothesis
8. Repeat until DEV results look promising

ONLY THEN:
9. Run on TEST set (single run, pre-registered)
10. Report TEST results as Phase 1 outcome
```

**Why DEV/TEST Matters:**
- Without split: 3 revision cycles = fitting to noise, TEST is contaminated
- With split: 3 revision cycles = genuine learning, TEST remains valid
- TEST result is the answer to "does it work?"
- DEV result is the answer to "can we make it work better?"

#### Exit Criteria (When to Pivot)

**We pivot away from extraction (not give up on CTN) when:**

| Condition | Evidence | Pivot To |
|-----------|----------|----------|
| **3 revision cycles exhausted** | 10 days invested, no signal | Tool calling evaluation |
| **Fundamental task mismatch** | CTN makes extraction worse consistently | Different task (constraint satisfaction?) |
| **Feasibility block** | Cannot generate valid test data | Different domain or task |

**We do NOT exit when:**
- First run shows no signal (investigate first)
- Signal is weak but present (amplify it)
- One metric fails but others succeed (focus on successes)
- Specific document types fail (narrow scope)

#### Pivot Decision Framework

```
After 3 Revision Cycles with No Signal
                │
                ▼
        PIVOT DECISION MEETING
                │
        Review:
        - What did we learn in each cycle?
        - Is there any signal anywhere?
        - What's our hypothesis for why it's not working?
                │
                ▼
        ┌───────────────────────────────────────┐
        │           PIVOT OPTIONS               │
        ├───────────────────────────────────────┤
        │                                       │
        │  A. PIVOT TASK                        │
        │     CTN might help tool calling       │
        │     instead of extraction             │
        │     → Start Tool Calling Phase 1      │
        │                                       │
        │  B. PIVOT DOMAIN                      │
        │     Extraction might work on          │
        │     different document types          │
        │     → New domain, restart Phase 1     │
        │                                       │
        │  C. PIVOT APPROACH                    │
        │     CTN kernel design is wrong        │
        │     → Fundamental kernel redesign     │
        │     → Restart Phase 1                 │
        │                                       │
        │  D. PARK AND REVISIT                  │
        │     Not working now, might later      │
        │     → Document learnings              │
        │     → Park for 1 month                │
        │     → Revisit with fresh perspective  │
        │                                       │
        └───────────────────────────────────────┘
```

#### Validation Criteria (Must Pass to Exit Phase 1)

All criteria evaluated on **TEST set only** (DEV set is for learning, not deciding):

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| CTN beats null-baseline | >0.15 absolute above null baseline | Proves CTN does something |
| CTN beats idiomatic | ≥0.05 absolute, p<0.05 | Primary gate |
| Fabrication rate | <10% both kernels | Infrastructure works |
| Schema compliance | >90% both kernels | Outputs are valid |
| Ground truth verified | 80 docs manually checked (40 per set) | Data is trustworthy |
| At least one metric signal | ≥0.05 absolute improvement | Something is working |

#### Summary: Phase 1 Success Definition

**Minimum Success (proceed to Phase 2):**
- TEST set: CTN composite > Idiomatic by ≥0.05 absolute
- p < 0.05
- At least one primary metric shows ≥0.05 improvement
- No major infrastructure failures

**Ideal Success:**
- TEST set: CTN composite > Idiomatic by ≥0.10 absolute
- p < 0.05, d > 0.5
- Multiple metrics show improvement
- Effect consistent across document types

**Acceptable Narrowed Success:**
- Signal present in subset of document types
- → Proceed to Phase 2 with narrowed scope
- Document which types and hypothesize why

### 3.11 Scorer Validation (Pre-Flight Check)

Before paying vendors or running on TEST, validate the scorer against 10 adversarial tabletop cases. All must behave exactly as intended.

**Create synthetic doc + GT, craft model outputs by hand:**

| # | Scenario | Expected Behavior |
|---|----------|-------------------|
| 1 | Wrong value + real quote + correct page + status ok | Composite ≤ 0.15 (schema only) |
| 2 | Correct value + fabricated quote | evidence=0, page=0, composite ≈ 0.45 |
| 3 | Ambiguous GT; model outputs ambiguous + 10 random real candidates | Low score (value F1 low → evidence/page scaled down) |
| 4 | Ambiguous GT; model outputs ok with one choice | Penalized (status wrong, value may be partial) |
| 5 | Missing GT; model outputs non-null with real quote | Hallucination → composite near 0 |
| 6 | Missing GT; model outputs missing but includes non-null quote | Format violation → schema penalty |
| 7 | Duplicate field entries in output | Schema failure for that field |
| 8 | Field omitted entirely from output | Automatic miss (all scores 0 for that field) |
| 9 | Correct value but quote doesn't contain value substring | evidence=0 (gate 2 fails) |
| 10 | Correct value, correct quote, page number off-by-one | page=0, rest scored normally |

**Implementation:**

```python
def test_scorer_adversarial():
    """Run before any real evaluation."""
    
    cases = [
        # Case 1: Wrong value + real quote
        {
            "gt": {"exists": True, "ambiguous": False, "value": "1234.56"},
            "ext": {"value": "9999.99", "quote": "Total: 9999.99", "page": 1, "status": "ok"},
            "doc": "Invoice\nTotal: 9999.99\nActual: 1234.56",
            "expect": {"composite_max": 0.20}  # Should be ~0.15
        },
        # Case 2: Correct value + fabricated quote
        {
            "gt": {"exists": True, "ambiguous": False, "value": "1234.56"},
            "ext": {"value": "1234.56", "quote": "This quote is not in doc", "page": 1, "status": "ok"},
            "doc": "Invoice\nTotal: 1234.56",
            "expect": {"evidence": 0.0, "page": 0.0}
        },
        # ... cases 3-10
    ]
    
    for i, case in enumerate(cases, 1):
        result = composite_score(...)
        for key, expected in case["expect"].items():
            if key.endswith("_max"):
                assert result[key.replace("_max", "")] <= expected, f"Case {i} failed"
            else:
                assert result[key] == expected, f"Case {i} failed"
    
    print("All 10 adversarial cases passed. Scorer is valid.")
```

**Gate:** Do not proceed to DEV evaluation until all 10 cases pass.

### 3.12 Phase 1 Deliverables

1. **Infrastructure:** Complete `ctn_extraction/` package (reusable for all phases)

2. **Data:** 
   - 200 synthetic documents (100 DEV + 100 TEST) with ground truth
   - 80 vendor-verified documents (40 per set)
   - Verification report from annotation vendor
   - BigQuery export of all analysis metrics

3. **Report:** "CTN Extraction Kernel: Practical Evaluation"
   - DEV set: exploratory analysis, failure modes, iteration history
   - TEST set: formal comparison results (primary outcome)
   - Per-metric breakdown
   - Per-document-type breakdown
   - Sample extractions (good and bad)
   - Null baseline comparison

4. **Recommendation:** Use CTN / Don't use CTN / Needs more investigation

---

## 4. Phase 2: Applied Research

*Only executed if Phase 1 gate passes.*

### 4.1 Objective

Answer: **"Does the CTN effect replicate on real data and generalize?"**

### 4.2 Additions to Phase 1

| Addition | Purpose |
|----------|---------|
| Literal NL kernel | Enables CTN vs Literal comparison (isolates notation from content) |
| Real documents | Tests on non-synthetic data |
| Multiple domains | Tests generalization |
| Blind validation | Independent replication |

### 4.3 Test Arms

| Arm | Description | New? |
|-----|-------------|------|
| CTN | Full CTN kernel | No |
| Idiomatic | Best-effort concise English | No |
| **Literal** | 1-1 prose translation of CTN | **Yes** |

**Literal NL Kernel:**
```
You are a structured extraction system.

System Initialization:
- Authority: System specification
- Domain: Structured extraction from documents
- Constraint: Extract with evidence

Cognitive Guidelines (Processing Traits):

Trait 1 - Atomic Clarity: Parse each field independently. Do not blend 
fields or meanings. Hidden meanings should approach zero.

Trait 2 - Extraction Accuracy: Extract exactly what appears in the 
document. Minimize errors. Do not paraphrase.

Trait 3 - Context Isolation: Each field extraction is independent. Do 
not let one field influence another.

Trait 4 - Schema Over Fluency: Output must match required schema exactly. 
No prose or explanation.

Trait 5 - Reality Anchor: If value not in document: value=null, 
status=missing. If ambiguous: value=null, status=ambiguous, candidates 
required. NEVER invent values. NEVER fabricate quotes.

Trait 6 - Minimal Extraction: Extract only requested fields.

Trait 7 - Output Schema Lock: Valid JSON only.

Strategic Solver:
For each field f in schema:
    Search document for evidence of field f
    
    If evidence set is empty:
        value = null
        status = "missing"
    
    If evidence set has exactly one item:
        value = extracted value
        evidence.quote = verbatim quote
        status = "ok"
    
    If evidence set has multiple items:
        value = null
        status = "ambiguous"
        candidates = all options with their quotes

Constraints:
- Every non-null value MUST have verbatim quote from document
- Every candidate MUST have verbatim quote from document
- Quotes must exist exactly in document text

{{SCHEMA}}

{{OUTPUT_FORMAT}}
```

### 4.4 Comparisons

| Comparison | Question | Primary? |
|------------|----------|----------|
| CTN vs Idiomatic | Does CTN package beat good English? | Yes |
| CTN vs Literal | Does notation matter (vs same content in prose)? | Yes |
| Literal vs Idiomatic | Does additional content matter? | Secondary |

### 4.5 Data Sources

| Source | Count | Purpose |
|--------|-------|---------|
| Synthetic (Phase 1) | 50 | Replication check |
| Public dataset | 150 | Real document variation |
| User-provided | 100 | Domain relevance |
| **Subtotal Phase 2** | **300** | |
| Blind external | 100+ | Independent validation |

**Public Dataset Adaptation:**
- Select from FUNSD, CORD, or Kleister-NDA based on available evidence annotations
- Adapt to our schema and ground truth format
- Document any annotation work required

**User-Provided Documents:**
Requirements for your collected data:
```json
{
  "document_id": "doc_001",
  "text": "---PAGE 1---\n...",
  "schema": "invoice",
  "ground_truth": [
    {
      "field": "vendor_name",
      "exists_in_document": true,
      "correct_value": "Acme Corp",
      "acceptable_values": ["Acme Corp", "Acme Corporation"],
      "evidence_must_contain": "Acme",
      "is_ambiguous": false
    }
  ]
}
```

### 4.6 Blind External Validation (Phase 3 Handoff)

**What collaborator receives:**
```
ctn_extraction/
├── README_EXTERNAL.md      # Instructions
├── run_evaluation.py
├── requirements.txt
├── core/
├── metrics/
├── statistics/
├── kernels/
│   ├── ctn_extraction.txt
│   ├── idiomatic.txt
│   └── literal.txt
└── templates/
    ├── config_template.yaml
    └── ground_truth_template.json
```

**What collaborator provides:**
```
external_data/
├── config.yaml             # Their configuration
├── documents/              # Their documents (text format)
├── schema.json             # Their schema
└── ground_truth.json       # Their annotations
```

**Blind Protocol:**
1. Collaborator commits hash of data before running
2. Runs evaluation once
3. Reports aggregate metrics only (no per-document details initially)
4. After results: may disclose domain for publication

### 4.7 Statistical Analysis

**Multiple Comparisons:**
- 3 pairwise comparisons
- Bonferroni correction: α = 0.05 / 3 = 0.017

**Method:** Document-level paired t-test with cluster bootstrap for robustness check

### 4.8 Decision Gate

#### Phase 2 Budget

| Resource | Budget | Hard Limit |
|----------|--------|------------|
| Time | 5 days initial + 5 days investigation | 10 days total |
| Cost | ~$50 initial + ~$50 investigation | $150 |
| Revision cycles | 2 | 2 |

#### Outcome Tree

```
Phase 2 Initial Run Complete
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESULT CLASSIFICATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  A. REPLICATION SUCCESS                                         │
│     Phase 1 effect holds on real data (within 3% of Phase 1)   │
│     CTN > Idiomatic, p<0.017 (Bonferroni corrected)            │
│     → Proceed to blind validation                               │
│                                                                 │
│  B. PARTIAL REPLICATION                                         │
│     Effect smaller than Phase 1 but still positive (3-5%)      │
│     p<0.05 (marginal after correction)                          │
│     → Investigate, possibly proceed with caveats                │
│                                                                 │
│  C. DOMAIN-SPECIFIC EFFECT                                      │
│     Effect holds for some document types, not others           │
│     → Document boundaries, proceed with scope limits            │
│                                                                 │
│  D. NOTATION EFFECT IDENTIFIED                                  │
│     CTN > Literal (notation helps)                              │
│     OR CTN ≈ Literal > Idiomatic (content helps, not notation) │
│     → Valuable finding either way, proceed                      │
│                                                                 │
│  E. REPLICATION FAILURE                                         │
│     Phase 1 effect disappears on real data                     │
│     → Investigation Branch                                      │
│                                                                 │
│  F. BLIND VALIDATION OUTCOMES (after main Phase 2)             │
│     F1: Blind positive → Strong evidence for generalization    │
│     F2: Blind negative → Domain-specific, document findings    │
│     F3: Blind mixed → Partial generalization, investigate      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Investigation Branch (Outcome E: Replication Failure)

```
Effect Disappeared on Real Data
            │
            ▼
    DIAGNOSTIC CHECKLIST
            │
            ├─── Q1: Data Quality Issue?
            │    Real data ground truth less reliable than synthetic
            │    └── YES → Action: Improve annotations
            │              - Double-annotate subset
            │              - Compute inter-rater agreement
            │              - Fix systematic annotation errors
            │              → Re-run (Revision Cycle 1)
            │
            ├─── Q2: Distribution Shift?
            │    Real docs fundamentally different from synthetic
            │    └── YES → Action: Analyze differences
            │              - What's different? (length, complexity, domain)
            │              - Can we generate better synthetic data?
            │              - Should we retrain expectations?
            │              → Revise synthetic generator or scope
            │
            ├─── Q3: Synthetic Overfitting?
            │    Model learned synthetic patterns, not extraction
            │    └── YES → Action: This is a real finding
            │              - Document that synthetic != real
            │              - CTN may help synthetic only
            │              - Consider implications
            │              → Publish with caveats
            │
            └─── Q4: Specific Failure Mode?
                 Effect present but masked by new failure type
                 └── YES → Action: Isolate and address
                           - What's failing on real data?
                           - Can kernel handle it?
                           → Revise kernel for real-world patterns
```

#### Blind Validation Decision Framework

```
Blind Validation Complete
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 BLIND VALIDATION OUTCOMES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POSITIVE (CTN > Idiomatic on blind domain)                    │
│  → Strong evidence for generalization                          │
│  → Phase 3 has high probability of success                     │
│  → Publish Applied Research findings                           │
│  → Proceed to Phase 3 if pursuing academic                     │
│                                                                 │
│  NEGATIVE (CTN ≤ Idiomatic on blind domain)                    │
│  → Effect is domain-specific                                   │
│  → Document which domains work                                 │
│  → Still valuable: "CTN helps for X, not Y"                    │
│  → Decision: Pursue Phase 3 for working domains?               │
│                                                                 │
│  MIXED (Some metrics positive, some negative)                  │
│  → Partial generalization                                      │
│  → Analyze: Which aspects transfer?                            │
│  → Refine claims: "CTN improves [specific thing]"              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Exit Criteria (When to Stop at Phase 2)

**We stop at Phase 2 (satisfied) when:**

| Condition | Outcome | Deliverable |
|-----------|---------|-------------|
| Practical question answered | Know when to use CTN | Tool recommendation |
| Generalization tested | Blind validation complete | Confidence in claims |
| Not pursuing publication | Applied findings sufficient | Technical report |

**We proceed to Phase 3 when:**

| Condition | Why |
|-----------|-----|
| Effect is strong and replicates | Worth deeper investigation |
| Mechanism question is interesting | Scientific contribution |
| Publication is a goal | Need academic rigor |
| Resources available | Time and interest |

**We pivot when:**

| Condition | Evidence | Pivot To |
|-----------|----------|----------|
| Replication fails after 2 cycles | 10 days invested, effect gone | Re-examine Phase 1, or park |
| Blind validation strongly negative | External domain fails | Narrow claims to tested domains |
| Feasibility block | Can't get quality real data | Synthetic-only claims (weaker) |

#### Validation Criteria (Must Pass to Claim Phase 2 Success)

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Effect replicates | Within 5% of Phase 1 | Not overfitting |
| At least one comparison significant | p<0.017 (corrected) | Real effect |
| Blind validation attempted | Run completed | Honest test |
| Annotation quality | Agreement >80% on audited subset | Trust data |

#### Summary: Phase 2 Success Definition

**Full Success:**
- Phase 1 effect replicates on real data
- CTN vs Literal comparison informative (notation vs content)
- Blind validation positive
- Clear recommendation: "Use CTN for [X]"

**Partial Success:**
- Effect replicates but weaker
- OR domain-specific effect identified
- OR notation vs content question answered (either way)
- Deliverable: Conditional recommendation with scope limits

**Valuable Negative:**
- Effect doesn't replicate, but we know why
- Synthetic ≠ Real is a finding
- Deliverable: "CTN helps synthetic benchmarks but not production"

### 4.9 Phase 2 Deliverables

1. **Data:**
   - 100 real documents with vendor-annotated ground truth
   - 20 double-annotated documents with agreement metrics
   - 150 adapted public dataset documents
   - Blind validation kit sent to external collaborator

2. **Report:** Technical report with replication results
   - Phase 1 vs Phase 2 comparison
   - Per-domain breakdown
   - Annotation quality metrics (inter-rater agreement)

3. **Boundary Analysis:** Which document types benefit

4. **Blind Validation Results:** External confirmation (when received)

5. **Recommendation:** Refined guidance on when to use CTN

---

## 5. Phase 3: Academic

*Only executed if Phase 2 demonstrates replicable effect.*

### 5.1 Objective

Answer: **"What is the mechanism? Why does CTN work?"**

### 5.2 Additions

| Addition | Purpose |
|----------|---------|
| Idiomatic-Long kernel | Length-matched control |
| Placebo-Math kernel | Notation without content |
| Baselines (null, heuristic) | Lower bounds |
| Ablation framework | Which tensors matter |
| Cross-model testing | Generalization beyond Claude |
| Mixed-effects models | Rigorous statistics |

### 5.3 Test Arms (Full)

| Arm | Content | Length | Notation | Purpose |
|-----|---------|--------|----------|---------|
| CTN | Full | ~450 | Math | Test condition |
| Literal | Full | ~500 | Prose | Structure control |
| Idiomatic-Long | Full | ~450 | English | Length-matched control |
| Idiomatic-Short | Minimal | ~150 | English | Brevity baseline |
| Placebo-Math | Minimal | ~200 | Math glyphs | Notation placebo |
| Null-Baseline | None | ~50 | None | Lower bound |
| Heuristic | None | ~100 | Regex | Naive baseline |

### 5.4 Factorial Comparisons

| Comparison | Isolates |
|------------|----------|
| CTN vs Literal | Notation effect (same content) |
| CTN vs Idiomatic-Long | Notation + structure vs good English (same length) |
| Idiomatic-Long vs Idiomatic-Short | Content/length effect |
| CTN vs Placebo-Math | CTN content vs just math symbols |
| All vs Null-Baseline | Does prompting matter at all? |
| All vs Heuristic | Does LLM beat simple regex? |

### 5.5 Ablation Study

Test CTN kernel with individual tensors removed:

| Variant | Removed | Tests |
|---------|---------|-------|
| CTN-full | None | Baseline |
| CTN-no-v1 | Atomic_Clarity | Field isolation |
| CTN-no-v2 | Extraction_Accuracy | Precision guidance |
| CTN-no-v5 | Reality_Anchor | Hallucination control |
| CTN-no-v7 | OutputSchema_Lock | Format enforcement |
| CTN-minimal | v3,v4,v6 only | Core vs auxiliary |

### 5.6 Cross-Model Testing

| Model | Purpose |
|-------|---------|
| Claude claude-sonnet-4-20250514 | Primary (Phase 1-2) |
| GPT-4 | Cross-family generalization |
| Gemini 1.5 Pro | Cross-family generalization |

**Question:** Does CTN advantage hold across model families, or is it Claude-specific?

### 5.7 Statistical Analysis

**Mixed-Effects Model:**
```
score ~ kernel + (1|document) + (1|doc_type) + (1|difficulty)
```

- Fixed effect: kernel (what we're testing)
- Random effects: document, doc_type, difficulty (sources of variance)

**Multiple Comparisons:**
- 15+ pairwise comparisons
- Holm-Bonferroni correction
- Report effect sizes for all

### 5.8 Decision Gate

#### Phase 3 Budget

| Resource | Budget | Hard Limit |
|----------|--------|------------|
| Time | 10 days initial + 5 days revision | 15 days total |
| Cost | ~$150 initial + ~$100 revision | $300 |
| Revision cycles | 2 | 2 |

#### Why We're Here

We only reach Phase 3 if:
1. Phase 1 showed signal (CTN > Idiomatic)
2. Phase 2 replicated on real data
3. We want to understand WHY and publish

Phase 3 is optional. Practical value is already established. This is for scientific contribution.

#### Outcome Tree

```
Phase 3 Factorial Analysis Complete
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESULT CLASSIFICATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  A. NOTATION EFFECT CONFIRMED                                   │
│     CTN > Literal (same content, different notation)           │
│     Effect size meaningful (d > 0.3)                           │
│     → Notation itself provides value                            │
│     → Strong paper: "Mathematical notation improves..."         │
│                                                                 │
│  B. CONTENT EFFECT (NOT NOTATION)                               │
│     CTN ≈ Literal > Idiomatic-Long                             │
│     → Extra content helps, notation doesn't matter             │
│     → Still publishable: "Structured prompts improve..."       │
│     → Practical recommendation: Use Literal (more readable)    │
│                                                                 │
│  C. LENGTH EFFECT                                               │
│     Idiomatic-Long > Idiomatic-Short                           │
│     CTN ≈ Idiomatic-Long                                       │
│     → It's just more instructions, not notation                │
│     → Publishable: "Longer prompts help, notation neutral"     │
│                                                                 │
│  D. ABLATION INSIGHTS                                           │
│     Specific tensors (v1, v2, v5) drive effect                 │
│     → Identifies key components                                │
│     → Publishable: "Reality Anchor reduces hallucination"      │
│                                                                 │
│  E. CROSS-MODEL DIVERGENCE                                      │
│     Effect holds for Claude, not GPT/Gemini                    │
│     → Model-specific finding                                   │
│     → Publishable with scope: "CTN helps Claude extraction"    │
│                                                                 │
│  F. PLACEBO EFFECT                                              │
│     Placebo-Math ≈ CTN                                         │
│     → Math symbols help attention, not CTN content             │
│     → Interesting negative: "Novel glyphs, not structure"      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Every Outcome is Publishable

This is key: Phase 3 cannot "fail" in the traditional sense. Every outcome answers a question:

| Outcome | Paper Title |
|---------|-------------|
| A: Notation helps | "Mathematical Notation Improves LLM Extraction Accuracy" |
| B: Content helps | "Structured Prompts Outperform Concise Instructions" |
| C: Length helps | "Prompt Length Predicts Extraction Performance" |
| D: Specific tensors | "Reality Anchor: Reducing Hallucination Through Constraint Specification" |
| E: Model-specific | "Model-Dependent Effects of Formal Notation in Prompts" |
| F: Placebo works | "Attention Effects of Novel Symbols in LLM Prompts" |

**Phase 3 success = answering the "why" question, regardless of answer.**

#### Exit Criteria

**We complete Phase 3 when:**

| Condition | Deliverable |
|-----------|-------------|
| Factorial comparison complete | Know notation vs content vs length |
| Ablation complete | Know which components matter |
| Cross-model tested | Know if it generalizes |
| Mechanism hypothesis formed | Can explain findings |

**We stop early when:**

| Condition | Reason | Action |
|-----------|--------|--------|
| Resources exhausted | Time/cost limit hit | Publish what we have |
| Clear answer emerges early | Don't need full factorial | Focus paper on key finding |
| Diminishing returns | More tests won't change conclusion | Write up |

#### Validation Criteria (Must Pass for Publication)

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Pre-registered analysis | Followed plan | Credibility |
| Multiple comparisons corrected | All p-values adjusted | Rigor |
| Effect sizes reported | Cohen's d for all comparisons | Interpretability |
| Ablation completed | At least 4 tensor variants tested | Mechanism insight |
| Reproducibility package | Code + data available | Open science |

#### Summary: Phase 3 Success Definition

**Strong Success:**
- Causal mechanism identified
- Effect generalizes across models
- Clear practical recommendations
- Paper accepted at top venue

**Good Success:**
- Mechanism partially identified
- Some cross-model evidence
- Nuanced recommendations
- Paper accepted at workshop/journal

**Minimum Success:**
- "Why" question answered (even if answer is "it doesn't matter")
- Findings documented
- Technical report published

### 5.9 Phase 3 Deliverables

1. **Paper:** Peer-reviewable manuscript
   - Introduction: CTN hypothesis
   - Methods: Full experimental design
   - Results: All comparisons, ablations, cross-model
   - Discussion: Mechanism interpretation
   - Professional figures (vendor-produced)
   - Copy-edited prose

2. **Supplementary Materials:**
   - Full results tables (all 7 arms × all metrics)
   - Ablation details
   - Cross-model comparison
   - Statistical analysis code

3. **Data & Code:** Full reproducibility package
   - All kernels
   - All test documents (synthetic + real)
   - Ground truth
   - Evaluation code
   - Analysis notebooks

4. **Claims Supported (depending on results):**
   - "CTN notation improves extraction" (if CTN > Literal)
   - "Effect is not just prompt length" (if CTN > Idiomatic-Long)
   - "Effect generalizes across models" (if cross-model holds)
   - "v₅ (Reality_Anchor) drives hallucination reduction" (if ablation shows)

---

## 6. Infrastructure Principles

### 6.1 Extensibility Requirements

Every component must support future extension without rewrite:

| Component | Phase 1 | Phase 2 | Phase 3 | Extension Point |
|-----------|---------|---------|---------|-----------------|
| Kernels | 2 | 3 | 7 | Drop .txt file |
| Document formats | Text | Text | Text + PDF | Subclass Document |
| Metrics | All | All | All | Already implemented |
| Statistics | Simple | Clustered | Mixed-effects | Config switch |
| Models | Claude | Claude | Claude + GPT + Gemini | Config switch |
| Baselines | None | None | 2 | Add to baselines/ |

### 6.2 No Technical Debt Rule

Before implementing any shortcut, ask:
> "Will this block us in Phase 3?"

If yes, implement the general solution now.

**Examples:**

| Shortcut | Blocks Phase 3? | Decision |
|----------|-----------------|----------|
| Hardcode 2 kernels | Yes (need 7) | Use kernel loader |
| Skip candidate validation | Yes (need for evidence scoring) | Implement now |
| Field-level statistics | Yes (need clustering) | Aggregate to doc-level |
| Inline config | Yes (need phase configs) | YAML from start |

### 6.3 Testing Requirements

All code must have tests:

```python
# test_metrics.py
def test_evidence_validity_null_on_existing_field():
    """If field exists but model says null, evidence = 0."""
    gt = GroundTruth(exists_in_document=True)
    ext = Extraction(value=None, evidence=None)
    assert evidence_validity(ext, gt, "doc text") == 0.0

def test_evidence_validity_fabricated_quote():
    """If quote doesn't exist in document, evidence = 0."""
    gt = GroundTruth(exists_in_document=True)
    ext = Extraction(value="123", evidence=Evidence(quote="fabricated"))
    assert evidence_validity(ext, gt, "actual document text") == 0.0

def test_composite_null_strategy():
    """Null-everywhere strategy should score ~0.36."""
    score = simulate_null_strategy(test_documents)
    assert 0.30 < score < 0.40
```

---

## 7. Risk Registry

### 7.1 Phase 1 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Synthetic data too easy | Ceiling effect, no differentiation | Include hard/adversarial cases |
| Synthetic data unrealistic | Results don't transfer | Phase 2 validates on real data |
| Scoring bugs | Invalid results | Unit tests, manual audit |
| LLM generation leaks style | Inflated performance | Use different model or templates |

### 7.2 Phase 2 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Annotation quality | Noisy ground truth | Double annotation, inter-rater agreement |
| Domain mismatch | Poor performance | Select relevant domains |
| Blind collaborator unavailable | No external validation | Find backup collaborator |

### 7.3 Phase 3 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Multiple comparison inflation | False positives | Pre-registered tests, correction |
| Cross-model inconsistency | Limited generalization | Report honestly, scope claims |
| Mechanism unclear despite data | No causal story | Ablations, detailed analysis |

---

## 8. Resource Strategy: Buy vs Build

### 8.1 Philosophy

**Buy grunt work. Build decisions.**

Time is the scarce resource. Money is not (budget threshold: $50k). Every hour spent on annotation is an hour not spent on analysis and iteration.

### 8.2 What Can Be Bought

| Category | Buyable | Not Buyable |
|----------|---------|-------------|
| **Data work** | Annotation, verification, transcription | Ground truth design |
| **Computation** | API calls, cloud resources | - |
| **Production** | Figures, formatting, copy-editing | Analysis, interpretation |
| **Expertise** | Statistical review, domain experts | Core experimental decisions |

### 8.3 Phase 1: Buy vs Build

| Task | Build (Hours) | Buy (Cost) | Decision |
|------|---------------|------------|----------|
| Core infrastructure | 16 hrs | Can't buy | **Build** |
| Synthetic generator | 4 hrs | Can't buy | **Build** |
| Generate 200 docs (100 DEV + 100 TEST) | 0.5 hrs | ~$10 API | **Buy** |
| Verify 80 docs ground truth (40 per set) | 8 hrs | $800 | **Buy** |
| Run evaluation (DEV iterations + TEST) | 3 hrs | ~$40 API | **Buy** |
| Analyze results | 2 hrs | Can't buy | **Build** |

**Phase 1 Totals:**

| Path | Your Hours | Cost | Calendar Days |
|------|------------|------|---------------|
| 100% Build | 30 hrs | $50 | 6 days |
| **Recommended (Buy verification)** | **22 hrs** | **$850** | **4 days** |

### 8.4 Phase 2: Buy vs Build

| Task | Build (Hours) | Buy (Cost) | Decision |
|------|---------------|------------|----------|
| Literal kernel | 2 hrs | Can't buy | **Build** |
| Collect 100 real docs | 3 hrs | $0 (you have) | **Build** |
| **Annotate 100 real docs** | **40 hrs** | **$2,500** | **BUY** |
| Double-annotate 20 docs | 5 hrs | $500 | **Buy** |
| Annotation adjudication | 3 hrs | Can't buy | **Build** |
| Adapt public dataset (150 docs) | 12 hrs | $1,200 | **Buy** |
| Run evaluation | 3 hrs | ~$75 API | **Buy** |
| Build blind validation kit | 2 hrs | Can't buy | **Build** |
| Analyze results | 4 hrs | Can't buy | **Build** |

**Phase 2 Totals:**

| Path | Your Hours | Cost | Calendar Days |
|------|------------|------|---------------|
| 100% Build | 75 hrs | $75 | 15 days |
| **Recommended (Buy annotation)** | **17 hrs** | **$4,350** | **6 days** |

### 8.5 Phase 3: Buy vs Build

| Task | Build (Hours) | Buy (Cost) | Decision |
|------|---------------|------------|----------|
| Additional kernels (4) | 4 hrs | Can't buy | **Build** |
| Baselines implementation | 2 hrs | Can't buy | **Build** |
| Run 7-arm factorial | 4 hrs | ~$250 API | **Buy** |
| Cross-model testing (GPT, Gemini) | 2 hrs | ~$200 API | **Buy** |
| Ablation runs | 2 hrs | ~$150 API | **Buy** |
| Statistical analysis | 10 hrs | $1,500 (review) | **Build + Buy review** |
| Write paper | 40 hrs | Can't buy (need your voice) | **Build** |
| Figures/visualization | 8 hrs | $800 | **Buy** |
| Copy-editing | 0 hrs | $500 | **Buy** |

**Phase 3 Totals:**

| Path | Your Hours | Cost | Calendar Days |
|------|------------|------|---------------|
| 100% Build | 72 hrs | $600 | 15 days |
| **Recommended (Buy production)** | **58 hrs** | **$3,400** | **12 days** |

### 8.6 Total Project Budget

| Phase | Your Hours | Cost | Calendar Days |
|-------|------------|------|---------------|
| Phase 1 | 22 hrs | $850 | 4 days |
| Phase 1 Investigation (if needed) | +15 hrs | +$300 | +4 days |
| Phase 2 | 17 hrs | $4,350 | 6 days |
| Phase 2 Investigation (if needed) | +10 hrs | +$500 | +3 days |
| Phase 3 | 58 hrs | $3,400 | 12 days |
| Phase 3 Revision (if needed) | +15 hrs | +$500 | +3 days |
| **Totals (worst case)** | **137 hrs** | **$9,900** | **32 days** |
| **Totals (happy path)** | **97 hrs** | **$8,600** | **22 days** |

**Conservative estimate with 50% buffer: ~$15,000**

This is well under the $50k threshold. Cost is not a constraint.

### 8.7 Vendor Recommendations

**Annotation Services (Phase 2):**

| Service | Est. Cost/Doc | Quality | Speed | Recommended For |
|---------|---------------|---------|-------|-----------------|
| Scale AI | $20-35 | High | 3-5 days | Complex documents |
| Labelbox | $15-25 | High | 3-5 days | Structured forms |
| Surge AI | $15-30 | High | 2-4 days | Good instructions |
| Upwork (vetted) | $25-40/hr | Variable | Negotiate | Custom schemas |

**Recommendation:** Surge AI or Scale AI with detailed annotation guidelines.

**Statistical Review (Phase 3):**

| Option | Cost | What You Get |
|--------|------|--------------|
| Upwork statistician | $500-1,000 | Analysis review |
| Academic collaborator | $0-500 | Co-authorship trade |
| Statistical consulting firm | $1,500-3,000 | Full review + recommendations |

**Recommendation:** Upwork for review, unless pursuing top venue (then consulting firm).

**Production (Phase 3):**

| Service | Cost | What You Get |
|---------|------|--------------|
| Fiverr (figures) | $200-500 | Publication-ready charts |
| Technical illustrator | $500-1,000 | Custom diagrams |
| Copy-editor (academic) | $300-600 | Grammar, style, clarity |

### 8.8 Annotation Specification

When engaging annotation service, provide:

```markdown
## Annotation Task: Document Field Extraction

### Document Types
- Invoices, Contracts, Medical Records, Receipts, Resumes

### Per Document, Extract:
For each field in the schema:

1. **value**: The extracted value (or null if not present)
2. **evidence_quote**: Verbatim text from document containing the value
3. **evidence_page**: Page number where found
4. **status**: "ok" | "missing" | "ambiguous"
5. **candidates**: If ambiguous, list all possible values with quotes

### Quality Requirements
- Quotes must be EXACT (copy-paste, not retyped)
- If field not in document, mark as missing (don't guess)
- If multiple valid values exist, mark as ambiguous
- Include page numbers for all evidence

### Deliverable Format
JSON file per document (schema provided)

### Quality Check
20% of documents will be double-annotated for agreement check
```

### 8.9 Timeline Comparison

```
100% BUILD PATH (135 hours, 32 days, $600)
══════════════════════════════════════════════════════════════════

Week 1       Week 2       Week 3       Week 4       Week 5
─────────────────────────────────────────────────────────────────
│ Phase 1    │ Phase 1    │ Phase 2    │ Phase 2    │ Phase 2
│ Build      │ Verify     │ Annotate   │ Annotate   │ Run
│            │ (5 hrs)    │ (20 hrs)   │ (20 hrs)   │
─────────────────────────────────────────────────────────────────

Week 6       Week 7       Week 8
─────────────────────────────────────────────────────────────────
│ Phase 3    │ Phase 3    │ Phase 3
│ Build+Run  │ Analyze    │ Write
│            │            │ (40 hrs)
─────────────────────────────────────────────────────────────────


RECOMMENDED BUY PATH (95 hours, 22 days, $9,000)
══════════════════════════════════════════════════════════════════

Week 1       Week 2       Week 3       Week 4
─────────────────────────────────────────────────────────────────
│ Phase 1    │ Phase 2    │ Phase 2    │ Phase 3    
│ Build+Run  │ Setup      │ Run+Analyze│ Build+Run  
│            │ ├─Annotators working──┤ │            
│            │ (parallel)            │ │            
─────────────────────────────────────────────────────────────────

Week 5
─────────────────────────────────────────────────────────────────
│ Phase 3
│ Write
│ ├─Figures vendor──┤
│ ├─Copy-edit───────┤
─────────────────────────────────────────────────────────────────

                         ▲
                         │
              10 days saved by buying annotation
              Annotation happens in parallel with Phase 2 setup
```

### 8.10 Decision Rules

**Always buy:**
- Annotation (>20 docs)
- API compute
- Copy-editing
- Figures (unless you enjoy it)

**Never buy:**
- Kernel design
- Experimental decisions
- Result interpretation
- Paper narrative

**Buy if stuck:**
- Statistical consulting (if reviewers challenge methods)
- Domain expert review (if results are surprising)
- Second annotator (if agreement is low)

## 9. Timeline Summary

### 9.1 Resource Budget Overview (Buy Path - Recommended)

| Phase | Your Hours | Cost | Calendar Days |
|-------|------------|------|---------------|
| **1: Practical** | 22 hrs | $850 | 4 days |
| **1: Investigation (if needed)** | +15 hrs | +$300 | +4 days |
| **2: Applied** | 17 hrs | $4,350 | 6 days |
| **2: Investigation (if needed)** | +10 hrs | +$500 | +3 days |
| **3: Academic** | 58 hrs | $3,400 | 12 days |
| **3: Revision (if needed)** | +15 hrs | +$500 | +3 days |
| **Happy Path Total** | **97 hrs** | **$8,600** | **22 days** |
| **Worst Case Total** | **137 hrs** | **$9,900** | **32 days** |

*Conservative estimate with 50% buffer: ~$15,000*
*Budget threshold for concern: $50,000 (we are well under)*

### Decision Tree Timeline (Buy Path)

```
Day 0-4: Phase 1 Build + Run
         Parallel: Verification vendor checking 40 docs
         │
         ├── Signal? → Day 5: Begin Phase 2
         │              Parallel: Annotation vendor starts 100 docs
         │
         └── No Signal? → Day 5-8: Investigation Cycles (max 3)
                          │
                          ├── Signal Found → Begin Phase 2
                          │
                          └── No Signal after 3 cycles → Pivot Decision

Day 5-12: Phase 2 (if reached)
          Day 5-6: Setup + Literal kernel
          Day 5-10: Annotation vendor working (parallel)
          Day 10-11: Run evaluation  
          Day 11-12: Analyze + send blind validation
          │
          ├── Replicates? → Wait for blind validation
          │                  │
          │                  ├── Positive → Phase 2 Success
          │                  └── Negative → Document scope limits
          │
          └── Doesn't Replicate? → Day 12-16: Investigation (max 2 cycles)

Day 13-22: Phase 3 (if pursuing academic)
           Day 13-15: Build additional kernels + baselines
           Day 15-18: Run factorial + cross-model
           Day 18-19: Statistical analysis
           Day 19-22: Write paper
           Parallel: Figures vendor (Day 18-21)
           Parallel: Copy-edit (Day 21-22)
           │
           └── Any outcome is publishable
```

### Realistic Scenarios (Buy Path)

| Scenario | Timeline | Cost | Your Hours | Outcome |
|----------|----------|------|------------|---------|
| **Fast fail** | 8 days | $1,200 | 37 hrs | "No signal after investigation" |
| **Narrow win** | 10 days | $1,400 | 42 hrs | "CTN helps for subset of docs" |
| **Practical success** | 14 days | $5,500 | 57 hrs | "CTN works, here's when to use it" |
| **Full applied** | 16 days | $6,000 | 62 hrs | "CTN works, blind validated" |
| **Full academic** | 22 days | $9,000 | 97 hrs | Peer-reviewed paper |
| **Worst case** | 32 days | $15,000 | 137 hrs | All investigations, still publishable |

### Checkpoint Schedule (Buy Path)

| Day | Checkpoint | Decision |
|-----|------------|----------|
| 4 | Phase 1 complete | Signal? Investigation? Phase 2? |
| 6 | Revision cycle 1 (if needed) | Working? Another revision? |
| 8 | Phase 1 hard limit | Proceed or pivot |
| 10 | Phase 2 setup complete, annotations in | Quality check annotations |
| 12 | Phase 2 evaluation complete | Replicated? Blind validation? |
| 14 | Blind validation sent | Wait for external |
| 16 | Phase 2 hard limit | Proceed to Phase 3? |
| 22 | Phase 3 complete | Write paper |

### Investment Summary

| Scenario | Time | Cost | Your Hours |
|----------|------|------|------------|
| **Fast fail (Phase 1 no signal after investigation)** | 8 days | $1,200 | 37 hrs |
| **Practical win (Phase 1 + Phase 2)** | 14 days | $5,500 | 57 hrs |
| **Full academic (all phases)** | 22 days | $9,000 | 97 hrs |
| **Worst case (all investigations)** | 32 days | $15,000 | 137 hrs |

---

## 10. Success Definition by Phase

### Philosophy: Founder Mentality

**Try like a founder:** We don't give up at the first obstacle. No signal? Investigate. Still no signal? Revise. Still nothing? Try a different angle.

**Be smart like a founder:** We don't throw infinite resources at a dead end. We set budgets. We define pivot triggers. We recognize when to change direction vs when to push harder.

**Every run teaches us something:** A "failed" run that shows ceiling effect teaches us to increase difficulty. A "failed" run that shows floor effect teaches us our task is too hard. There are no wasted experiments if we learn from them.

### Phase 1 Success Spectrum

| Level | Criteria | Meaning |
|-------|----------|---------|
| **Ideal** | CTN > Idiomatic by >10%, p<0.05, d>0.5 | Strong signal, high confidence |
| **Good** | CTN > Idiomatic by 5-10%, p<0.05 | Clear signal, proceed confidently |
| **Acceptable** | CTN > Idiomatic by 3-5%, p<0.10, after revision | Weak but real signal |
| **Narrow** | Signal in subset of document types | Scoped success |
| **Pivot** | No signal after 3 cycles, 10 days | Change direction, not give up |

### Phase 2 Success Spectrum

| Level | Criteria | Meaning |
|-------|----------|---------|
| **Ideal** | Replicates + Blind positive + Notation helps | Full validation |
| **Good** | Replicates + Blind mixed | Partial generalization |
| **Acceptable** | Replicates, blind not available | Unvalidated but promising |
| **Narrow** | Domain-specific effect | "Works for X" |
| **Informative Negative** | Synthetic ≠ Real | Important finding |

### Phase 3 Success Spectrum

| Level | Criteria | Meaning |
|-------|----------|---------|
| **Ideal** | Mechanism identified, cross-model, top venue | Major contribution |
| **Good** | Mechanism identified, single model | Solid paper |
| **Acceptable** | Partial mechanism, good ablations | Workshop paper |
| **Minimum** | "Why" answered (even if "doesn't matter") | Technical report |

### What "Failure" Actually Means

| Apparent Failure | What We Learn | Value |
|------------------|---------------|-------|
| Phase 1 no signal | CTN doesn't help extraction (or our kernel is wrong) | Avoid wasted effort |
| Phase 2 doesn't replicate | Synthetic benchmarks misleading | Important caveat |
| Phase 3 notation doesn't matter | Content > notation | Practical guidance |
| Cross-model doesn't hold | Model-specific effect | Scope understanding |

**There are no true failures, only findings.** But we do set boundaries on how much we invest before changing direction.

### Project Exit Conditions

We stop working on CTN evaluation entirely when:

| Condition | After | Meaning |
|-----------|-------|---------|
| **Exhaustion** | 3 pivots across tasks | CTN doesn't help anywhere we can find |
| **Success** | Phase 2 or 3 complete | Question answered, ship it |
| **Opportunity cost** | Better use of time identified | Strategic pivot |

We never stop because:
- First experiment didn't work
- It's hard
- We're frustrated

### Decision Framework Summary

```
                    ┌─────────────────┐
                    │  Run Experiment │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Signal present? │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐   ┌─────▼─────┐  ┌─────▼─────┐
         │  Yes    │   │   Weak    │  │    No     │
         └────┬────┘   └─────┬─────┘  └─────┬─────┘
              │              │              │
              ▼              ▼              ▼
         Next Phase    Investigate     Investigate
                             │              │
                      ┌──────▼──────┐ ┌─────▼─────┐
                      │Signal found?│ │Found why? │
                      └──────┬──────┘ └─────┬─────┘
                             │              │
                     ┌───────┼───────┐      │
                     │       │       │      │
                ┌────▼──┐ ┌──▼──┐ ┌──▼───┐  │
                │ Yes   │ │Maybe│ │  No  │  │
                └───┬───┘ └──┬──┘ └──┬───┘  │
                    │        │       │      │
                    ▼        ▼       ▼      ▼
               Next      Revise   Revise  Revise
               Phase     & Run    & Run   & Run
                                     │      │
                           ┌─────────┴──────┴─────────┐
                           │ 3 revision cycles done?  │
                           └─────────────┬────────────┘
                                         │
                              ┌──────────┼──────────┐
                              │          │          │
                         ┌────▼───┐ ┌────▼────┐ ┌───▼───┐
                         │Signal? │ │Partial? │ │  No   │
                         └────┬───┘ └────┬────┘ └───┬───┘
                              │          │          │
                              ▼          ▼          ▼
                         Next Phase  Narrow     Pivot or
                                     Scope       Park
```

---

## Appendix A: Final Red Team Status

### Tier-1 Issues: FIXED

**Issue 1: Evidence/page/status could score high for wrong values**
- Problem: Wrong value + real quote = 0.70 composite
- Fix: Evidence/page/status are now gated on value correctness
- Result: Wrong value + real quote = 0.15 composite (schema only)

**Issue 2: Output format contradiction (evidence null vs object)**
- Problem: Conflicting rules about `evidence: null` vs `evidence: {quote: null, page: null}`
- Fix: Standardized to object form only, clear rules per status
- Result: No ambiguity, page type is always int or null

### Tier-2 Issues: DOCUMENTED

| Issue | Status | Mitigation |
|-------|--------|------------|
| Small-model judge false negatives | Documented | Calibrate on DEV, downweight if <95% |
| Evidence set definition implicit | Documented | Phase-1 limitation, address in Phase-3 |
| Extra fields ignored | Tracked | Report extra_field_rate, add penalty if needed |

### Validation Path

**Do NOT need another full red team round.**

**DO need:** 10-case scorer micro red team before any real evaluation.

| Case | Tests |
|------|-------|
| 1 | Wrong value + real quote → low score |
| 2 | Correct value + fabricated quote → evidence=0 |
| 3 | Ambiguous + random candidates → low score |
| 4 | Ambiguous GT but status=ok → penalized |
| 5 | Missing GT + hallucinated value → near 0 |
| 6 | Missing + non-null quote → format penalty |
| 7 | Duplicate fields → schema failure |
| 8 | Omitted field → miss |
| 9 | Quote doesn't contain value → evidence=0 |
| 10 | Page off-by-one → page=0 |

**Gate:** All 10 pass → proceed to DEV evaluation.

### Final Status

✅ **GREEN LIGHT** (pending 10-case scorer validation)

---
