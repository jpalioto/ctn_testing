# FUTURE_DIRECTIONS.md

Running list of observations, hypotheses, and potential extensions. Capture now, pursue when signal warrants.

---

## Critical Learnings

### 2024-12-13: Gemini Implicit Caching Invalidates Cross-Run Comparisons

**Problem:** Gemini 2.5 models have implicit caching enabled by default. When the same prompt prefix is sent multiple times, subsequent requests return cached responses. This caused identical scores between CTN and idiomatic kernels on repeated runs — the cache was masking real differences.

**Symptoms:**
- Run 1: CTN 1.00, Idiomatic 0.89 (Δ = +0.11)
- Run 2: CTN 0.89, Idiomatic 0.89 (Δ = 0.00) ← Both cached!

**Root cause:** Implicit caching matches on prompt prefix. Same document + same kernel template = cache hit. No API parameter exists to disable this in the `google-genai` SDK.

**Solution:** Prepend a zero-width Unicode character to each prompt. Different character = different prefix = no cache hit.

```python
# Zero-width characters (invisible, but tokenized differently)
ZW_CHARS = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff']
ZW_NAMES = ['ZWSP', 'ZWNJ', 'ZWJ', 'WJ', 'BOM']

# Deterministic cycling
prefix_idx = test_index % len(ZW_CHARS)
prefix_char = ZW_CHARS[prefix_idx]
salted_prompt = prefix_char + original_prompt
```

**Why zero-width:** 
- Regular characters/numbers could affect extraction (e.g., "9" at start confuses date parsing)
- Emojis/symbols might cause model to respond with emojis
- Zero-width chars are invisible, don't affect semantics, but break prefix matching

**Critical:** Apply to ALL models, not just Gemini. Ensures consistent methodology across providers.

**Documentation:** Store `cache_prefix` name in results JSON for reproducibility.

---

## Observations

### 2024-12-13: Gemini shows larger CTN delta than Claude

**Data:** n=2 (early, not significant)

| Model | CTN | Idiomatic | Δ |
|-------|-----|-----------|---|
| Claude | 1.00 | 0.95 | +0.05 |
| Gemini | 1.00 | 0.89 | +0.11 |

**Hypothesis:** Models with less post-training (instruction-following) benefit more from CTN's tighter prompt geometry. Gemini's focus on pre-training means it drifts toward priors faster; CTN's structured constraints compensate.

**Testable predictions:**
1. Larger CTN delta on Gemini persists at n=30+
2. Delta grows with document difficulty
3. Evidence scores show larger improvement than value scores (structure anchors extraction)

**If confirmed:** Paper candidate — "Prompt Geometry as Compensatory Structure for Instruction-Following Variance Across Model Families"

---

## Future Extensions

### Fine-tuned model comparison
- Hypothesis: Fine-tuned models need CTN least (structure baked into weights)
- Test: Run same eval on fine-tuned extraction model
- Expected: Minimal CTN delta, high baseline performance

### Per-component score analysis
- Break down CTN benefit by: value, evidence, page, status
- Identify which extraction components CTN helps most
- Inform kernel design

### Cross-domain testing
- Current: Invoice extraction
- Future: Contracts, medical records, scientific papers
- Question: Does CTN benefit generalize or domain-specific?

### Difficulty stratification
- Tag documents: easy, medium, hard, adversarial
- Analyze CTN delta by difficulty tier
- Prediction: CTN benefit increases with difficulty

### Token efficiency analysis
- CTN uses more input tokens (structured prompt)
- Track: (composite_score_delta) / (token_delta)
- ROI calculation for production deployment

### Reasoning mode comparison
- Claude extended thinking vs standard
- Gemini thinking mode vs standard
- Does reasoning reduce CTN benefit? (internal structure compensates)

---

## Technical Debt

- [x] Fix null_baseline display bug in summary
- [x] Add cache-bust prefix for reproducible Gemini results
- [ ] Add OpenAI provider implementation
- [ ] Config validation with better error messages
- [ ] Results comparison tool (diff two runs)
- [ ] HTML report generator
- [ ] Print statistical comparisons to console

---

## Experimental Methodology

### Single-Blind Kernel Generation (2024-12-14)

**Problem:** Experimenter knowledge of CTN could leak into idiomatic baseline design. Even unconscious choices (e.g., including "list candidates for ambiguous fields") could give idiomatic kernel CTN-like behavior, contaminating the comparison.

**Solution:** Single-blind kernel generation.

1. Prompt sent to Gemini (a model not involved in the evaluation):
   ```
   I need a system prompt for an LLM that extracts information from business documents.
   
   Requirements:
   - The output must be JSON
   - The LLM will receive a document as the user message
   - The system prompt should be concise (under 100 words)
   
   Write only the system prompt, nothing else.
   ```

2. Gemini's response used verbatim as the idiomatic kernel. No edits.

**Source of experimental error:**
- Gemini's training may include CTN-like patterns
- The prompt itself ("extract", "JSON") carries some structural bias
- Different prompt to Gemini would yield different baseline

**Mitigation:**
- Document the exact prompt used
- Document the exact response received
- Future work: multiple idiomatic baselines from different generators

**Classification:** Known confound. Documented for transparency. Does not invalidate results but limits generalization of "idiomatic" label.

---

## Notes

*Add observations here as experiments progress.*
