"""Main evaluation runner."""
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from datetime import datetime

from ..core import (
    EvaluationConfig,
    ModelConfig,
    DocumentSchema,
)
from ..utils.network import make_client

from ..metrics import composite_score
from .kernel import Kernel, NullBaseline, load_kernel
from .results import RunResults, DocumentResult, FieldResult
from .judge import judge_extraction, JudgeResult
from ..utils.hashing import md5_hash, hash_config

from ..core import DocumentWithGroundTruth
from ..core.loaders import load_document_set
from ..utils.hashing import hash_file
from .kernel import Kernel, NullBaseline, RenderedPrompt

# Zero-width characters for cache busting (invisible, deterministic)
ZW_CHARS = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff']
ZW_NAMES = ['ZWSP', 'ZWNJ', 'ZWJ', 'WJ', 'BOM']


@dataclass
class RunConfig:
    """Runtime configuration for a run."""
    config: EvaluationConfig
    data_dir: Path
    output_dir: Path
    schema: DocumentSchema
    verbose: bool = True

@dataclass
class ExtractionContext:
    """Immutable context for a single extraction attempt.
    
    Pre-computes hashes and metadata so _extract_single stays focused
    on the actual extraction logic.
    """
    doc: DocumentWithGroundTruth
    model: ModelConfig
    kernel: Kernel
    timestamp: datetime
    
    # Pre-computed hashes
    document_hash: str | None
    kernel_hash: str | None
    gt_hash: str
    model_config_hash: str
    
    # Cache prefix (for Gemini cache-busting)
    cache_prefix: str
    cache_prefix_char: str


class Runner:
    """
    Evaluation runner.

    Two-pass architecture:
    1. Extract: models × kernels × documents → raw responses
    2. Judge: raw responses + ground truth → scored fields
    """

    def __init__(self, run_config: RunConfig):
        self._config = run_config.config
        self._data_dir = run_config.data_dir
        self._output_dir = run_config.output_dir
        self._schema = run_config.schema
        self._verbose = run_config.verbose

        self._results: RunResults | None = None
        self._clients: dict[str, Any] = {}
        self._kernels: dict[str, Kernel] = {}

        self._test_index = 0

    def _log(self, msg: str):
        if self._verbose:
            print(msg)

    def _get_client(self, model: ModelConfig):
        """Get or create client for model."""
        if model.name not in self._clients:
            self._clients[model.name] = make_client(model)
        return self._clients[model.name]

    def _get_kernel(self, kernel_name: str, kernel_path: str) -> Kernel:
        """Get or create kernel."""
        if kernel_name not in self._kernels:
            path = self._data_dir.parent.parent / kernel_path
            self._kernels[kernel_name] = load_kernel(path, self._schema)
        return self._kernels[kernel_name]

    def _get_judges(self, model: ModelConfig) -> list[ModelConfig]:
        """Get judge models based on policy."""
        policy = self._config.judge_policy

        if policy == "single":
            return [self._config.judge_models[0]] if self._config.judge_models else []

        elif policy == "match_provider":
            matched = [j for j in self._config.judge_models if j.provider == model.provider]
            return matched if matched else self._config.judge_models[:1]

        elif policy == "full_cross":
            return self._config.judge_models

        return []

    # =========================================================================
    # Pass 1: Extraction
    # =========================================================================

    def _prepare_context(
        self,
        doc: DocumentWithGroundTruth,
        model: ModelConfig,
        kernel: Kernel,
    ) -> ExtractionContext:
        """Build immutable context for extraction.
        
        Handles hashing for both text-mode and file-mode documents.
        """
        # Hash document content
        if doc.document.has_text and doc.document.text:
            doc_hash = md5_hash(doc.document.text)
        elif doc.document.has_file and doc.document.file_path:
            doc_hash = hash_file(str(doc.document.file_path))
        else:
            doc_hash = None
        
        # Hash kernel template
        kernel_hash = None
        if hasattr(kernel, 'raw_content') and kernel.raw_content:
            kernel_hash = md5_hash(kernel.raw_content)
        
        # Cache prefix rotation
        prefix_idx = self._test_index % len(ZW_CHARS)
        self._test_index += 1
        
        return ExtractionContext(
            doc=doc,
            model=model,
            kernel=kernel,
            timestamp=datetime.now(),
            document_hash=doc_hash,
            kernel_hash=kernel_hash,
            gt_hash=md5_hash(str(doc.ground_truth)),
            model_config_hash=hash_config(model),
            cache_prefix=ZW_NAMES[prefix_idx],
            cache_prefix_char=ZW_CHARS[prefix_idx],
        )
    
    def _build_result(
        self,
        ctx: ExtractionContext,
        fields: list[FieldResult],
        raw_response: str | None = None,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> DocumentResult:
        """Build DocumentResult from context.
        
        Single place to construct results ensures consistency.
        """
        return DocumentResult(
            doc_id=ctx.doc.document.id,
            model=ctx.model.name,
            kernel=ctx.kernel.name,
            fields=fields,
            raw_response=raw_response,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            cache_prefix=ctx.cache_prefix,
            timestamp=ctx.timestamp,
            document_hash=ctx.document_hash,
            kernel_hash=ctx.kernel_hash,
            gt_hash=ctx.gt_hash,
            model_config_hash=ctx.model_config_hash,
            model_temperature=ctx.model.temperature,
            model_max_tokens=ctx.model.max_tokens,
        )
    
    def _extract_null_baseline(self, ctx: ExtractionContext) -> DocumentResult:
        """Handle null baseline extraction (no API call).
        
        Null baseline returns empty extractions for all fields,
        useful for testing scoring logic.
        """
        kernel = ctx.kernel
        if not isinstance(kernel, NullBaseline):
            raise ValueError("Expected NullBaseline kernel")
        
        extractions = kernel.extract_directly()
        fields: list[FieldResult] = []
        
        # Provide defaults for optional document fields
        doc_text = ctx.doc.document.text or ""
        doc_pages = ctx.doc.document.pages or []
        
        for ext in extractions:
            gt = ctx.doc.ground_truth.get(ext.field_name)
            if not gt:
                continue
            
            score = composite_score(ext, gt, doc_text, doc_pages)
            
            fields.append(FieldResult(
                field_name=ext.field_name,
                extracted_value=ext.value,
                expected_value=gt.value,
                quote=ext.evidence.quote,
                page=ext.evidence.page,
                status=ext.status.value,
                value_score=score.value,
                evidence_score=score.evidence,
                page_score=score.page,
                status_score=score.status,
                schema_score=score.schema,
                composite_score=score.composite,
            ))
        
        return self._build_result(ctx, fields)
    
    def _extract_single(
        self,
        doc: DocumentWithGroundTruth,
        model: ModelConfig,
        kernel: Kernel,
    ) -> DocumentResult:
        """Extract from a single document.
        
        Returns raw response only - scoring happens in judge pass.
        """
        ctx = self._prepare_context(doc, model, kernel)
        
        # Null baseline: no API call needed
        if isinstance(kernel, NullBaseline):
            return self._extract_null_baseline(ctx)
        
        # Validate document has content
        if not doc.document.has_content:
            return self._build_result(
                ctx, 
                fields=[], 
                error="Document has no content (no text or file)",
            )
        
        # Render prompt (now accepts Document, handles text vs file mode)
        prompt = kernel.render(doc.document)
        
        # Apply cache-bust prefix to user message
        salted_prompt = RenderedPrompt(
            system=prompt.system,
            user=ctx.cache_prefix_char + prompt.user,
            kernel_name=prompt.kernel_name,
            document=prompt.document,
        )
        
        # Make API call
        client = self._get_client(model)
        start = time.time()
        
        try:
            result = client(salted_prompt)
        except Exception as e:
            return self._build_result(ctx, fields=[], error=str(e))
        
        latency_ms = (time.time() - start) * 1000
        
        return self._build_result(
            ctx,
            fields=[],
            raw_response=result.text,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    # =========================================================================
    # Pass 2: Judging
    # =========================================================================

    def _judge_single(
        self,
        doc: DocumentWithGroundTruth,
        result: DocumentResult,
        judge_model: ModelConfig,
    ) -> JudgeResult:
        """Judge a single extraction. Returns scored fields."""

        if not result.raw_response:
            return JudgeResult(fields=[], raw_response="", outcome="ERROR: no raw response")

        judge_prompt_path = self._data_dir.parent.parent / "prompts" / "judge.txt"

        if not doc.document.text:
            raise ValueError(f"Document {doc.document.id} has no text content for judging")
        
        field_results = judge_extraction(
            judge_model=judge_model,
            judge_prompt_path=judge_prompt_path,
            document_text=doc.document.text,
            ground_truth=doc.ground_truth,
            raw_response=result.raw_response,
        )

        return field_results

    # =========================================================================
    # Orchestration
    # =========================================================================

    def run(self) -> RunResults:
        """Execute the full evaluation: extract then judge."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._results = RunResults(
            run_id=run_id,
            config_name=self._config.name,
            started_at=datetime.now(),
        )

        self._log(f"Loading documents from {self._data_dir}...")
        documents = load_document_set(self._data_dir)
        self._log(f"  Loaded {len(documents)} documents")

        # Build lookup for documents by id
        docs_by_id = {doc.document.id: doc for doc in documents}

        matrix = self._config.run_matrix()
        total_runs = len(matrix) * len(documents)

        # =====================================================================
        # Pass 1: Extract
        # =====================================================================
        self._log(f"\n{'='*60}")
        self._log("PASS 1: EXTRACTION")
        self._log(f"{'='*60}")
        self._log(f"Run matrix: {len(matrix)} model×kernel pairs × {len(documents)} docs = {total_runs} extractions")

        completed = 0
        for model, kernel_config in matrix:
            kernel = self._get_kernel(kernel_config.name, kernel_config.path)

            for doc in documents:
                self._log(f"  [{completed + 1}/{total_runs}] {model.name} × {kernel.name} × {doc.document.id}")

                result = self._extract_single(doc, model, kernel)
                self._results.add(result)

                if result.error:
                    self._log(f"    ERROR: {result.error}")
                else:
                    self._log(f"    OK ({result.input_tokens} in, {result.output_tokens} out)")

                completed += 1

                if self._config.execution.delay > 0:
                    time.sleep(self._config.execution.delay)

        # =====================================================================
        # Pass 2: Judge
        # =====================================================================
        self._log(f"\n{'='*60}")
        judge_names = [j.name for j in self._config.judge_models] if self._config.judge_models else ["none"]
        self._log(f"PASS 2: JUDGING (policy={self._config.judge_policy}, judges={judge_names})")
        self._log(f"{'='*60}")

        results_to_judge = [r for r in self._results.results.values() if r.raw_response and not r.error]
        self._log(f"Judging {len(results_to_judge)} extractions...")

        judged = 0
        for result in results_to_judge:
            doc = docs_by_id.get(result.doc_id)
            if not doc:
                continue

            # Get judge for this model
            model_config = next(
                (m for m in self._config.models if m.name == result.model),
                None
            )
            if not model_config:
                continue

            judges = self._get_judges(model_config)
            if not judges:
                self._log(f"  [{judged + 1}/{len(results_to_judge)}] {result.model} × {result.kernel} × {result.doc_id}: NO JUDGE")
                judged += 1
                continue

            self._log(f"  [{judged + 1}/{len(results_to_judge)}] {result.model} × {result.kernel} × {result.doc_id}")

            try:
                result.fields = []
                result.judge_raw_response = None
                result.judge_outcome = "PENDING"
                
                judge_result = self._judge_single(doc, result, judges[0])
                
                result.fields = judge_result.fields
                result.judge_raw_response = judge_result.raw_response
                result.judge_outcome = judge_result.outcome

                # Add judge reproducibility info
                result.judge_model = judges[0].name
                result.judge_temperature = judges[0].temperature
                result.judge_max_tokens = judges[0].max_tokens
                result.judge_config_hash = hash_config(judges[0])
            
                judge_prompt_path = self._data_dir.parent.parent / "prompts" / "judge_system.txt"
                if judge_prompt_path.exists():
                    result.judge_prompt_hash = md5_hash(judge_prompt_path.read_text())
                    
                if judge_result.outcome == "OK":
                    self._log(f"    Composite: {result.composite_score:.2f}")
                elif judge_result.outcome.startswith("ERROR:"):
                    self._log(f"    {judge_result.outcome}")
                else:
                    raise ValueError(f"Unexpected judge outcome: {judge_result.outcome}")
                    
            except Exception as e:
                existing = result.judge_outcome or "PENDING"
                result.judge_outcome = f"ERROR: {existing} | {e}"
                self._log(f"    {result.judge_outcome}")

            judged += 1

            if self._config.execution.delay > 0:
                time.sleep(self._config.execution.delay)

        # =====================================================================
        # Finalize
        # =====================================================================
        self._results.completed_at = datetime.now()

        self._log(f"\nSaving results to {self._output_dir}...")
        self._results.save(self._output_dir / f"run_{run_id}")

        self._log("\n" + "=" * 60)
        self._log("SUMMARY")
        self._log("=" * 60)
        summary = self._results.summary()
        for key, stats in summary.get("by_model_kernel", {}).items():
            model, kernel = key.split("|")
            self._log(f"{model} × {kernel}:")
            self._log(f"  Composite: {stats['composite_mean']:.3f} (min={stats['composite_min']:.2f}, max={stats['composite_max']:.2f})")
            self._log(f"  Value:     {stats['value_mean']:.3f}")
            if stats['errors'] > 0:
                self._log(f"  Errors:    {stats['errors']}")

        if summary.get("comparisons"):
            self._log("\n" + "-" * 40)
            self._log("COMPARISONS (CTN vs Idiomatic):")
            for comp_key, comp in summary["comparisons"].items():
                model = comp_key.split("|")[0]
                self._log(f"  {model}:")
                self._log(f"    Delta = {comp['mean_diff']:+.3f} ({comp['effect_interpretation']} effect)")
                self._log(f"    p = {comp['p_value']:.3f}, n = {comp['n']}")

        return self._results


def run_evaluation(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    schema: DocumentSchema,
    verbose: bool = True,
) -> RunResults:
    """Convenience function to run an evaluation."""
    config = EvaluationConfig.from_yaml(config_path)

    run_config = RunConfig(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        schema=schema,
        verbose=verbose,
    )

    runner = Runner(run_config)
    return runner.run()
