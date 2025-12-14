"""Main evaluation runner."""
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core import (
    EvaluationConfig,
    ModelConfig,
    get_client,
    DocumentSchema,
)
from ..metrics import composite_score
from .document import DocumentWithGroundTruth, load_document_set
from .kernel import Kernel, NullBaseline, load_kernel, OUTPUT_FORMAT
from .results import RunResults, DocumentResult, FieldResult

# Zero-width characters for cache busting (invisible, deterministic)
# each test runs has one of these characters added to the top of the file
# the purpose of which is to invalidate any implicit chaching done by the model
# this was done in reposne to the observation that certain models seemed to 
# improve after our first run even though we were not explicitly caching.
# This is done for all models reagardless of implicit caching on the model
# side for uniformity. 
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


class Runner:
    """
    Evaluation runner.
    
    Executes: models × kernels × documents
    Scores each extraction against ground truth.
    Aggregates and saves results.
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
            self._clients[model.name] = get_client(model)
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
    
    def _evaluate_single(
        self,
        doc: DocumentWithGroundTruth,
        model: ModelConfig,
        kernel: Kernel,
    ) -> DocumentResult:
        """Evaluate a single document with a single model/kernel."""
        
        # Handle null baseline specially
        if isinstance(kernel, NullBaseline):
            extractions = kernel.extract_directly()
            field_results = []
            
            for ext in extractions:
                gt = doc.ground_truth.get(ext.field_name)
                if not gt:
                    continue
                
                score = composite_score(
                    ext, gt, doc.document.text, doc.document.pages
                )
                
                field_results.append(FieldResult(
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
            
            return DocumentResult(
                doc_id=doc.document.id,
                model=model.name,
                kernel=kernel.name,
                fields=field_results,
            )
        
        prompt = kernel.render(self._schema, doc.document.text, OUTPUT_FORMAT)

        # Cache-bust prefix (deterministic, logged)
        prefix_idx = self._test_index % len(ZW_CHARS)
        prefix_char = ZW_CHARS[prefix_idx]
        prefix_name = ZW_NAMES[prefix_idx]
        self._test_index += 1
        salted_prompt_text = prefix_char + prompt.text
        
        client = self._get_client(model)
        start_time = time.time()
        
        try:
            result = client.complete(salted_prompt_text)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            return DocumentResult(
                doc_id=doc.document.id,
                model=model.name,
                kernel=kernel.name,
                fields=[],
                error=str(e),
                cache_prefix=prefix_name,
            )
        
        try:
            extractions = kernel.parse_response(result.text)
        except Exception as e:
            return DocumentResult(
                doc_id=doc.document.id,
                model=model.name,
                kernel=kernel.name,
                fields=[],
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                error=f"Parse error: {e}",
                cache_prefix=prefix_name,
            )
        
        ext_by_field = {e.field_name: e for e in extractions}
        
        field_results = []
        for field_name, gt in doc.ground_truth.items():
            ext = ext_by_field.get(field_name)
            
            if ext is None:
                from ..core.types import Extraction, Evidence, ExtractionStatus
                ext = Extraction(
                    field_name=field_name,
                    value=None,
                    evidence=Evidence(quote=None, page=None),
                    status=ExtractionStatus.MISSING,
                    candidates=[],
                )
            
            score = composite_score(
                ext, gt, doc.document.text, doc.document.pages
            )
            
            field_results.append(FieldResult(
                field_name=field_name,
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
        
        return DocumentResult(
            doc_id=doc.document.id,
            model=model.name,
            kernel=kernel.name,
            fields=field_results,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_prefix=prefix_name,
        )
    
    def run(self) -> RunResults:
        """Execute the full evaluation."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._results = RunResults(
            run_id=run_id,
            config_name=self._config.name,
            started_at=datetime.now(),
        )
        
        self._log(f"Loading documents from {self._data_dir}...")
        documents = load_document_set(self._data_dir)
        self._log(f"  Loaded {len(documents)} documents")
        
        matrix = self._config.run_matrix()
        total_runs = len(matrix) * len(documents)
        self._log(f"Run matrix: {len(matrix)} model×kernel pairs × {len(documents)} docs = {total_runs} evaluations")
        
        completed = 0
        for model, kernel_config in matrix:
            kernel = self._get_kernel(kernel_config.name, kernel_config.path)
            
            for doc in documents:
                self._log(f"  [{completed + 1}/{total_runs}] {model.name} × {kernel.name} × {doc.document.id}")
                
                result = self._evaluate_single(doc, model, kernel)
                self._results.add(result)
                
                if result.error:
                    self._log(f"    ERROR: {result.error}")
                else:
                    self._log(f"    Composite: {result.composite_score:.2f}")
                
                completed += 1
                
                if self._config.execution.delay > 0:
                    time.sleep(self._config.execution.delay)
        
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
                self._log(f"    Δ = {comp['mean_diff']:+.3f} ({comp['effect_interpretation']} effect)")
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
