"""Tests for result persistence - ensuring no data loss through serialization."""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ctn_testing.runners.results import FieldResult, DocumentResult, RunResults


def make_field_result(
    field_name: str = "test_field",
    extracted_value: Any = "extracted",
    expected_value: Any = "expected",
    composite_score: float = 1.0,
    scores: dict[str, float] | None = None,
) -> FieldResult:
    """Helper to create FieldResult with new schema."""
    if scores is None:
        scores = {"exact": 1.0, "semantic": 1.0, "usable": 1.0, "complete": 1.0}
    return FieldResult(
        field_name=field_name,
        extracted_value=extracted_value,
        expected_value=expected_value,
        composite_score=composite_score,
        scores=scores,
    )


class TestFieldResultPersistence:
    """Tests for FieldResult serialization."""
    
    def test_to_dict_all_fields_present(self):
        """Every dataclass field must appear in dict output."""
        field = make_field_result(
            field_name="invoice_number",
            extracted_value="INV-001",
            expected_value="INV-001",
            composite_score=0.95,
            scores={"exact": 1.0, "semantic": 1.0, "usable": 1.0, "complete": 0.8},
        )
        d = field.to_dict()
        
        assert d["field"] == "invoice_number"
        assert d["extracted"] == "INV-001"
        assert d["expected"] == "INV-001"
        assert d["composite"] == 0.95
        assert d["scores"]["exact"] == 1.0
        assert d["scores"]["semantic"] == 1.0
        assert d["scores"]["usable"] == 1.0
        assert d["scores"]["complete"] == 0.8
    
    def test_field_result_from_dict_roundtrip(self):
        """FieldResult.from_dict must reconstruct equivalent object."""
        original = make_field_result(
            field_name="amount",
            extracted_value=123.45,
            expected_value=123.45,
            composite_score=0.97,
            scores={"exact": 1.0, "semantic": 1.0, "usable": 0.9, "complete": 1.0},
        )
        
        d = original.to_dict()
        restored = FieldResult.from_dict(d)
        
        assert restored.field_name == original.field_name
        assert restored.extracted_value == original.extracted_value
        assert restored.expected_value == original.expected_value
        assert restored.composite_score == original.composite_score
        assert restored.scores == original.scores
    
    def test_null_values_preserved(self):
        """Null values must serialize as null, not be dropped."""
        field = make_field_result(
            field_name="due_date",
            extracted_value=None,
            expected_value=None,
            composite_score=0.0,
            scores={"exact": 0.0, "semantic": 0.0, "usable": 0.0, "complete": 0.0},
        )
        
        d = field.to_dict()
        assert d["extracted"] is None
        assert d["expected"] is None
        
        # JSON roundtrip
        json_str = json.dumps(d)
        restored_d = json.loads(json_str)
        assert restored_d["extracted"] is None
        assert restored_d["expected"] is None
    
    def test_unicode_preserved(self):
        """Unicode strings must survive serialization."""
        field = make_field_result(
            field_name="vendor_name",
            extracted_value="株式会社テスト",
            expected_value="株式会社テスト",
            composite_score=1.0,
            scores={"exact": 1.0, "semantic": 1.0, "usable": 1.0, "complete": 1.0},
        )
        
        d = field.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        restored_d = json.loads(json_str)
        restored = FieldResult.from_dict(restored_d)
        
        assert restored.extracted_value == "株式会社テスト"
        assert restored.expected_value == "株式会社テスト"


class TestDocumentResultPersistence:
    """Tests for DocumentResult serialization."""
    
    def test_to_dict_all_fields_present(self):
        """Every dataclass field must appear in dict output."""
        result = DocumentResult(
            doc_id="test_doc",
            model="test_model",
            kernel="test_kernel",
            fields=[],
            raw_response="test response",
            judge_raw_response="judge response",
            judge_outcome="OK",
            timestamp=datetime(2025, 12, 14, 12, 0, 0),
            latency_ms=1234.5,
            input_tokens=100,
            output_tokens=50,
            error=None,
            cache_prefix="ZWSP",
            document_hash="abc123",
            kernel_hash="def456",
            gt_hash="ghi789",
            judge_prompt_hash="jkl012",
            model_config_hash="mno345",
            judge_config_hash="pqr678",
            model_temperature=0.0,
            model_max_tokens=2048,
            judge_model="claude-haiku-4-5",
            judge_temperature=0.3,
            judge_max_tokens=512,
            api_request_id="req_123",
        )
        
        d = result.to_dict()
        
        # Check all keys present
        assert "doc_id" in d
        assert "model" in d
        assert "kernel" in d
        assert "fields" in d
        assert "raw_response" in d
        assert "judge_raw_response" in d
        assert "judge_outcome" in d
        assert "timestamp" in d
        assert "latency_ms" in d
        assert "input_tokens" in d
        assert "output_tokens" in d
        assert "error" in d
        assert "cache_prefix" in d
        assert "document_hash" in d
        assert "kernel_hash" in d
        assert "gt_hash" in d
        assert "judge_prompt_hash" in d
        assert "model_config_hash" in d
        assert "judge_config_hash" in d
        assert "model_temperature" in d
        assert "model_max_tokens" in d
        assert "judge_model" in d
        assert "judge_temperature" in d
        assert "judge_max_tokens" in d
        assert "api_request_id" in d
    
    def test_error_state_preserved(self):
        """Error state must survive serialization."""
        result = DocumentResult(
            doc_id="test_doc",
            model="test_model",
            kernel="test_kernel",
            fields=[],
            error="API timeout after 30s",
        )
        
        d = result.to_dict()
        restored = DocumentResult.from_dict(d)
        
        assert restored.error == "API timeout after 30s"
    
    def test_json_roundtrip(self):
        """JSON serialize -> deserialize must preserve all data."""
        field = make_field_result(
            field_name="amount",
            extracted_value=1234.56,
            expected_value=1234.56,
            composite_score=1.0,
            scores={"exact": 1.0, "semantic": 1.0, "usable": 1.0, "complete": 1.0},
        )
        
        result = DocumentResult(
            doc_id="invoice_001",
            model="claude-sonnet-4-5",
            kernel="ctn",
            fields=[field],
            raw_response='{"amount": 1234.56}',
            timestamp=datetime(2025, 12, 14, 12, 30, 45),
            latency_ms=1500.0,
            input_tokens=100,
            output_tokens=50,
        )
        
        json_str = json.dumps(result.to_dict())
        restored_d = json.loads(json_str)
        restored = DocumentResult.from_dict(restored_d)
        
        assert restored.doc_id == "invoice_001"
        assert len(restored.fields) == 1
        assert restored.fields[0].extracted_value == 1234.56
    
    def test_from_dict_roundtrip(self):
        """to_dict -> from_dict must reconstruct equivalent object."""
        ts = datetime(2025, 12, 14, 12, 30, 45)

        original = DocumentResult(
            doc_id="invoice_001",
            model="claude-sonnet-4-5",
            kernel="ctn",
            fields=[
                make_field_result(
                    field_name="amount",
                    extracted_value=100.0,
                    expected_value=100.0,
                    composite_score=1.0,
                )
            ],
            raw_response="raw",
            judge_raw_response="judge",
            judge_outcome="OK",
            timestamp=ts,
            latency_ms=1000.0,
            input_tokens=50,
            output_tokens=100,
            error=None,
            cache_prefix="ZWSP",
            document_hash="doc_abc",
            kernel_hash="kern_def",
            gt_hash="gt_ghi",
            judge_prompt_hash="jp_jkl",
            model_config_hash="mc_mno",
            judge_config_hash="jc_pqr",
            model_temperature=0.0,
            model_max_tokens=2048,
            judge_model="claude-haiku-4-5",
            judge_temperature=0.3,
            judge_max_tokens=512,
            api_request_id="req_xyz",
        )

        d = original.to_dict()
        restored = DocumentResult.from_dict(d)

        assert restored.doc_id == original.doc_id
        assert restored.model == original.model
        assert restored.kernel == original.kernel
        assert restored.timestamp == original.timestamp
        assert restored.latency_ms == original.latency_ms
        assert restored.document_hash == original.document_hash
        assert restored.judge_model == original.judge_model
        assert len(restored.fields) == len(original.fields)
    
    def test_from_dict_handles_missing_optional_fields(self):
        """from_dict must handle missing optional fields gracefully."""
        minimal_dict = {
            "doc_id": "test",
            "model": "test_model",
            "kernel": "test_kernel",
            "fields": [],
        }
        
        result = DocumentResult.from_dict(minimal_dict)
        
        assert result.doc_id == "test"
        assert result.raw_response is None
        assert result.timestamp is None
        assert result.error is None


class TestRunResultsPersistence:
    """Tests for RunResults serialization."""
    
    def test_save_creates_expected_structure(self):
        """Save must create proper directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = RunResults(
                run_id="test_run",
                config_name="test_config",
                started_at=datetime(2025, 12, 14, 12, 0, 0),
            )

            run_dir = Path(tmpdir) / "run_test_run"
            results.save(run_dir)

            assert run_dir.exists()
            assert (run_dir / "summary.json").exists()

    def test_save_load_roundtrip(self):
        """Save -> Load must reconstruct equivalent results."""
        original = RunResults(
            run_id="test_run",
            config_name="test_config",
            started_at=datetime(2025, 12, 14, 12, 0, 0),
            completed_at=datetime(2025, 12, 14, 12, 5, 0),
        )

        field = make_field_result(
            field_name="test_field",
            extracted_value="extracted",
            expected_value="expected",
            composite_score=0.92,
            scores={"exact": 0.9, "semantic": 1.0, "usable": 0.8, "complete": 1.0},
        )

        doc_result = DocumentResult(
            doc_id="doc_001",
            model="test_model",
            kernel="test_kernel",
            fields=[field],
            raw_response="test response",
            timestamp=datetime(2025, 12, 14, 12, 1, 0),
        )

        original.add(doc_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_test_run"
            original.save(run_dir)

            loaded = RunResults.load(run_dir)

            assert loaded.run_id == original.run_id
            assert loaded.config_name == original.config_name
            assert len(loaded.results) == 1
            
            # results is dict keyed by (model, kernel, doc_id)
            key = ("test_model", "test_kernel", "doc_001")
            assert key in loaded.results
            assert loaded.results[key].doc_id == "doc_001"
            assert loaded.results[key].fields[0].composite_score == 0.92

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_string_vs_null(self):
        """Empty string must not be confused with null."""
        result = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            raw_response="",
            error="",
        )
        
        d = result.to_dict()
        assert d["raw_response"] == ""
        assert d["error"] == ""
        
        restored = DocumentResult.from_dict(d)
        assert restored.raw_response == ""
        assert restored.error == ""
    
    def test_very_long_raw_response(self):
        """Very long strings must not be truncated."""
        long_response = "x" * 100_000
        
        result = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            raw_response=long_response,
        )
        
        d = result.to_dict()
        json_str = json.dumps(d)
        restored_d = json.loads(json_str)
        restored = DocumentResult.from_dict(restored_d)
        
        assert len(restored.raw_response) == 100_000
    
    def test_special_characters_in_strings(self):
        """Special characters must be preserved."""
        special = 'Quote: "test"\nTab:\t\nBackslash: \\'
        
        result = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            raw_response=special,
        )
        
        json_str = json.dumps(result.to_dict())
        restored = DocumentResult.from_dict(json.loads(json_str))
        
        assert restored.raw_response == special
    
    def test_numeric_precision(self):
        """Float precision must be maintained."""
        field = make_field_result(
            field_name="test",
            extracted_value=0.123456789,
            expected_value=0.123456789,
            composite_score=0.444444444,
            scores={
                "exact": 0.333333333,
                "semantic": 0.666666666,
                "usable": 0.999999999,
                "complete": 0.111111111,
            },
        )
        
        d = field.to_dict()
        json_str = json.dumps(d)
        restored = FieldResult.from_dict(json.loads(json_str))
        
        assert abs(restored.extracted_value - 0.123456789) < 1e-9
        assert abs(restored.scores["exact"] - 0.333333333) < 1e-9
    
    def test_nested_extracted_value(self):
        """Extracted values can be dicts/lists, must serialize correctly."""
        field = make_field_result(
            field_name="line_items",
            extracted_value=[
                {"item": "Widget", "qty": 10, "price": 9.99},
                {"item": "Gadget", "qty": 5, "price": 19.99},
            ],
            expected_value=[
                {"item": "Widget", "qty": 10, "price": 9.99},
                {"item": "Gadget", "qty": 5, "price": 19.99},
            ],
            composite_score=1.0,
        )
        
        json_str = json.dumps(field.to_dict())
        restored = FieldResult.from_dict(json.loads(json_str))
        
        assert len(restored.extracted_value) == 2
        assert restored.extracted_value[0]["item"] == "Widget"
    
    def test_hash_fields_roundtrip(self):
        """Hash fields must survive roundtrip."""
        result = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            document_hash="a" * 32,
            kernel_hash="b" * 32,
            gt_hash="c" * 32,
            judge_prompt_hash="d" * 32,
            model_config_hash="e" * 32,
            judge_config_hash="f" * 32,
        )
        
        d = result.to_dict()
        restored = DocumentResult.from_dict(d)
        
        assert restored.document_hash == "a" * 32
        assert restored.kernel_hash == "b" * 32
        assert restored.gt_hash == "c" * 32
        assert restored.judge_prompt_hash == "d" * 32
        assert restored.model_config_hash == "e" * 32
        assert restored.judge_config_hash == "f" * 32
    
    def test_timestamp_roundtrip(self):
        """Timestamp must survive ISO format roundtrip."""
        ts = datetime(2025, 12, 14, 12, 30, 45, 123456)
        
        result = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            timestamp=ts,
        )
        
        d = result.to_dict()
        restored = DocumentResult.from_dict(d)
        
        assert restored.timestamp == ts


class TestGroundTruthPersistence:
    """Tests for ground truth persistence in results."""
    
    def test_ground_truth_roundtrip(self):
        """Ground truth survives serialization."""
        gt = {
            "invoice_number": {
                "field_name": "invoice_number",
                "value": "INV-001",
                "acceptable_values": ["INV-001", "INV001"],
                "candidate_values": [],
                "exists_in_document": True,
                "is_ambiguous": False,
                "evidence_quote": "Invoice #INV-001",
                "evidence_page": 0,
                "notes": None,
            },
            "total_amount": {
                "field_name": "total_amount",
                "value": "$1,234.56",
                "acceptable_values": [],
                "candidate_values": ["$1,234.56", "$1234.56"],
                "exists_in_document": True,
                "is_ambiguous": True,
                "evidence_quote": None,
                "evidence_page": None,
                "notes": "Multiple formats present",
            },
        }
        
        result = DocumentResult(
            doc_id="test_doc",
            model="test_model",
            kernel="test_kernel",
            fields=[],
            ground_truth=gt,
        )
        
        d = result.to_dict()
        restored = DocumentResult.from_dict(d)
        
        assert restored.ground_truth == gt
    
    def test_ground_truth_null_preserved(self):
        """Null ground truth preserved."""
        result = DocumentResult(
            doc_id="test_doc",
            model="test_model",
            kernel="test_kernel",
            fields=[],
            ground_truth=None,
        )
        
        d = result.to_dict()
        restored = DocumentResult.from_dict(d)
        
        assert restored.ground_truth is None
    
    def test_ground_truth_json_roundtrip(self):
        """Ground truth survives JSON serialization."""
        gt = {
            "field1": {
                "field_name": "field1",
                "value": "test value",
                "acceptable_values": [],
                "candidate_values": [],
                "exists_in_document": True,
                "is_ambiguous": False,
                "evidence_quote": "test value here",
                "evidence_page": 1,
                "notes": None,
            }
        }
        
        result = DocumentResult(
            doc_id="test_doc",
            model="test_model",
            kernel="test_kernel",
            fields=[],
            ground_truth=gt,
        )
        
        json_str = json.dumps(result.to_dict())
        restored = DocumentResult.from_dict(json.loads(json_str))
        
        assert restored.ground_truth == gt