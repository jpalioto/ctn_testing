"""Tests for data persistence integrity.

These tests ensure that what we save is what we can load back.
Data integrity is the foundation - if we can't trust disk, nothing else matters.
"""
import json
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

from ctn_testing.runners.results import (
    FieldResult,
    DocumentResult,
    RunResults,
)


class TestFieldResultPersistence:
    """Test FieldResult serialization."""
    
    def test_to_dict_all_fields_present(self):
        """Every dataclass field must appear in dict output."""
        field = FieldResult(
            field_name="invoice_number",
            extracted_value="INV-001",
            expected_value="INV-001",
            quote="Invoice: INV-001",
            page=1,
            status="ok",
            value_score=1.0,
            evidence_score=1.0,
            page_score=1.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=1.0,
            judge_scores={"accuracy": 0.95},
        )
        
        # FieldResult doesn't have to_dict, it's serialized by DocumentResult
        # Test via DocumentResult
        doc = DocumentResult(
            doc_id="test",
            model="test-model",
            kernel="test-kernel",
            fields=[field],
        )
        
        d = doc.to_dict()
        f = d["fields"][0]
        
        # Check all values present
        assert f["field"] == "invoice_number"
        assert f["extracted"] == "INV-001"
        assert f["expected"] == "INV-001"
        assert f["quote"] == "Invoice: INV-001"
        assert f["page"] == 1
        assert f["status"] == "ok"
        assert f["scores"]["value"] == 1.0
        assert f["scores"]["evidence"] == 1.0
        assert f["scores"]["page"] == 1.0
        assert f["scores"]["status"] == 1.0
        assert f["scores"]["schema"] == 1.0
        assert f["scores"]["composite"] == 1.0
        assert f["judge_scores"] == {"accuracy": 0.95}
    
    def test_field_result_from_dict_roundtrip(self):
        """FieldResult.from_dict must reconstruct equivalent object."""
        original = FieldResult(
            field_name="amount",
            extracted_value=123.45,
            expected_value=123.45,
            quote="Total: $123.45",
            page=2,
            status="ok",
            value_score=1.0,
            evidence_score=0.9,
            page_score=1.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=0.97,
            judge_scores={"confidence": 0.88},
        )
        
        # Simulate what DocumentResult.to_dict produces for a field
        field_dict = {
            "field": original.field_name,
            "extracted": original.extracted_value,
            "expected": original.expected_value,
            "quote": original.quote,
            "page": original.page,
            "status": original.status,
            "scores": {
                "value": original.value_score,
                "evidence": original.evidence_score,
                "page": original.page_score,
                "status": original.status_score,
                "schema": original.schema_score,
                "composite": original.composite_score,
            },
            "judge_scores": original.judge_scores,
        }
        
        restored = FieldResult.from_dict(field_dict)
        
        assert restored.field_name == original.field_name
        assert restored.extracted_value == original.extracted_value
        assert restored.expected_value == original.expected_value
        assert restored.quote == original.quote
        assert restored.page == original.page
        assert restored.status == original.status
        assert restored.value_score == original.value_score
        assert restored.evidence_score == original.evidence_score
        assert restored.page_score == original.page_score
        assert restored.status_score == original.status_score
        assert restored.schema_score == original.schema_score
        assert restored.composite_score == original.composite_score
        assert restored.judge_scores == original.judge_scores
    
    def test_null_values_preserved(self):
        """Null values must serialize as null, not be dropped."""
        field = FieldResult(
            field_name="due_date",
            extracted_value=None,
            expected_value=None,
            quote=None,
            page=None,
            status="missing",
            value_score=1.0,
            evidence_score=0.0,
            page_score=0.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=0.65,
        )
        
        doc = DocumentResult(
            doc_id="test",
            model="test-model",
            kernel="test-kernel",
            fields=[field],
        )
        
        d = doc.to_dict()
        f = d["fields"][0]
        
        # Nulls must be explicitly present
        assert "extracted" in f and f["extracted"] is None
        assert "expected" in f and f["expected"] is None
        assert "quote" in f and f["quote"] is None
        assert "page" in f and f["page"] is None
    
    def test_unicode_preserved(self):
        """Unicode strings must survive serialization."""
        field = FieldResult(
            field_name="vendor_name",
            extracted_value="株式会社テスト",
            expected_value="株式会社テスト",
            quote="会社: 株式会社テスト",
            page=1,
            status="ok",
            value_score=1.0,
            evidence_score=1.0,
            page_score=1.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=1.0,
        )
        
        doc = DocumentResult(
            doc_id="test",
            model="test-model",
            kernel="test-kernel",
            fields=[field],
        )
        
        d = doc.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert loaded["fields"][0]["extracted"] == "株式会社テスト"
        assert loaded["fields"][0]["quote"] == "会社: 株式会社テスト"


class TestDocumentResultPersistence:
    """Test DocumentResult serialization."""
    
    def test_to_dict_all_fields_present(self):
        """Every dataclass field must appear in dict output."""
        ts = datetime(2025, 12, 14, 12, 30, 45)
        
        doc = DocumentResult(
            doc_id="invoice_001",
            model="claude-sonnet-4-5",
            kernel="ctn",
            fields=[],
            raw_response="```json\n{}\n```",
            judge_raw_response="```json\n[]\n```",
            judge_outcome="OK",
            timestamp=ts,
            latency_ms=1234.56,
            input_tokens=100,
            output_tokens=200,
            error=None,
            cache_prefix="ZWSP",
            # Reproducibility - hashes
            document_hash="abc123",
            kernel_hash="def456",
            gt_hash="ghi789",
            judge_prompt_hash="jkl012",
            model_config_hash="mno345",
            judge_config_hash="pqr678",
            # Reproducibility - explicit params
            model_temperature=0.0,
            model_max_tokens=2048,
            judge_model="claude-haiku-4-5",
            judge_temperature=0.3,
            judge_max_tokens=512,
            # Debug
            api_request_id="req_abc123",
        )
        
        d = doc.to_dict()
        
        # Identity
        assert d["doc_id"] == "invoice_001"
        assert d["model"] == "claude-sonnet-4-5"
        assert d["kernel"] == "ctn"
        
        # Raw outputs
        assert d["raw_response"] == "```json\n{}\n```"
        assert d["judge_raw_response"] == "```json\n[]\n```"
        assert d["judge_outcome"] == "OK"
        
        # Timing
        assert d["timestamp"] == "2025-12-14T12:30:45"
        assert d["latency_ms"] == 1234.56
        
        # Tokens
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 200
        
        # Errors
        assert d["error"] is None
        assert d["cache_prefix"] == "ZWSP"
        
        # Reproducibility - hashes
        assert d["document_hash"] == "abc123"
        assert d["kernel_hash"] == "def456"
        assert d["gt_hash"] == "ghi789"
        assert d["judge_prompt_hash"] == "jkl012"
        assert d["model_config_hash"] == "mno345"
        assert d["judge_config_hash"] == "pqr678"
        
        # Reproducibility - explicit params
        assert d["model_temperature"] == 0.0
        assert d["model_max_tokens"] == 2048
        assert d["judge_model"] == "claude-haiku-4-5"
        assert d["judge_temperature"] == 0.3
        assert d["judge_max_tokens"] == 512
        
        # Debug
        assert d["api_request_id"] == "req_abc123"
        
        # Computed
        assert d["composite_score"] == 0.0
        assert d["value_score"] == 0.0
        assert d["fields"] == []
    
    def test_error_state_preserved(self):
        """Error states must serialize correctly."""
        doc = DocumentResult(
            doc_id="invoice_001",
            model="claude-sonnet-4-5",
            kernel="ctn",
            fields=[],
            error="API timeout after 60s",
            judge_outcome="ERROR: PENDING | API timeout",
        )
        
        d = doc.to_dict()
        
        assert d["error"] == "API timeout after 60s"
        assert d["judge_outcome"] == "ERROR: PENDING | API timeout"
    
    def test_json_roundtrip(self):
        """JSON serialize -> deserialize must preserve all data."""
        field = FieldResult(
            field_name="amount",
            extracted_value=1234.56,
            expected_value=1234.56,
            quote="Total: $1,234.56",
            page=1,
            status="ok",
            value_score=1.0,
            evidence_score=1.0,
            page_score=1.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=1.0,
        )
        
        doc = DocumentResult(
            doc_id="invoice_001",
            model="claude-sonnet-4-5",
            kernel="ctn",
            fields=[field],
            raw_response="test response",
            judge_raw_response="test judge",
            judge_outcome="OK",
            timestamp=datetime(2025, 12, 14, 12, 0, 0),
            latency_ms=1234.56,
            input_tokens=100,
            output_tokens=200,
            document_hash="hash123",
            model_temperature=0.0,
        )
        
        # Serialize and deserialize
        json_str = json.dumps(doc.to_dict())
        loaded = json.loads(json_str)
        
        # Verify all data
        assert loaded["doc_id"] == "invoice_001"
        assert loaded["latency_ms"] == 1234.56
        assert loaded["timestamp"] == "2025-12-14T12:00:00"
        assert loaded["document_hash"] == "hash123"
        assert loaded["model_temperature"] == 0.0
        assert loaded["fields"][0]["extracted"] == 1234.56
        assert loaded["fields"][0]["scores"]["composite"] == 1.0
    
    def test_from_dict_roundtrip(self):
        """to_dict -> from_dict must reconstruct equivalent object."""
        ts = datetime(2025, 12, 14, 12, 30, 45)
        
        original = DocumentResult(
            doc_id="invoice_001",
            model="claude-sonnet-4-5",
            kernel="ctn",
            fields=[
                FieldResult(
                    field_name="amount",
                    extracted_value=100.0,
                    expected_value=100.0,
                    quote="Amount: $100",
                    page=1,
                    status="ok",
                    value_score=1.0,
                    evidence_score=1.0,
                    page_score=1.0,
                    status_score=1.0,
                    schema_score=1.0,
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
            # Hashes
            document_hash="doc_abc",
            kernel_hash="kern_def",
            gt_hash="gt_ghi",
            judge_prompt_hash="jp_jkl",
            model_config_hash="mc_mno",
            judge_config_hash="jc_pqr",
            # Params
            model_temperature=0.0,
            model_max_tokens=2048,
            judge_model="claude-haiku-4-5",
            judge_temperature=0.3,
            judge_max_tokens=512,
            # Debug
            api_request_id="req_xyz",
        )
        
        d = original.to_dict()
        restored = DocumentResult.from_dict(d)
        
        # Identity
        assert restored.doc_id == original.doc_id
        assert restored.model == original.model
        assert restored.kernel == original.kernel
        
        # Raw outputs
        assert restored.raw_response == original.raw_response
        assert restored.judge_raw_response == original.judge_raw_response
        assert restored.judge_outcome == original.judge_outcome
        
        # Timing
        assert restored.timestamp == original.timestamp
        assert restored.latency_ms == original.latency_ms
        
        # Tokens
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        
        # Errors
        assert restored.error == original.error
        assert restored.cache_prefix == original.cache_prefix
        
        # Hashes
        assert restored.document_hash == original.document_hash
        assert restored.kernel_hash == original.kernel_hash
        assert restored.gt_hash == original.gt_hash
        assert restored.judge_prompt_hash == original.judge_prompt_hash
        assert restored.model_config_hash == original.model_config_hash
        assert restored.judge_config_hash == original.judge_config_hash
        
        # Params
        assert restored.model_temperature == original.model_temperature
        assert restored.model_max_tokens == original.model_max_tokens
        assert restored.judge_model == original.judge_model
        assert restored.judge_temperature == original.judge_temperature
        assert restored.judge_max_tokens == original.judge_max_tokens
        
        # Debug
        assert restored.api_request_id == original.api_request_id
        
        # Fields
        assert len(restored.fields) == len(original.fields)
        assert restored.fields[0].field_name == original.fields[0].field_name
        assert restored.fields[0].extracted_value == original.fields[0].extracted_value
    
    def test_from_dict_handles_missing_optional_fields(self):
        """from_dict must handle older data without new fields."""
        # Simulate old data format without new fields
        old_format = {
            "doc_id": "test",
            "model": "test-model",
            "kernel": "test-kernel",
            "composite_score": 0.0,
            "value_score": 0.0,
            "latency_ms": 1000.0,
            "input_tokens": 100,
            "output_tokens": 200,
            "raw_response": "test",
            "fields": [],
        }
        
        restored = DocumentResult.from_dict(old_format)
        
        assert restored.doc_id == "test"
        assert restored.document_hash is None
        assert restored.kernel_hash is None
        assert restored.model_temperature is None
        assert restored.api_request_id is None


class TestRunResultsPersistence:
    """Test RunResults save/load."""
    
    def test_save_creates_expected_structure(self):
        """Save must create summary.json and raw/{model}/{kernel}/{doc}.json."""
        results = RunResults(
            run_id="test_run",
            config_name="test_config",
            started_at=datetime(2025, 12, 14, 12, 0, 0),
            completed_at=datetime(2025, 12, 14, 12, 5, 0),
        )
        
        results.add(DocumentResult(
            doc_id="doc_001",
            model="model_a",
            kernel="kernel_x",
            fields=[],
        ))
        
        results.add(DocumentResult(
            doc_id="doc_001",
            model="model_a",
            kernel="kernel_y",
            fields=[],
        ))
        
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run_test"
            results.save(output_dir)
            
            # Check structure
            assert (output_dir / "summary.json").exists()
            assert (output_dir / "raw" / "model_a" / "kernel_x" / "doc_001.json").exists()
            assert (output_dir / "raw" / "model_a" / "kernel_y" / "doc_001.json").exists()
    
    def test_save_load_roundtrip(self):
        """Save -> Load must reconstruct equivalent results."""
        original = RunResults(
            run_id="test_run",
            config_name="test_config",
            started_at=datetime(2025, 12, 14, 12, 0, 0),
            completed_at=datetime(2025, 12, 14, 12, 5, 0),
        )
        
        field = FieldResult(
            field_name="test_field",
            extracted_value="extracted",
            expected_value="expected",
            quote="quote",
            page=1,
            status="ok",
            value_score=0.9,
            evidence_score=0.8,
            page_score=1.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=0.92,
        )
        
        original.add(DocumentResult(
            doc_id="doc_001",
            model="model_a",
            kernel="kernel_x",
            fields=[field],
            raw_response="raw response",
            judge_raw_response="judge response",
            judge_outcome="OK",
            timestamp=datetime(2025, 12, 14, 12, 1, 0),
            latency_ms=1234.5,
            input_tokens=100,
            output_tokens=200,
            document_hash="doc_hash",
            kernel_hash="kern_hash",
            model_temperature=0.0,
            model_max_tokens=2048,
            judge_model="judge-model",
        ))
        
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run_test"
            original.save(output_dir)
            
            # Load back
            loaded = RunResults.load(output_dir)
            
            assert loaded.run_id == original.run_id
            assert loaded.config_name == original.config_name
            assert len(loaded.results) == len(original.results)
            
            doc = loaded.get("model_a", "kernel_x", "doc_001")
            assert doc is not None
            assert doc.raw_response == "raw response"
            assert doc.judge_outcome == "OK"
            assert doc.document_hash == "doc_hash"
            assert doc.kernel_hash == "kern_hash"
            assert doc.model_temperature == 0.0
            assert doc.judge_model == "judge-model"
            assert len(doc.fields) == 1
            assert doc.fields[0].field_name == "test_field"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_string_vs_null(self):
        """Empty string must not become null."""
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            raw_response="",
            error="",
        )
        
        d = doc.to_dict()
        
        assert d["raw_response"] == ""
        assert d["error"] == ""
    
    def test_very_long_raw_response(self):
        """Very long strings must not be truncated."""
        long_response = "x" * 100000
        
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            raw_response=long_response,
        )
        
        d = doc.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert len(loaded["raw_response"]) == 100000
    
    def test_special_characters_in_strings(self):
        """Special characters must survive serialization."""
        special = 'Quote: "test"\nNewline\tTab\\Backslash'
        
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            raw_response=special,
        )
        
        d = doc.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert loaded["raw_response"] == special
    
    def test_numeric_precision(self):
        """Float precision must be maintained."""
        field = FieldResult(
            field_name="test",
            extracted_value=0.123456789,
            expected_value=0.123456789,
            quote=None,
            page=None,
            status="ok",
            value_score=0.333333333,
            evidence_score=0.666666666,
            page_score=0.999999999,
            status_score=0.111111111,
            schema_score=0.222222222,
            composite_score=0.444444444,
        )
        
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[field],
        )
        
        d = doc.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        # JSON preserves float precision
        assert abs(loaded["fields"][0]["extracted"] - 0.123456789) < 1e-9
        assert abs(loaded["fields"][0]["scores"]["value"] - 0.333333333) < 1e-9
    
    def test_nested_extracted_value(self):
        """Extracted values can be dicts/lists, must serialize correctly."""
        field = FieldResult(
            field_name="line_items",
            extracted_value=[
                {"item": "Widget", "qty": 10, "price": 9.99},
                {"item": "Gadget", "qty": 5, "price": 19.99},
            ],
            expected_value=[
                {"item": "Widget", "qty": 10, "price": 9.99},
                {"item": "Gadget", "qty": 5, "price": 19.99},
            ],
            quote=None,
            page=None,
            status="ok",
            value_score=1.0,
            evidence_score=1.0,
            page_score=1.0,
            status_score=1.0,
            schema_score=1.0,
            composite_score=1.0,
        )
        
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[field],
        )
        
        d = doc.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert loaded["fields"][0]["extracted"][0]["item"] == "Widget"
        assert loaded["fields"][0]["extracted"][1]["price"] == 19.99
    
    def test_hash_fields_roundtrip(self):
        """All hash fields must survive roundtrip."""
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            document_hash="d41d8cd98f00b204e9800998ecf8427e",
            kernel_hash="098f6bcd4621d373cade4e832627b4f6",
            gt_hash="5d41402abc4b2a76b9719d911017c592",
            judge_prompt_hash="7d793037a0760186574b0282f2f435e7",
            model_config_hash="e99a18c428cb38d5f260853678922e03",
            judge_config_hash="ad0234829205b9033196ba818f7a872b",
        )
        
        d = doc.to_dict()
        restored = DocumentResult.from_dict(d)
        
        assert restored.document_hash == "d41d8cd98f00b204e9800998ecf8427e"
        assert restored.kernel_hash == "098f6bcd4621d373cade4e832627b4f6"
        assert restored.gt_hash == "5d41402abc4b2a76b9719d911017c592"
        assert restored.judge_prompt_hash == "7d793037a0760186574b0282f2f435e7"
        assert restored.model_config_hash == "e99a18c428cb38d5f260853678922e03"
        assert restored.judge_config_hash == "ad0234829205b9033196ba818f7a872b"
    
    def test_timestamp_roundtrip(self):
        """Timestamp must survive roundtrip with correct format."""
        ts = datetime(2025, 12, 14, 12, 30, 45, 123456)
        
        doc = DocumentResult(
            doc_id="test",
            model="test",
            kernel="test",
            fields=[],
            timestamp=ts,
        )
        
        d = doc.to_dict()
        restored = DocumentResult.from_dict(d)
        
        # Note: microseconds may be truncated depending on isoformat
        assert restored.timestamp is not None
        assert restored.timestamp.year == 2025
        assert restored.timestamp.month == 12
        assert restored.timestamp.day == 14
        assert restored.timestamp.hour == 12
        assert restored.timestamp.minute == 30
        assert restored.timestamp.second == 45
