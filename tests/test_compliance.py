"""
Unit Tests — Governance & Compliance
───────────────────────────────────────────────────────────────────────────────
Tests compliance module for:
  - PII detection and masking
  - Audit logging (write, read, query, count)
  - Data snapshot creation and retrieval
  - Compliance report generation
  - GDPR erasure plan building
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compliance import (
    AuditLogger,
    AuditEntry,
    DatasSnapshot,
    PIICategory,
    SnapshotManager,
    build_erasure_plan,
    detect_pii_columns,
    generate_compliance_report,
    mask_pii_columns,
    DataSnapshot,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """DataFrame with no PII."""
    return pd.DataFrame({
        "record_id": range(100),
        "feature_a": np.random.randn(100),
        "feature_b": np.random.choice(["X", "Y", "Z"], 100),
        "score": np.random.uniform(0, 1, 100),
    })


@pytest.fixture
def pii_df():
    """DataFrame containing PII columns."""
    return pd.DataFrame({
        "record_id": range(5),
        "email": ["alice@example.com", "bob@test.org", "charlie@co.uk", "dave@io", "eve@site.net"],
        "ssn_column": ["123-45-6789", "987-65-4321", "111-22-3333", "444-55-6666", "777-88-9999"],
        "phone": ["212-555-0100", "310-555-0200", "415-555-0300", "617-555-0400", "312-555-0500"],
        "credit_card": ["4111-1111-1111-1111", "5500-0000-0000-0004", "3400-0000-0000-009", "3000-0000-0000-0004", "6011-0000-0000-0004"],
        "feature_score": np.random.randn(5),
    })


# ── PII Detection Tests ───────────────────────────────────────────────────────

class TestPIIDetection:
    """Test PII detection and masking."""

    def test_no_pii_in_clean_data(self, clean_df):
        """Clean data should have no PII detected."""
        result = detect_pii_columns(clean_df)
        assert len(result) == 0

    def test_pii_detected_by_column_name(self, pii_df):
        """Columns named 'email', 'ssn' etc should be detected by heuristic."""
        result = detect_pii_columns(pii_df)
        assert "email" in result
        assert "ssn_column" in result
        assert "phone" in result

    def test_pii_category_types(self, pii_df):
        """Detected PII should have correct categories."""
        result = detect_pii_columns(pii_df)
        assert PIICategory.EMAIL in result.get("email", [])
        assert PIICategory.PHONE in result.get("phone", [])

    def test_credit_card_detected_by_content(self, pii_df):
        """Credit card numbers should be detected by regex in content."""
        result = detect_pii_columns(pii_df)
        assert "credit_card" in result

    def test_mask_email(self, pii_df):
        """Email masking should hide local part, keep domain."""
        masked = mask_pii_columns(pii_df)
        for val in masked["email"]:
            assert "@" in str(val)
            local = str(val).split("@")[0]
            assert local == "*" * len(local) or local == "[REDACTED]"

    def test_mask_ssn(self, pii_df):
        """SSN masking should show only last 4."""
        masked = mask_pii_columns(pii_df, {"ssn_column": [PIICategory.SSN]})
        for val in masked["ssn_column"]:
            s = str(val)
            assert "***" in s or s == "[REDACTED]"

    def test_mask_preserves_structure(self, pii_df):
        """Masking should not change row count or types."""
        masked = mask_pii_columns(pii_df)
        assert len(masked) == len(pii_df)
        assert set(masked.columns) == set(pii_df.columns)


# ── Audit Logger Tests ────────────────────────────────────────────────────────

class TestAuditLogger:
    """Test audit logging."""

    def test_log_creates_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_path=str(Path(tmp) / "audit.jsonl"))
            entry = audit.log(
                operation="training",
                actor="test_user",
                model_id="M_test",
                records_affected=1000,
            )
            assert entry.operation == "training"
            assert entry.actor == "test_user"
            assert entry.model_id == "M_test"
            assert entry.records_affected == 1000
            assert entry.status == "success"

    def test_log_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "audit.jsonl")
            audit = AuditLogger(log_path=path)
            audit.log(operation="inference", actor="ci")
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["operation"] == "inference"

    def test_get_recent_returns_n_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_path=str(Path(tmp) / "audit.jsonl"))
            for i in range(10):
                audit.log(operation=f"op_{i}", actor="test")
            recent = audit.get_recent(n=3)
            assert len(recent) == 3

    def test_query_filters_by_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_path=str(Path(tmp) / "audit.jsonl"))
            audit.log(operation="training", actor="alice")
            audit.log(operation="inference", actor="bob")
            audit.log(operation="training", actor="carol")

            results = audit.query(operation="training")
            assert len(results) == 2

    def test_query_filters_by_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_path=str(Path(tmp) / "audit.jsonl"))
            audit.log(operation="training", actor="alice")
            audit.log(operation="inference", actor="alice")
            audit.log(operation="training", actor="bob")

            results = audit.query(actor="alice")
            assert len(results) == 2

    def test_count_by_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_path=str(Path(tmp) / "audit.jsonl"))
            audit.log(operation="training", actor="a")
            audit.log(operation="training", actor="b")
            audit.log(operation="inference", actor="c")

            counts = audit.count_by_operation()
            assert counts["training"] == 2
            assert counts["inference"] == 1


# ── Snapshot Manager Tests ────────────────────────────────────────────────────

class TestSnapshotManager:
    """Test data snapshot management."""

    def test_create_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SnapshotManager(snapshot_dir=str(Path(tmp) / "snapshots"))
            snapshot = manager.create_snapshot(
                dataset_hash="abc123def456",
                num_rows=1000,
                num_features=50,
                mlflow_run_id="run_001",
            )
            assert snapshot.dataset_hash == "abc123def456"
            assert snapshot.num_rows == 1000
            assert snapshot.num_features == 50
            assert snapshot.mlflow_run_id == "run_001"

    def test_snapshot_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SnapshotManager(snapshot_dir=str(Path(tmp) / "snapshots"))
            manager.create_snapshot(
                dataset_hash="xyz789",
                num_rows=500,
                num_features=20,
            )
            # List should return 1 snapshot
            snapshots = manager.list_snapshots()
            assert len(snapshots) == 1
            assert snapshots[0].dataset_hash == "xyz789"

    def test_get_snapshot_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SnapshotManager(snapshot_dir=str(Path(tmp) / "snapshots"))
            s1 = manager.create_snapshot(
                dataset_hash="hash_001",
                num_rows=100,
                num_features=10,
            )
            loaded = manager.get_snapshot(s1.snapshot_id)
            assert loaded is not None
            assert loaded.dataset_hash == "hash_001"

    def test_get_nonexistent_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SnapshotManager(snapshot_dir=str(Path(tmp) / "snapshots"))
            loaded = manager.get_snapshot("snap_nonexistent")
            assert loaded is None


# ── Compliance Report Tests ───────────────────────────────────────────────────

class TestComplianceReport:
    """Test compliance report generation."""

    def test_report_on_clean_data(self, clean_df):
        """Clean data should have no PII flagged."""
        report = generate_compliance_report(clean_df, dataset_name="clean_test")
        assert report["has_pii"] is False
        assert report["data_classification"] == "internal"
        assert report["dataset_shape"]["rows"] == 100

    def test_report_on_pii_data(self, pii_df):
        """Data with PII should flag it."""
        report = generate_compliance_report(pii_df, dataset_name="pii_test")
        assert report["has_pii"] is True
        assert report["data_classification"] == "pii"
        assert len(report["pii_scan"]) > 0

    def test_report_includes_column_metadata(self, clean_df):
        """Report should include per-column metadata."""
        report = generate_compliance_report(clean_df)
        assert "columns" in report
        assert "record_id" in report["columns"]
        assert report["columns"]["record_id"]["dtype"] is not None

    def test_report_includes_audit_entries(self, clean_df):
        """Report should include recent audit entries if logger provided."""
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_path=str(Path(tmp) / "audit.jsonl"))
            audit.log(operation="training", actor="tester")

            report = generate_compliance_report(
                clean_df,
                dataset_name="with_audit",
                audit_logger=audit,
            )
            assert len(report["audit_entries"]) > 0


# ── GDPR Erasure Tests ────────────────────────────────────────────────────────

class TestGDPRErasure:
    """Test GDPR erasure plan building."""

    def test_build_erasure_plan(self):
        """Erasure plan should list affected tables."""
        plan = build_erasure_plan(
            user_id="user_123",
            tables=["bronze.raw", "silver.features", "gold.predictions"],
        )
        assert plan.user_id == "user_123"
        assert len(plan.tables_to_modify) == 3
        assert plan.tables_to_modify[0]["table"] == "bronze.raw"

    def test_erasure_plan_includes_delete_statement(self):
        """Each table entry should have a DELETE statement."""
        plan = build_erasure_plan(
            user_id="user_456",
            tables=["silver.customers"],
        )
        stmt = plan.tables_to_modify[0]["statement"]
        assert "DELETE" in stmt
        assert "user_456" in stmt
