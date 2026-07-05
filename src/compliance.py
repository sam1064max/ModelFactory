"""
Governance & Compliance Module
───────────────────────────────────────────────────────────────────────────────
Implements audit logging, data versioning, PII detection, and GDPR/CCPA/HIPAA
compliance utilities for the MLOps platform.

Capabilities:
  - **Audit Trail**: Structured JSON logging of all pipeline operations with
    who-did-what-when for SOX compliance.
  - **Data Versioning**: SHA-256 hashing of training datasets for reproducibility.
    Snapshots record which Delta version, feature store version, and code commit
    produced each model.
  - **PII Detection**: Scan columns for potentially sensitive data (emails, SSNs,
    credit cards, phone numbers) and flag / redact / mask.
  - **GDPR Right to Erasure**: Generate Delta DELETE statements for a given user ID
    across all tables.
  - **Data Minimisation**: Feature-level tagging (pii_category, retention_days)
    to ensure only necessary data is stored.
  - **Access Control Stubs**: Structured declarations mirroring Unity Catalog RBAC.

In production (Databricks), Unity Catalog provides the underlying enforcement.
This module provides the policy definition and local/CI compliance validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils import logger


# ── Enums & Data Classes ─────────────────────────────────────────────────────

class PIICategory(str, Enum):
    """PII categories for data classification."""
    EMAIL = "email"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"
    NON_PII = "non_pii"


class DataClassification(str, Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"  # Protected Health Information (HIPAA)


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    operation: str            # e.g., "training", "inference", "data_access"
    actor: str                # user or service principal
    model_id: Optional[str] = None
    dataset_hash: Optional[str] = None
    feature_table: Optional[str] = None
    target_table: Optional[str] = None
    records_affected: int = 0
    status: str = "success"   # success | failure
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataSnapshot:
    """
    A reproducible snapshot of data used for training or inference.

    This links together:
      - The Delta Lake version (or Parquet hash)
      - The Feature Store table versions
      - The Git commit SHA
      - The MLflow run ID
      - The model config version
    """
    snapshot_id: str
    created_at: str
    dataset_hash: str
    delta_version: Optional[int] = None
    feature_table_versions: dict[str, str] = field(default_factory=dict)
    git_commit_sha: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    model_config_version: Optional[str] = None
    num_rows: int = 0
    num_features: int = 0
    classification: str = "internal"
    retention_days: int = 365


# ── PII Detection ─────────────────────────────────────────────────────────────

# Regex patterns for common PII
PII_PATTERNS: dict[PIICategory, list[re.Pattern]] = {
    PIICategory.EMAIL: [re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")],
    PIICategory.SSN: [re.compile(r"\b\d{3}-\d{2}-\d{4}\b")],
    PIICategory.CREDIT_CARD: [
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        re.compile(r"\b\d{16}\b"),
    ],
    PIICategory.PHONE: [
        re.compile(r"\b\+?1?\d{10}\b"),
        re.compile(r"\b\(\d{3}\)\s*\d{3}-\d{4}\b"),
    ],
    PIICategory.IP_ADDRESS: [re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")],
}

# Columns names that suggest PII content (heuristic)
PII_COLUMN_HINTS: dict[str, PIICategory] = {
    "email": PIICategory.EMAIL,
    "ssn": PIICategory.SSN,
    "phone": PIICategory.PHONE,
    "address": PIICategory.ADDRESS,
    "name": PIICategory.NAME,
    "ip_address": PIICategory.IP_ADDRESS,
}


def detect_pii_columns(df: pd.DataFrame) -> dict[str, list[PIICategory]]:
    """
    Scan a DataFrame for columns that may contain PII.

    Uses two methods:
      1. Column name heuristics (e.g., 'email' → EMAIL)
      2. Content regex matching on string columns

    Returns:
        dict mapping column_name → [list of detected PII categories]
    """
    results: dict[str, list[PIICategory]] = defaultdict(list)

    # Method 1: Column name hints
    for col in df.columns:
        col_lower = col.lower()
        for keyword, category in PII_COLUMN_HINTS.items():
            if keyword in col_lower:
                results[col].append(category)

    # Method 2: Content scanning (only on string columns)
    for col in df.select_dtypes(include=["object", "string"]):
        sample = df[col].dropna().astype(str).head(1000)
        for category, patterns in PII_PATTERNS.items():
            matches = sum(1 for val in sample if any(p.search(val) for p in patterns))
            if matches > len(sample) * 0.1:  # >10% match rate
                if category not in results[col]:
                    results[col].append(category)

    if results:
        for col, cats in results.items():
            logger.info(f"  PII detected: '{col}' → {[c.value for c in cats]}")
    else:
        logger.info("  No PII columns detected in scan")

    return dict(results)


def mask_pii_columns(
    df: pd.DataFrame,
    pii_map: Optional[dict[str, list[PIICategory]]] = None,
) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with PII columns masked.

    Masking strategies:
      - Email: keep domain, mask local part
      - SSN: show last 4 digits
      - Credit card: show last 4 digits
      - Phone: mask middle digits
      - Other: replace with '[REDACTED]'
    """
    if pii_map is None:
        pii_map = detect_pii_columns(df)

    result = df.copy()

    for col, categories in pii_map.items():
        if col not in result.columns:
            continue

        for cat in categories:
            if cat == PIICategory.EMAIL:
                result[col] = result[col].astype(str).apply(
                    lambda x: re.sub(
                        r"([a-zA-Z0-9._%+-]+)(@.*)",
                        lambda m: "*" * len(m.group(1)) + m.group(2),
                        x,
                    )
                )
            elif cat == PIICategory.SSN:
                result[col] = result[col].astype(str).apply(
                    lambda x: re.sub(r"\d{3}-\d{2}-(\d{4})", r"***-**-\1", x)
                )
            elif cat == PIICategory.CREDIT_CARD:
                result[col] = result[col].astype(str).apply(
                    lambda x: re.sub(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?(\d{4})", r"****-****-****-\1", x)
                )
            elif cat == PIICategory.PHONE:
                result[col] = result[col].astype(str).apply(
                    lambda x: re.sub(r"(\d{3})[-.\s]?\d{3}[-.\s]?(\d{4})", r"\1-***-\2", x)
                )
            else:
                result[col] = result[col].apply(lambda x: "[REDACTED]")

    logger.info(f"  Masked {len(pii_map)} PII columns")
    return result


# ── Audit Logging ─────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Structured audit log for pipeline operations.

    Writes JSON-line entries to a file. Each entry records who-did-what-when
    for compliance and troubleshooting.

    In Databricks production, audit logs are written to Unity Catalog
    (`ml_platform.monitoring.audit_log`) for queryability.
    """

    def __init__(self, log_path: str = "data/audit/audit_log.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[AuditEntry] = []

    def log(
        self,
        operation: str,
        actor: str = "pipeline",
        model_id: Optional[str] = None,
        dataset_hash: Optional[str] = None,
        feature_table: Optional[str] = None,
        target_table: Optional[str] = None,
        records_affected: int = 0,
        status: str = "success",
        details: Optional[dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record an audit entry."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            actor=actor,
            model_id=model_id,
            dataset_hash=dataset_hash,
            feature_table=feature_table,
            target_table=target_table,
            records_affected=records_affected,
            status=status,
            details=details or {},
        )
        self._entries.append(entry)
        self._flush(entry)
        return entry

    def get_recent(self, n: int = 50) -> list[AuditEntry]:
        """Return the most recent N entries."""
        return self._entries[-n:]

    def query(
        self,
        operation: Optional[str] = None,
        model_id: Optional[str] = None,
        actor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[AuditEntry]:
        """Filter audit log by criteria."""
        results = self._entries
        if operation:
            results = [e for e in results if e.operation == operation]
        if model_id:
            results = [e for e in results if e.model_id == model_id]
        if actor:
            results = [e for e in results if e.actor == actor]
        if status:
            results = [e for e in results if e.status == status]
        return results

    def count_by_operation(self) -> dict[str, int]:
        """Return a summary of operations."""
        counts: dict[str, int] = defaultdict(int)
        for e in self._entries:
            counts[e.operation] += 1
        return dict(counts)

    def _flush(self, entry: AuditEntry) -> None:
        """Append a JSON line to the audit log file."""
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except OSError:
            logger.warning(f"Audit log write failed: {self._path}")


# ── Data Versioning / Snapshot Manager ────────────────────────────────────────

class SnapshotManager:
    """
    Manages reproducible data snapshots for model training and inference.

    Each snapshot captures:
      - Data hash (SHA-256 of training DataFrame)
      - Feature table versions
      - Git commit SHA (if available)
      - MLflow run ID
      - Model config version

    Snapshots are stored as JSON in a designated directory.
    """

    def __init__(self, snapshot_dir: str = "data/snapshots"):
        self._dir = Path(snapshot_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        dataset_hash: str,
        num_rows: int,
        num_features: int,
        feature_table_versions: Optional[dict[str, str]] = None,
        mlflow_run_id: Optional[str] = None,
        model_config_version: Optional[str] = None,
        classification: str = "internal",
        retention_days: int = 365,
    ) -> DataSnapshot:
        """Create a data snapshot for reproducibility."""
        snapshot_id = f"snap_{dataset_hash[:12]}"
        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_hash=dataset_hash,
            feature_table_versions=feature_table_versions or {},
            git_commit_sha=self._get_git_sha(),
            mlflow_run_id=mlflow_run_id,
            model_config_version=model_config_version,
            num_rows=num_rows,
            num_features=num_features,
            classification=classification,
            retention_days=retention_days,
        )
        self._save(snapshot)
        logger.info(
            f"Snapshot created: {snapshot_id} "
            f"({num_rows:,} rows × {num_features} features, "
            f"hash: {dataset_hash})"
        )
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Optional[DataSnapshot]:
        """Load a snapshot by ID."""
        path = self._dir / f"{snapshot_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return DataSnapshot(**data)

    def list_snapshots(self, limit: int = 20) -> list[DataSnapshot]:
        """List recent snapshots."""
        files = sorted(self._dir.glob("snap_*.json"), reverse=True)[:limit]
        snapshots = []
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
                snapshots.append(DataSnapshot(**data))
        return snapshots

    def _save(self, snapshot: DataSnapshot) -> None:
        path = self._dir / f"{snapshot.snapshot_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)

    @staticmethod
    def _get_git_sha() -> Optional[str]:
        """Try to get the current Git commit SHA."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None


# ── Compliance Report ─────────────────────────────────────────────────────────

def generate_compliance_report(
    df: pd.DataFrame,
    dataset_name: str = "unnamed",
    snapshots: Optional[list[DataSnapshot]] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> dict[str, Any]:
    """
    Generate a compliance report for a dataset.

    Covers:
      - PII scan results
      - Data classification
      - Column-level metadata
      - Snapshot history
      - Recent audit entries
    """
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "pii_scan": {},
        "data_classification": "internal",
        "snapshots": [],
        "audit_entries": [],
    }

    # PII scan
    pii_results = detect_pii_columns(df)
    report["pii_scan"] = {
        col: [c.value for c in cats]
        for col, cats in pii_results.items()
    }
    report["has_pii"] = len(pii_results) > 0

    if report["has_pii"]:
        report["data_classification"] = "pii"
        report["masking_required"] = True
        report["masked_columns"] = list(pii_results.keys())

    # Column metadata
    col_meta = {}
    for col in df.columns:
        col_meta[col] = {
            "dtype": str(df[col].dtype),
            "null_rate": float(df[col].isna().mean()),
            "unique_ratio": float(df[col].nunique() / len(df)) if len(df) > 0 else 0,
            "is_pii": col in pii_results,
        }
    report["columns"] = col_meta

    # Snapshots
    if snapshots:
        report["snapshots"] = [asdict(s) for s in snapshots]

    # Audit log summary
    if audit_logger:
        report["audit_entries"] = [
            e.to_dict() for e in audit_logger.get_recent(10)
        ]

    return report


# ── GDPR / CCPA Utilities ─────────────────────────────────────────────────────

@dataclass
class ErasurePlan:
    """
    Plan for erasing a user's data across all tables (GDPR Right to Erasure).
    """
    user_id: str
    tables_to_modify: list[dict[str, Any]]
    estimated_records: int
    cascade_to_downstream: bool = False


def build_erasure_plan(
    user_id: str,
    tables: list[str],
    record_id_column: str = "record_id",
    cascade: bool = False,
) -> ErasurePlan:
    """
    Build a plan for erasing a user's data (GDPR Article 17 / CCPA deletion).

    In production, this generates Delta Lake DELETE statements executed
    through Unity Catalog. Locally, it produces a report of what would be
    deleted.

    Args:
        user_id: The user identifier to erase.
        tables: List of table names to search.
        record_id_column: The column that maps to user identity.
        cascade: If True, also erase downstream derived tables.

    Returns:
        ErasurePlan with table-level details.
    """
    table_details = []
    for table in tables:
        table_details.append({
            "table": table,
            "record_id_column": record_id_column,
            "delete_condition": f"{record_id_column} = '{user_id}'",
            "statement": (
                f"DELETE FROM {table} WHERE {record_id_column} = '{user_id}'"
            ),
        })

    plan = ErasurePlan(
        user_id=user_id,
        tables_to_modify=table_details,
        estimated_records=len(tables),  # Placeholder
        cascade_to_downstream=cascade,
    )

    logger.info(
        f"Erasure plan for user '{user_id}': "
        f"{len(tables)} tables affected"
    )
    return plan
