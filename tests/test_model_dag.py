"""
Unit Tests — Model Dependency DAG
───────────────────────────────────────────────────────────────────────────────
Tests the ModelDependencyGraph for:
  - Topological sort correctness
  - Cycle detection
  - Sub-graph extraction (upstream/downstream)
  - Execution layer computation
  - DAG validation
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_dag import ModelDependencyGraph, DAGCycleError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_chain_models():
    """A -> B -> C simple chain."""
    return [
        {"model_id": "M_A", "depends_on": []},
        {"model_id": "M_B", "depends_on": ["M_A"]},
        {"model_id": "M_C", "depends_on": ["M_B"]},
    ]


@pytest.fixture
def diamond_models():
    """Diamond: A -> (B, C) -> D."""
    return [
        {"model_id": "M_root", "depends_on": []},
        {"model_id": "M_left", "depends_on": ["M_root"]},
        {"model_id": "M_right", "depends_on": ["M_root"]},
        {"model_id": "M_leaf", "depends_on": ["M_left", "M_right"]},
    ]


@pytest.fixture
def independent_models():
    """No dependencies between any models."""
    return [
        {"model_id": "M_1", "depends_on": []},
        {"model_id": "M_2", "depends_on": []},
        {"model_id": "M_3", "depends_on": []},
    ]


@pytest.fixture
def cyclic_models():
    """A -> B -> C -> A (cycle)."""
    return [
        {"model_id": "M_A", "depends_on": ["M_C"]},
        {"model_id": "M_B", "depends_on": ["M_A"]},
        {"model_id": "M_C", "depends_on": ["M_B"]},
    ]


@pytest.fixture
def unknown_dep_models():
    """Model depends on a non-existent model."""
    return [
        {"model_id": "M_A", "depends_on": []},
        {"model_id": "M_B", "depends_on": ["M_NONEXISTENT"]},
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestModelDependencyGraph:
    """Test suite for ModelDependencyGraph."""

    def test_simple_chain_order(self, simple_chain_models):
        """Chain should resolve to M_A -> M_B -> M_C."""
        dag = ModelDependencyGraph(simple_chain_models)
        order = dag.resolve_execution_order()
        assert order == ["M_A", "M_B", "M_C"]

    def test_diamond_order(self, diamond_models):
        """Diamond: root before left/right, leaf last."""
        dag = ModelDependencyGraph(diamond_models)
        order = dag.resolve_execution_order()
        assert order[0] == "M_root"
        assert order[-1] == "M_leaf"
        assert set(order[1:3]) == {"M_left", "M_right"}

    def test_independent_models_any_order(self, independent_models):
        """Independent models can be in any order (all valid)."""
        dag = ModelDependencyGraph(independent_models)
        order = dag.resolve_execution_order()
        assert set(order) == {"M_1", "M_2", "M_3"}

    def test_cycle_detection(self, cyclic_models):
        """Cycle should raise DAGCycleError."""
        dag = ModelDependencyGraph(cyclic_models)
        with pytest.raises(DAGCycleError, match="Circular dependency"):
            dag.resolve_execution_order()

    def test_get_dependency_graph_upstream(self, diamond_models):
        """Upstream should return direct parents."""
        dag = ModelDependencyGraph(diamond_models)
        graph = dag.get_dependency_graph("M_leaf", transitive=False)
        assert set(graph["upstream"]) == {"M_left", "M_right"}

    def test_get_dependency_graph_transitive(self, diamond_models):
        """Transitive upstream should include root."""
        dag = ModelDependencyGraph(diamond_models)
        graph = dag.get_dependency_graph("M_leaf", transitive=True)
        assert "M_root" in graph["upstream"]

    def test_get_dependency_graph_downstream(self, diamond_models):
        """Downstream should identify models that depend on this one."""
        dag = ModelDependencyGraph(diamond_models)
        graph = dag.get_dependency_graph("M_root", transitive=False)
        assert set(graph["downstream"]) == {"M_left", "M_right"}

    def test_get_dependency_graph_unknown_model(self, diamond_models):
        """Unknown model should raise ValueError."""
        dag = ModelDependencyGraph(diamond_models)
        with pytest.raises(ValueError, match="not found"):
            dag.get_dependency_graph("M_nonexistent")

    def test_execution_layers(self, diamond_models):
        """Execution layers should group independent models."""
        dag = ModelDependencyGraph(diamond_models)
        layers = dag.execution_layers()
        # Layer 0: M_root (no dependencies)
        assert layers[0] == ["M_root"]
        # Layer 1: M_left, M_right (depend only on root)
        assert set(layers[1]) == {"M_left", "M_right"}
        # Layer 2: M_leaf (depends on left + right)
        assert layers[2] == ["M_leaf"]

    def test_validation_passes(self, diamond_models):
        """Valid DAG should produce no issues."""
        dag = ModelDependencyGraph(diamond_models)
        issues = dag.validate()
        assert len(issues) == 0

    def test_validation_detects_unknown_dep(self, unknown_dep_models):
        """Validation should flag unknown dependency."""
        dag = ModelDependencyGraph(unknown_dep_models)
        issues = dag.validate()
        assert any("unknown model" in i for i in issues)

    def test_validation_detects_orphans(self):
        """Orphaned models (no deps, no dependents) should be flagged."""
        models = [
            {"model_id": "M_A", "depends_on": []},
            {"model_id": "M_orphan", "depends_on": []},
        ]
        dag = ModelDependencyGraph(models)
        issues = dag.validate()
        assert any("orphaned" in i for i in issues)

    def test_validation_detects_cycles(self, cyclic_models):
        """Validation should detect cycles."""
        dag = ModelDependencyGraph(cyclic_models)
        issues = dag.validate()
        assert any("Circular" in i for i in issues)

    def test_export_dot(self, diamond_models):
        """DOT export should produce valid output."""
        dag = ModelDependencyGraph(diamond_models)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.dot")
            result = dag.export_dot(path)
            with open(path) as f:
                content = f.read()
            assert "digraph" in content
            assert "M_root" in content
            assert result == path

    def test_model_ids_property(self, diamond_models):
        """model_ids should return all IDs."""
        dag = ModelDependencyGraph(diamond_models)
        assert set(dag.model_ids) == {"M_root", "M_left", "M_right", "M_leaf"}
