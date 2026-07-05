"""
Model Dependency DAG — Topological Resolution
───────────────────────────────────────────────────────────────────────────────
Resolves `depends_on` relationships between models declared in
`model_registry.yaml`. Supports:

  - **DAG construction**: Build a directed graph from model config dependencies
  - **Topological sort**: Produce an execution order respecting dependencies
  - **Cycle detection**: Fail fast on circular dependencies
  - **Sub-graph extraction**: Resolve dependencies for a single model
  - **Visualisation**: Export DOT graph for debugging

Usage:
    dag = ModelDependencyGraph(model_registry["models"])
    ordered = dag.resolve_execution_order()   # Topologically sorted
    dep_graph = dag.get_dependency_graph("clf_segment_propensity")
    dag.export_dot("dependencies.dot")        # Visualise

Config format (in model_registry.yaml):
    models:
      - model_id: "clust_customer_segments"
        depends_on: []                          # No dependencies

      - model_id: "clf_segment_propensity"
        depends_on: ["clust_customer_segments"] # Uses segment output

      - model_id: "clf_segment_ltv"
        depends_on: ["clust_customer_segments"] # Also depends on segments

      - model_id: "reg_ensemble_model"
        depends_on: ["clf_segment_propensity", "clf_segment_ltv"]
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Optional

from src.utils import logger


class DAGCycleError(Exception):
    """Raised when a circular dependency is detected in the model DAG."""


class ModelDependencyGraph:
    """
    Directed Acyclic Graph (DAG) for model dependency resolution.

    Each model can declare which other models it depends on. The DAG resolves
    the correct execution order so that upstream models are trained/scored
    before their downstream consumers.

    Graphs are immutable after construction — build a new one if config changes.
    """

    def __init__(self, models: list[dict[str, Any]]):
        self._models = {m["model_id"]: m for m in models}
        self._graph: dict[str, set[str]] = {}
        self._build_graph()
        self._topo_order: Optional[list[str]] = None

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def model_ids(self) -> list[str]:
        """Return all model IDs in the DAG."""
        return list(self._models.keys())

    def resolve_execution_order(self) -> list[str]:
        """
        Return model IDs in topological order (dependencies first).

        Raises DAGCycleError if a cycle is detected.
        """
        if self._topo_order is not None:
            return self._topo_order

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {m: 0 for m in self._models}
        adjacency: dict[str, list[str]] = {m: [] for m in self._models}

        for model_id, deps in self._graph.items():
            for dep in deps:
                if dep in adjacency:
                    adjacency[dep].append(model_id)
                    in_degree[model_id] = in_degree.get(model_id, 0) + 1

        queue = deque([m for m, deg in in_degree.items() if deg == 0])
        sorted_order: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_order.append(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(self._models):
            cyclic = set(self._models.keys()) - set(sorted_order)
            raise DAGCycleError(
                f"Circular dependency detected among models: {cyclic}"
            )

        self._topo_order = sorted_order
        logger.info(
            f"Model DAG: resolved {len(sorted_order)} models in topological order "
            f"[dim](layers: {self._compute_layers()})[/]"
        )
        return sorted_order

    def get_dependency_graph(
        self, model_id: str, transitive: bool = True
    ) -> dict[str, list[str]]:
        """
        Return the dependency sub-graph for a single model.

        Args:
            model_id: Target model ID.
            transitive: If True, include transitive (indirect) dependencies.

        Returns:
            dict with 'upstream' (what this model needs) and
            'downstream' (what needs this model).
        """
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not found in registry")

        upstream = set(self._graph.get(model_id, []))

        if transitive:
            # BFS for transitive closure
            visited = set(upstream)
            queue = deque(upstream)
            while queue:
                node = queue.popleft()
                deps = self._graph.get(node, [])
                for dep in deps:
                    if dep not in visited:
                        visited.add(dep)
                        queue.append(dep)
            upstream = visited

        # Downstream: models that depend on this one
        downstream = set()
        for mid, deps in self._graph.items():
            if model_id in deps:
                downstream.add(mid)
        if transitive:
            visited = set(downstream)
            queue = deque(downstream)
            while queue:
                node = queue.popleft()
                for mid, deps in self._graph.items():
                    if node in deps and mid not in visited:
                        visited.add(mid)
                        queue.append(mid)
            downstream = visited

        return {
            "model_id": model_id,
            "upstream": sorted(upstream),
            "downstream": sorted(downstream),
            "transitive": transitive,
        }

    def execution_layers(self) -> list[list[str]]:
        """
        Group models into parallelisable layers.

        Models in the same layer have no dependencies on each other and can
        be executed in parallel. Each layer depends only on models in earlier
        layers.

        Returns:
            list of lists: each inner list is a set of models that can run
            concurrently.
        """
        order = self.resolve_execution_order()
        layers: list[list[str]] = []
        processed: set[str] = set()

        remaining = set(order)
        while remaining:
            layer = [
                m for m in remaining
                if all(dep in processed for dep in self._graph.get(m, []))
            ]
            if not layer:
                # Should not happen if DAG is valid, but guard against it
                layer = [remaining.pop()]

            layers.append(sorted(layer))
            processed.update(layer)
            remaining -= set(layer)

        logger.info(
            f"Model DAG: {len(layers)} execution layers "
            f"({', '.join(str(len(l)) for l in layers)} models each)"
        )
        return layers

    def validate(self) -> list[str]:
        """
        Validate the entire DAG.

        Returns:
            list of warning/error messages (empty = all good).
        """
        issues: list[str] = []

        # Check all referenced dependencies exist
        for model_id, deps in self._graph.items():
            for dep in deps:
                if dep not in self._models:
                    issues.append(
                        f"Model '{model_id}' depends on unknown model '{dep}'"
                    )

        # Check for cycles
        try:
            self.resolve_execution_order()
        except DAGCycleError as e:
            issues.append(str(e))

        # Check for orphaned models (no depends_on and no dependents)
        all_referenced: set[str] = set()
        for deps in self._graph.values():
            all_referenced.update(deps)
        for model_id in self._models:
            if model_id not in all_referenced and not self._graph.get(model_id):
                issues.append(
                    f"Model '{model_id}' is orphaned — no dependencies or dependents"
                )

        if not issues:
            logger.info(f"Model DAG: validation passed ({len(self._models)} models)")
        else:
            for issue in issues:
                logger.warning(f"Model DAG validation: {issue}")

        return issues

    def export_dot(self, path: str = "model_dependencies.dot") -> str:
        """
        Export the DAG as Graphviz DOT for visualisation.

        Usage:
            dag.export_dot("deps.dot")
            # dot -Tpng deps.dot -o deps.png
        """
        lines = ['digraph ModelDependencies {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=rounded, fontname="monospace"];')
        lines.append('  edge [arrowhead=vee, color="#666666"];')

        for model_id in self._models:
            deps = self._graph.get(model_id, [])
            for dep in deps:
                lines.append(f'  "{dep}" -> "{model_id}";')

        lines.append('}')

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Model DAG: exported DOT → [cyan]{path}[/]")
        return path

    # ── Internal ─────────────────────────────────────────────────────────

    def _build_graph(self) -> None:
        """Build the internal adjacency list from model configs."""
        for model_id, model_cfg in self._models.items():
            deps = model_cfg.get("depends_on", [])
            if not isinstance(deps, list):
                deps = []
            # Normalise: strip empty strings, ensure strings
            deps = [str(d) for d in deps if d]
            self._graph[model_id] = set(deps)

    def _compute_layers(self) -> int:
        """Count the number of topological layers (optimised)."""
        try:
            return len(self.execution_layers())
        except DAGCycleError:
            return -1
