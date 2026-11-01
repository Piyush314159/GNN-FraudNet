"""
graph_analysis.py
=================
Graph analysis service — extracts structural information about a node's
neighborhood from the loaded PyG graph.

Provides neighbor label distribution, degree information, and structural
pattern flags that the investigation context builder can use.
"""
import logging
from dataclasses import dataclass, field

import torch
from torch_geometric.utils import k_hop_subgraph

logger = logging.getLogger(__name__)


@dataclass
class NeighborSummary:
    """Label distribution of a node's neighbors."""

    total: int = 0
    illicit: int = 0
    licit: int = 0
    unknown: int = 0
    illicit_ratio: float = 0.0


@dataclass
class GraphAnalysisResult:
    """Structured output from graph neighborhood analysis."""

    node_id: int
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0
    ground_truth_label: str = "unknown"
    neighbors_1hop: NeighborSummary = field(default_factory=NeighborSummary)
    neighbors_2hop: NeighborSummary = field(default_factory=NeighborSummary)
    structural_flags: list[str] = field(default_factory=list)
    neighbor_node_ids: list[int] = field(default_factory=list)


class GraphAnalysisService:
    """Analyzes the graph structure around a given node."""

    def __init__(self, data, known_label_mask=None):
        self.data = data
        # Dataset labels are evaluation answers, not operational intelligence.
        # Callers must explicitly identify labels known at investigation time.
        self.known_label_mask = (
            torch.zeros(data.num_nodes, dtype=torch.bool, device=data.y.device)
            if known_label_mask is None
            else torch.as_tensor(known_label_mask, dtype=torch.bool, device=data.y.device).clone()
        )
        if self.known_label_mask.shape != data.y.shape:
            raise ValueError("known_label_mask must have one entry per node")
        self._precompute_degrees()

    def _precompute_degrees(self) -> None:
        """Precompute in/out degree for all nodes."""
        num_nodes = self.data.x.size(0)
        src = self.data.edge_index[0]
        dst = self.data.edge_index[1]

        self._out_degree = torch.zeros(num_nodes, dtype=torch.long)
        self._in_degree = torch.zeros(num_nodes, dtype=torch.long)
        self._out_degree.scatter_add_(0, src, torch.ones_like(src))
        self._in_degree.scatter_add_(0, dst, torch.ones_like(dst))

    def analyze_node(self, node_id: int) -> GraphAnalysisResult:
        """Perform full structural analysis of a node's neighborhood."""
        if node_id < 0 or node_id >= self.data.num_nodes:
            raise ValueError(f"Node {node_id} not in graph")

        # Degree
        in_deg = self._in_degree[node_id].item()
        out_deg = self._out_degree[node_id].item()

        # 1-hop neighbors
        neighbors_1 = self._get_neighbor_summary(node_id, num_hops=1)
        # 2-hop neighbors
        neighbors_2 = self._get_neighbor_summary(node_id, num_hops=2)

        # Neighbor node IDs (1-hop only, for context)
        subset_1, _, _, _ = k_hop_subgraph(
            node_id, 1, self.data.edge_index, relabel_nodes=False
        )
        neighbor_ids = [n for n in subset_1.tolist() if n != node_id]

        # Structural flags
        flags = self._detect_structural_flags(
            node_id, in_deg, out_deg, neighbors_1
        )

        return GraphAnalysisResult(
            node_id=node_id,
            in_degree=in_deg,
            out_degree=out_deg,
            total_degree=in_deg + out_deg,
            neighbors_1hop=neighbors_1,
            neighbors_2hop=neighbors_2,
            structural_flags=flags,
            neighbor_node_ids=neighbor_ids[:50],  # cap at 50 for response size
        )

    def _get_neighbor_summary(self, node_id: int, num_hops: int) -> NeighborSummary:
        """Compute label distribution of k-hop neighbors."""
        subset, _, _, _ = k_hop_subgraph(
            node_id, num_hops, self.data.edge_index, relabel_nodes=False
        )

        # Exclude the target node itself
        neighbor_mask = subset != node_id
        neighbor_indices = subset[neighbor_mask]

        if len(neighbor_indices) == 0:
            return NeighborSummary()

        labels = self.data.y[neighbor_indices].clone()
        labels[~self.known_label_mask[neighbor_indices]] = -1
        n_illicit = (labels == 1).sum().item()
        n_licit = (labels == 0).sum().item()
        n_unknown = (labels == -1).sum().item()
        total = len(neighbor_indices)
        labeled = n_illicit + n_licit

        return NeighborSummary(
            total=total,
            illicit=n_illicit,
            licit=n_licit,
            unknown=n_unknown,
            illicit_ratio=n_illicit / labeled if labeled > 0 else 0.0,
        )

    def _detect_structural_flags(
        self,
        node_id: int,
        in_deg: int,
        out_deg: int,
        neighbors: NeighborSummary,
    ) -> list[str]:
        """Detect notable structural patterns around the node."""
        flags = []

        total_deg = in_deg + out_deg
        avg_deg = (self._in_degree + self._out_degree).float().mean().item()

        # High degree
        if total_deg > avg_deg * 3:
            flags.append("high_degree")

        # Fan-out
        if out_deg > 0 and out_deg > in_deg * 3:
            flags.append("fan_out_pattern")

        # Fan-in
        if in_deg > 0 and in_deg > out_deg * 3:
            flags.append("fan_in_pattern")

        # High illicit concentration
        if neighbors.total > 0 and neighbors.illicit_ratio > 0.5:
            flags.append("high_illicit_concentration")

        # Moderate illicit concentration
        if neighbors.total > 0 and 0.2 < neighbors.illicit_ratio <= 0.5:
            flags.append("moderate_illicit_concentration")

        # Isolated (no neighbors)
        if total_deg == 0:
            flags.append("isolated_node")

        # Mostly unknown neighbors (high uncertainty)
        if neighbors.total > 0 and neighbors.unknown / neighbors.total > 0.7:
            flags.append("mostly_unknown_neighbors")

        return flags
