"""Direction-preserving BFS hard prompts over saved ExplaGraphs retrieval egos."""

from collections import deque
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.utils import k_hop_subgraph
from tqdm.auto import tqdm

from src.utils.notebook_downstream import artifacts
from src.utils.notebook_preprocessing import kaggle_preprocess as preprocessing


def hierarchical_description(nodes, edges, center):
    """Build a BFS forest, serialize it in preorder, and retain every non-tree edge.

    nodes maps source node IDs to text. Each edge is (source edge ID, src, relation, dst).
    Traversal uses weak adjacency; printed arrows always retain the original direction.
    """
    if center not in nodes:
        raise ValueError("The ego center is missing from its nodes.")
    if len({edge[0] for edge in edges}) != len(edges):
        raise ValueError("Source edge IDs must identify individual edge occurrences.")
    adjacency = {node: [] for node in nodes}
    for edge in edges:
        edge_id, src, relation, dst = edge
        if src not in nodes or dst not in nodes:
            raise ValueError(f"Edge {edge_id} has an endpoint outside its ego.")
        adjacency[src].append((dst, edge))
        adjacency[dst].append((src, edge))
    children = {node: [] for node in nodes}
    visited, tree_edges, roots = set(), set(), []
    for root in [center, *sorted(node for node in nodes if node != center)]:
        if root in visited:
            continue
        roots.append(root)
        visited.add(root)
        queue = deque([root])
        while queue:
            parent = queue.popleft()
            for child, edge in adjacency[parent]:
                if child not in visited:
                    visited.add(child)
                    tree_edges.add(edge[0])
                    children[parent].append((child, edge))
                    queue.append(child)

    quote = lambda value: json.dumps(str(value), ensure_ascii=True)
    lines = []
    for root in roots:
        marker = "[ROOT]" if root == center else "[COMPONENT ROOT]"
        lines.append(f"{marker} n{root} {quote(nodes[root])}")
        stack = [(root, child, edge, 1) for child, edge in reversed(children[root])]
        while stack:
            parent, child, edge, depth = stack.pop()
            edge_id, src, relation, dst = edge
            arrow = (f"--{quote(relation)}-->" if src == parent
                     else f"<--{quote(relation)}--")
            lines.append(f"{'  ' * depth}[e{edge_id}] {arrow} n{child} {quote(nodes[child])}")
            stack.extend((child, grandchild, next_edge, depth + 1)
                         for grandchild, next_edge in reversed(children[child]))
    remaining = [edge for edge in edges if edge[0] not in tree_edges]
    if remaining:
        lines.append("Additional directed edges (not in the BFS tree):")
        for edge_id, src, relation, dst in remaining:
            lines.append(f"[e{edge_id}] n{src} --{quote(relation)}--> n{dst}")
    return "\n".join(lines)


def _retrieved_description(source, index, payload):
    nodes, edges = preprocessing.load_tables(
        source.prepared, source.records[index], "expla_graphs",
    )
    edge_index = torch.tensor([edges.src.tolist(), edges.dst.tolist()], dtype=torch.long)
    centers = payload["centers"]
    if (centers.dtype != torch.long or centers.ndim != 1 or not len(centers)
            or len(centers.unique()) != len(centers)
            or centers.min() < 0 or centers.max() >= len(nodes)):
        raise ValueError(f"Sample {index} has invalid saved retrieval centers.")
    source_nodes = payload["source_nodes"]
    if (source_nodes.dtype != torch.long or source_nodes.ndim != 1
            or len(source_nodes) != len(payload["node_features"])
            or source_nodes.min() < 0 or source_nodes.max() >= len(nodes)):
        raise ValueError(f"Sample {index} has invalid saved source node identities.")
    descriptions, expected_nodes, expected_edges, expected_source_edges = [], [], [], []
    offset = 0
    for number, center in enumerate(centers.tolist(), start=1):
        subset, local_edges, _, edge_mask = k_hop_subgraph(
            center, source.config["k"], edge_index, relabel_nodes=True, num_nodes=len(nodes),
        )
        selected_edges = edges.loc[edge_mask.numpy()]
        expected_nodes.append(subset)
        expected_edges.append(local_edges + offset)
        expected_source_edges.extend(selected_edges.source_edge_id.tolist())
        offset += len(subset)
        node_text = {node: nodes.iloc[node].node_attr for node in subset.tolist()}
        text_edges = [(int(row.source_edge_id), int(row.src), row.edge_attr, int(row.dst))
                      for row in selected_edges.itertuples(index=False)]
        descriptions.append(
            f"Ego {number} (center=n{center}; arrows preserve source direction)\n"
            + hierarchical_description(node_text, text_edges, center)
        )
    if (not torch.equal(torch.cat(expected_nodes), source_nodes[payload["node_ids"]])
            or not torch.equal(torch.cat(expected_edges, dim=1), payload["edge_index"])
            or not torch.equal(torch.tensor(expected_source_edges, dtype=torch.long),
                               payload["source_edges"])):
        raise ValueError(f"Sample {index}: reconstructed egos differ from the saved retrieval graph. "
                         "Restore its original preprocessing code; no graph was changed.")
    return "\n\n".join(descriptions)


class BFSTrainingDataset(artifacts.CompactTrainingDataset):
    """Change only hard-prompt text; keep the restored snapshot and graph tensors immutable."""

    def __init__(self, source):
        if isinstance(source, BFSTrainingDataset):
            raise ValueError("Wrap the original compact dataset, not an existing BFS view.")
        if (source.metadata["dataset"] != "expla_graphs"
                or source.config.get("pipeline") == "expla_full_graph"
                or not {"k", "topk", "topk_entity"}.issubset(source.config)):
            raise ValueError("BFS text requires an ExplaGraphs retrieval snapshot with saved centers.")
        self.source = source
        self.prepared, self.run = source.prepared, source.run
        self.config, self.metadata, self.records = source.config, source.metadata, source.records
        self.descriptions = [
            _retrieved_description(source, index, source.read_payload(index))
            for index in tqdm(range(len(source)), desc="Building complete retrieved-ego BFS text")
        ]
        self.text_view = {
            "format": 1, "view": "bfs", "selection": "complete_retrieved_egos",
            "traversal": "weak_adjacency_bfs_then_preorder_original_arrows",
            "source_data_config_sha256": source.config_hash,
            "helper_sha256": artifacts.checksum(__file__),
            "descriptions_sha256": hashlib.sha256(
                json.dumps(self.descriptions, ensure_ascii=True).encode()
            ).hexdigest(),
            "sample_count": len(source),
        }
        self.config_hash = hashlib.sha256(
            json.dumps(self.text_view, sort_keys=True).encode()
        ).hexdigest()

    def read_payload(self, index):
        return {**self.source.read_payload(index), "desc": self.descriptions[index]}


def restore_dataset(source, receipt_path, expected_config_hash):
    """Rebuild the exact signed BFS view for evaluation or interrupted training."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("helper_sha256") != artifacts.checksum(__file__):
        raise ValueError("BFS text helper changed; restore the original training source.")
    dataset = BFSTrainingDataset(source)
    if receipt != dataset.text_view or dataset.config_hash != expected_config_hash:
        raise ValueError("BFS text differs from the training run; do not mix prompts or checkpoints.")
    return dataset
