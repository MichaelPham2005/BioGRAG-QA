"""Streaming notebook preprocessing; run this file directly to avoid dataset imports."""

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil

import pandas as pd
import torch
import torch.nn.functional as F


MODEL = "sentence-transformers/all-roberta-large-v1"
PROMPT = (
    "Question: Do argument 1 and argument 2 support or counter each other? "
    "Answer in one word in the form of 'support' or 'counter'.\n\nAnswer:"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def text_tables(triples, dataset):
    nodes, edges = {}, []
    for head, relation, tail in triples:
        head, tail = head.lower(), tail.lower()
        if dataset == "expla_graphs":
            head, relation, tail = head.strip(), relation.lower().strip(), tail.strip()
        for node in (head, tail):
            if node not in nodes:
                nodes[node] = len(nodes)
        edges.append((nodes[head], relation, nodes[tail]))
    return (
        pd.DataFrame([(i, text) for text, i in nodes.items()],
                     columns=["node_id", "node_attr"]),
        pd.DataFrame(edges, columns=["src", "edge_attr", "dst"]),
    )


def prepare(dataset, output, raw=None, revision="main"):
    """Write text only: no import of the original model-loading dataset package."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        raise FileExistsError(f"{output} is already prepared; reuse it or choose a new directory.")
    if dataset == "webqsp":
        from datasets import load_dataset

        source = load_dataset("rmanluo/RoG-webqsp", revision=revision)
        source_splits, offset = {}, 0
        for name, split in (("train", "train"), ("val", "validation"), ("test", "test")):
            source_splits[name] = list(range(offset, offset + len(source[split])))
            offset += len(source[split])
        rows = (
            (row["question"], f'Question: {row["question"]}\nAnswer: ',
             "|".join(row["answer"]).lower(), row["graph"], row["id"])
            for split in ("train", "validation", "test") for row in source[split]
        )
        provenance = {
            "repository": "rmanluo/RoG-webqsp", "revision": revision,
            "fingerprints": {name: split._fingerprint for name, split in source.items()},
        }
    else:
        from sklearn.model_selection import train_test_split

        if raw is None:
            raise ValueError("ExplaGraphs requires --raw pointing to train_dev.tsv.")
        source = pd.read_csv(raw, sep="\t", keep_default_na=False)
        rows = []
        for row in source.itertuples(index=False):
            triples = [part.split(";") for part in re.findall(r"\((.*?)\)", row.graph)]
            if not triples or any(len(triple) != 3 for triple in triples):
                raise ValueError(f"Malformed ExplaGraphs graph at row {len(rows)}")
            question = f"Argument 1: {row.arg1}\nArgument 2: {row.arg2}\n{PROMPT}"
            rows.append((question, question, row.label, triples, None))
        train, rest = train_test_split(list(range(len(rows))), test_size=0.4, random_state=42)
        val, test = train_test_split(rest, test_size=0.5, random_state=42)
        source_splits = {"train": train, "val": val, "test": test}
        provenance = {"file": Path(raw).name, "sha256": digest(raw)}
    for directory in ("nodes", "edges", "split"):
        (output / directory).mkdir(exist_ok=True)
    source_locations = {
        index: (name, split_index)
        for name, indices in source_splits.items()
        for split_index, index in enumerate(indices)
    }
    records, exclusions = [], []
    for source_index, (query, question, label, triples, source_id) in enumerate(rows):
        if not query.strip() or not label.strip():
            raise ValueError(f"Empty question or label at source row {source_index}")
        source_split, source_split_index = source_locations[source_index]
        source_info = {
            "source_index": source_index, "source_id": source_id,
            "source_split": source_split, "source_split_index": source_split_index,
        }
        nodes, edges = text_tables(triples, dataset)
        if nodes.empty:
            if dataset != "webqsp":
                raise ValueError(f"Sample {source_index} has no nodes; inspect the source graph.")
            exclusions.append(dict(source_info, reason="empty_graph", question=question, label=label))
            print(f"Excluding empty graph: source_index={source_index} source_id={source_id} "
                  f"split={source_split} split_index={source_split_index}", flush=True)
            continue
        index = len(records)
        nodes.to_csv(output / "nodes" / f"{index}.csv", index=False)
        edges.to_csv(output / "edges" / f"{index}.csv", index=False)
        records.append({
            "id": index, **source_info, "query": query, "question": question, "label": label,
            "nodes": len(nodes), "edges": len(edges),
            "nodes_sha256": digest(output / "nodes" / f"{index}.csv"),
            "edges_sha256": digest(output / "edges" / f"{index}.csv"),
        })
    atomic_json(exclusions, output / "excluded.json")
    if not records:
        raise ValueError(f"No nonempty graphs remain. See {output / 'excluded.json'}.")
    source_to_id = {record["source_index"]: record["id"] for record in records}
    splits = {
        name: [source_to_id[index] for index in indices if index in source_to_id]
        for name, indices in source_splits.items()
    }
    atomic_json(records, output / "records.json")
    for name, indices in splits.items():
        (output / "split" / f"{name}_indices.txt").write_text(
            "\n".join(map(str, indices)) + "\n"
        )
    atomic_json({
        "dataset": dataset, "count": len(records), "source": provenance,
        "source_count": len(source_locations), "source_splits": source_splits,
        "empty_graph_policy": "exclude_with_audit" if dataset == "webqsp" else "reject",
        "excluded_count": len(exclusions), "exclusions_sha256": digest(output / "excluded.json"),
        "records_sha256": digest(output / "records.json"), "splits": splits,
    }, output / "prepared.json")
    print(f"Prepared {len(records)} of {len(source_locations)} {dataset} samples at {output}; "
          f"excluded {len(exclusions)} empty graphs (see excluded.json).", flush=True)


def read_prepared(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "prepared.json").read_text())
    if digest(directory / "records.json") != metadata["records_sha256"]:
        raise ValueError("Prepared records changed; use a new prepared directory.")
    records = json.loads((directory / "records.json").read_text())
    if "exclusions_sha256" in metadata:
        if digest(directory / "excluded.json") != metadata["exclusions_sha256"]:
            raise ValueError("Exclusion report changed; use a new prepared directory.")
        exclusions = json.loads((directory / "excluded.json").read_text())
        source_ids = [record["source_index"] for record in records]
        if (len(exclusions) != metadata["excluded_count"]
                or sorted(source_ids + [row["source_index"] for row in exclusions])
                != list(range(metadata["source_count"]))):
            raise ValueError("Retained and excluded rows do not partition the source data.")
        source_to_id = {record["source_index"]: record["id"] for record in records}
        expected_splits = {
            name: [source_to_id[index] for index in indices if index in source_to_id]
            for name, indices in metadata["source_splits"].items()
        }
        if metadata["splits"] != expected_splits:
            raise ValueError("Filtered split IDs do not match the original source splits.")
    indices = [i for split in metadata["splits"].values() for i in split]
    if (len(records) != metadata["count"]
            or [row["id"] for row in records] != list(range(len(records)))
            or sorted(indices) != list(range(len(records)))):
        raise ValueError("Prepared sample IDs or split partition are invalid.")
    return metadata, records


def configure_run(prepared, output, *, num_shards=2, batch_size=16,
                  max_length=512, fp16=False, k=2, topk=20, topk_entity=10,
                  model_revision="main"):
    """Refuse to mix resume files produced with different settings or inputs."""
    if min(num_shards, batch_size, max_length, topk, topk_entity) < 1 or k < 0:
        raise ValueError("Shard/batch/length/top-k values must be positive; k must be nonnegative.")
    if max_length > 512:
        raise ValueError("This RoBERTa encoder supports at most 512 input tokens.")
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    metadata, _ = read_prepared(prepared)
    config = {
        "format": 1, "prepared": str(prepared),
        "implementation_sha256": digest(__file__),
        "prepared_sha256": digest(prepared / "prepared.json"),
        "dataset": metadata["dataset"], "count": metadata["count"],
        "model": MODEL, "model_revision": model_revision,
        "num_shards": num_shards, "batch_size": batch_size, "max_length": max_length,
        "fp16": fp16, "k": k, "topk": topk, "topk_entity": topk_entity,
        "augment": "none", "webqsp_drop_all_repeated_endpoint_pairs": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    with exclusive_lock(output / "config.lock"):
        path = output / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Run settings changed. Choose a new output directory, not old caches.")
        atomic_json(config, path)
    return config


def read_config(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    if config["implementation_sha256"] != digest(__file__):
        raise ValueError("Preprocessing implementation changed; restore it or use a new run directory.")
    prepared = Path(config["prepared"])
    if digest(prepared / "prepared.json") != config["prepared_sha256"]:
        raise ValueError("Prepared input changed since this run was configured.")
    metadata, records = read_prepared(prepared)
    return config, metadata, records


class Encoder:
    def __init__(self, config, device):
        from transformers import AutoModel, AutoTokenizer

        self.config, self.device = config, device
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["model_revision"]
        )
        self.model = AutoModel.from_pretrained(
            config["model"], revision=config["model_revision"]
        ).to(device).eval()
        if config["fp16"]:
            self.model.half()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        if self.model.config.hidden_size != 1024:
            raise ValueError("Expected the repository's 1024-dimensional sentence encoder.")

    @torch.inference_mode()
    def __call__(self, texts):
        unique = list(dict.fromkeys(texts))
        if not unique:
            return torch.empty((0, 1024))
        embeddings = []
        size = self.config["batch_size"]
        for start in range(0, len(unique), size):
            tokens = self.tokenizer(
                unique[start:start + size], padding=True, truncation=True,
                max_length=self.config["max_length"], return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            embeddings.append(F.normalize(pooled, p=2, dim=1).cpu())
        embeddings = torch.cat(embeddings)
        if (embeddings.shape != (len(unique), 1024)
                or not torch.isfinite(embeddings).all()
                or not torch.allclose(embeddings.norm(dim=1), torch.ones(len(unique)), atol=1e-4)):
            raise ValueError("Encoder returned invalid embeddings.")
        lookup = {text: i for i, text in enumerate(unique)}
        return embeddings[[lookup[text] for text in texts]]


def load_tables(prepared, record, dataset):
    prepared = Path(prepared)
    paths = [prepared / kind / f'{record["id"]}.csv' for kind in ("nodes", "edges")]
    for kind, path in zip(("nodes", "edges"), paths):
        if digest(path) != record[f"{kind}_sha256"]:
            raise ValueError(f"Text table changed: {path}")
    nodes = pd.read_csv(paths[0], keep_default_na=False, dtype={"node_attr": str})
    edges = pd.read_csv(paths[1], keep_default_na=False, dtype={"edge_attr": str})
    if nodes.node_id.tolist() != list(range(len(nodes))):
        raise ValueError(f'Node IDs are not contiguous for sample {record["id"]}')
    if len(edges) and (edges[["src", "dst"]].to_numpy().min() < 0
                       or edges[["src", "dst"]].to_numpy().max() >= len(nodes)):
        raise ValueError(f'Invalid edge endpoint for sample {record["id"]}')
    if dataset == "webqsp":
        pairs = list(zip(edges.src, edges.dst))
        counts = Counter(pairs)
        # Match the author's filter, but keep text and embeddings aligned.
        edges = edges.loc[[counts[pair] == 1 for pair in pairs]]
    return nodes, edges.reset_index(names="source_edge_id")


def retrieve_compact(nodes, edges, x, relations, relation_ids, query, config):
    """Keep overlapping egos logically separate; share their features on disk."""
    from torch_geometric.utils import k_hop_subgraph

    edge_index = torch.tensor([edges.src.tolist(), edges.dst.tolist()], dtype=torch.long)

    def ego(center):
        return k_hop_subgraph(center, config["k"], edge_index,
                             relabel_nodes=True, num_nodes=len(nodes))

    scores = []
    for center in range(len(nodes)):
        subset, _, _, edge_mask = ego(center)
        edge_features = relations[relation_ids[edge_mask]]
        pooled = (x[subset].sum(0) + edge_features.sum(0)) / (len(subset) + len(edge_features))
        scores.append(F.cosine_similarity(query, pooled, dim=0))
    scores = torch.stack(scores)
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite ego-graph relevance scores.")
    centers = (torch.topk(scores, config["topk"]).indices if len(nodes) > config["topk"]
               else torch.arange(len(nodes)))
    selected_nodes, selected_relations, edge_indices, source_edges, descriptions = [], [], [], [], []
    offset = 0
    for center in centers.tolist():
        subset, local_edges, _, edge_mask = ego(center)
        edge_rows = edge_mask.nonzero().flatten()
        selected_nodes.append(subset)
        selected_relations.append(relation_ids[edge_rows])
        edge_indices.append(local_edges + offset)
        source_edges.append(torch.tensor(edges.iloc[edge_rows.tolist()].source_edge_id.to_numpy()))
        offset += len(subset)
        node_scores = F.cosine_similarity(query.unsqueeze(0), x[subset], dim=-1)
        node_top = (torch.topk(node_scores, config["topk_entity"]).indices
                    if len(subset) > config["topk_entity"] else torch.arange(len(subset)))
        descriptions.append(nodes.iloc[subset[node_top].tolist()].to_csv(index=False))
        if len(edge_rows):
            edge_scores = F.cosine_similarity(
                query.unsqueeze(0), relations[relation_ids[edge_rows]], dim=-1
            )
            edge_top = (torch.topk(edge_scores, config["topk_entity"]).indices.sort().values
                        if len(edge_rows) > config["topk_entity"] else torch.arange(len(edge_rows)))
            descriptions.append(edges.iloc[edge_rows[edge_top].tolist()].to_csv(
                index=False, columns=["src", "edge_attr", "dst"]
            ))
    original_nodes = torch.cat(selected_nodes)
    used_nodes, node_ids = original_nodes.unique(sorted=True, return_inverse=True)
    used_relations, edge_ids = torch.cat(selected_relations).unique(sorted=True, return_inverse=True)
    return {
        "node_features": x[used_nodes].clone(),
        "edge_features": relations[used_relations].clone(),
        "node_ids": node_ids, "edge_ids": edge_ids,
        "edge_index": torch.cat(edge_indices, dim=1),
        "query": query.clone(), "source_nodes": used_nodes,
        "source_edges": torch.cat(source_edges),
        "centers": centers, "scores": scores,
        "desc": "\n".join(descriptions),
        "stats": {"raw_nodes": len(nodes), "filtered_edges": len(edges),
                  "retrieved_nodes": len(node_ids), "retrieved_edges": len(edge_ids)},
    }


def validate_payload(payload):
    query = payload["query"]
    width = query.numel()
    if query.ndim != 1 or width != 1024:
        raise ValueError("Query must be a 1024-dimensional vector.")
    for key in ("node_features", "edge_features"):
        value = payload[key]
        if value.ndim != 2 or value.shape[1] != width or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {key}.")
    if not torch.isfinite(query).all() or not torch.isfinite(payload["scores"]).all():
        raise ValueError("Non-finite query or ranking scores.")
    for key, features in (("node_ids", "node_features"), ("edge_ids", "edge_features")):
        indices = payload[key]
        if (indices.dtype != torch.long or indices.ndim != 1
                or (indices.numel() and (indices.min() < 0
                                       or indices.max() >= len(payload[features])))):
            raise ValueError(f"Invalid {key}.")
    edge_index = payload["edge_index"]
    if (edge_index.dtype != torch.long or edge_index.shape != (2, len(payload["edge_ids"]))
            or (edge_index.numel() and (edge_index.min() < 0
                                       or edge_index.max() >= len(payload["node_ids"])))):
        raise ValueError("Invalid retrieved edge endpoints.")
    if not len(payload["node_ids"]) or not payload["desc"].strip():
        raise ValueError("Retrieved graph or description is empty.")
    for key in ("question", "label"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"Invalid {key}.")


def load_payload(path):
    # Files contain only tensors and built-in containers, not pickled PyG objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_payload(payload)
    return payload


def materialize(payload):
    from torch_geometric.data import Data

    validate_payload(payload)
    x = payload["node_features"][payload["node_ids"]].float()
    edge_attr = payload["edge_features"][payload["edge_ids"]].float()
    query = payload["query"].float()
    graph = Data(
        x=x, edge_index=payload["edge_index"], edge_attr=edge_attr,
        question_node=query.unsqueeze(0).repeat(len(x), 1),
        question_edge=query.unsqueeze(0).repeat(len(edge_attr), 1),
        num_nodes=len(x),
    )
    sample = {key: payload[key] for key in ("id", "question", "label", "desc")}
    sample.update({key: payload[key] for key in
                   ("source_index", "source_id", "source_split", "source_split_index")
                   if key in payload})
    return sample | {"graph": graph}


class CompactDataset(torch.utils.data.Dataset):
    """GraphLLM sample contract, with an explicit run directory (not legacy caches)."""

    def __init__(self, output):
        self.output = Path(output)
        self.config, self.metadata, self.records = read_config(output)
        self.prompt = "Please answer the given question." if self.config["dataset"] == "webqsp" else PROMPT
        self.graph = None
        self.graph_type = "Knowledge Graph" if self.config["dataset"] == "webqsp" else "Explanation Graph"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        payload = load_payload(self.output / "samples" / f"{index}.pt")
        if payload["id"] != index or payload["config_sha256"] != digest(self.output / "config.json"):
            raise ValueError(f"Wrong sample ID or run configuration: {index}")
        for key in ("source_index", "source_id", "source_split", "source_split_index"):
            if key in self.records[index] and payload.get(key) != self.records[index][key]:
                raise ValueError(f"Wrong source mapping for sample {index}: {key}")
        return materialize(payload)

    def get_idx_split(self):
        return self.metadata["splits"]


def worker(output, shard, stop=None):
    output = Path(output)
    config, _, records = read_config(output)
    if not 0 <= shard < config["num_shards"]:
        raise ValueError("Shard must be in [0, num_shards).")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU T4 x2; each worker requires a visible CUDA device.")
    if stop is not None and not 0 < stop <= len(records):
        raise ValueError("Stop must be within the prepared dataset.")
    torch.set_num_threads(2)
    config_hash = digest(output / "config.json")
    with exclusive_lock(output / f"worker-{shard}.lock"):
        encoder = Encoder(config, "cuda:0")
        print(f"shard={shard} pid={os.getpid()} GPU={torch.cuda.get_device_name(0)} "
              f"model_revision={encoder.revision}", flush=True)
        for index in range(shard, stop if stop is not None else len(records), config["num_shards"]):
            destination = output / "samples" / f"{index}.pt"
            if destination.exists():
                payload = load_payload(destination)
                if payload["config_sha256"] != config_hash or payload["id"] != index:
                    raise ValueError(f"Stale sample {destination}; use a fresh run directory.")
                print(f"shard={shard} resumed={index}", flush=True)
                continue
            record = records[index]
            nodes, edges = load_tables(config["prepared"], record, config["dataset"])
            relation_text = list(dict.fromkeys(edges.edge_attr.tolist()))
            lookup = {text: i for i, text in enumerate(relation_text)}
            relation_ids = torch.tensor([lookup[text] for text in edges.edge_attr], dtype=torch.long)
            query = encoder([record["query"]])[0]
            x, relations = encoder(nodes.node_attr.tolist()), encoder(relation_text)
            payload = retrieve_compact(nodes, edges, x, relations, relation_ids, query, config)
            payload.update(id=index, question=record["question"], label=record["label"],
                           config_sha256=config_hash, model_revision=encoder.revision)
            payload.update({key: record[key] for key in
                            ("source_index", "source_id", "source_split", "source_split_index")})
            del x, relations
            validate_payload(payload)
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            if shutil.disk_usage(output).free < buffer.tell() + 2 * 2**30:
                raise OSError("Less than 2 GiB free after this sample; save outputs and resume elsewhere.")
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(buffer.getbuffer())
            os.replace(temporary, destination)
            print(f"shard={shard} saved={index} bytes={buffer.tell()} stats={payload['stats']}", flush=True)
            del payload, buffer


def validate_run(output):
    config, _, records = read_config(output)
    dataset = CompactDataset(output)
    for index in range(len(records)):
        sample = dataset[index]
        graph = sample["graph"]
        if graph.question_node.shape != graph.x.shape or graph.question_edge.shape != graph.edge_attr.shape:
            raise ValueError(f"Invalid query feature expansion: {index}")
    files = list((Path(output) / "samples").glob("*.pt"))
    if len(files) != config["count"]:
        raise ValueError("Unexpected sample files in output directory.")
    print(f"Validated {len(dataset)} compact samples and expanded model inputs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--dataset", choices=["webqsp", "expla_graphs"], required=True)
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--raw")
    preparation.add_argument("--revision", default="main")
    processing = commands.add_parser("worker")
    processing.add_argument("--output", required=True)
    processing.add_argument("--shard", type=int, required=True)
    processing.add_argument("--stop", type=int)
    validation = commands.add_parser("validate")
    validation.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare": prepare, "worker": worker, "validate": validate_run}[command](**args)


if __name__ == "__main__":
    main()
