"""ExplaGraphs full-graph artifacts, without ego extraction, ranking or pruning."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import shutil

import torch
from tqdm.auto import tqdm

from src.utils.notebook_preprocessing import kaggle_preprocess as preprocessing


PIPELINE = "expla_full_graph"


def configure_run(prepared, output, *, model_revision, num_shards=2, batch_size=16):
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("Pin the sentence encoder to its full Hugging Face commit SHA.")
    if num_shards < 1 or batch_size < 1:
        raise ValueError("Shard and embedding batch counts must be positive.")
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    metadata, _ = preprocessing.read_prepared(prepared)
    if metadata["dataset"] != "expla_graphs" or metadata.get("excluded_count") != 0:
        raise ValueError("Full-graph preprocessing requires all original ExplaGraphs rows.")
    config = {
        "format": 1, "pipeline": PIPELINE, "dataset": "expla_graphs",
        "prepared": str(prepared), "count": metadata["count"],
        "prepared_sha256": preprocessing.digest(prepared / "prepared.json"),
        "implementation_sha256": preprocessing.digest(__file__),
        "shared_helper_sha256": preprocessing.digest(preprocessing.__file__),
        "model": preprocessing.MODEL, "model_revision": model_revision,
        "num_shards": num_shards, "batch_size": batch_size,
        "max_length": 512, "fp16": False, "augment": "none",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    with preprocessing.exclusive_lock(output / "config.lock"):
        path = output / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Full-graph settings/data/code changed; use a new preprocessing directory.")
        preprocessing.atomic_json(config, path)
    return config


def encode_record(record, nodes, edges, encoder, config_hash, model_revision):
    relation_text = list(dict.fromkeys(edges.edge_attr.tolist()))
    lookup = {text: i for i, text in enumerate(relation_text)}
    payload = {
        "pipeline": PIPELINE, "id": record["id"], "question": record["question"],
        "label": record["label"], "query": encoder([record["query"]])[0],
        "node_features": encoder(nodes.node_attr.tolist()),
        "edge_features": encoder(relation_text),
        "node_ids": torch.arange(len(nodes)),
        "edge_ids": torch.tensor([lookup[text] for text in edges.edge_attr], dtype=torch.long),
        "edge_index": torch.tensor([edges.src.tolist(), edges.dst.tolist()], dtype=torch.long),
        "source_nodes": torch.arange(len(nodes)),
        "source_edges": torch.tensor(edges.source_edge_id.tolist(), dtype=torch.long),
        "centers": torch.empty(0, dtype=torch.long), "scores": torch.empty(0),
        "desc": nodes.to_csv(index=False) + "\n" + edges.to_csv(
            index=False, columns=["src", "edge_attr", "dst"]
        ),
        "stats": {"raw_nodes": len(nodes), "filtered_edges": len(edges),
                  "retrieved_nodes": len(nodes), "retrieved_edges": len(edges)},
        "config_sha256": config_hash, "model_revision": model_revision,
    }
    payload.update({key: record[key] for key in
                    ("source_index", "source_id", "source_split", "source_split_index")})
    validate_record(payload, record, nodes, edges, config_hash, model_revision)
    return payload


def validate_record(payload, record, nodes, edges, config_hash, model_revision):
    preprocessing.validate_payload(payload)
    expected_description = nodes.to_csv(index=False) + "\n" + edges.to_csv(
        index=False, columns=["src", "edge_attr", "dst"]
    )
    relations = list(dict.fromkeys(edges.edge_attr.tolist()))
    relation_lookup = {text: i for i, text in enumerate(relations)}
    expected_tensors = {
        "node_ids": torch.arange(len(nodes)), "source_nodes": torch.arange(len(nodes)),
        "source_edges": torch.tensor(edges.source_edge_id.tolist(), dtype=torch.long),
        "edge_ids": torch.tensor([relation_lookup[text] for text in edges.edge_attr], dtype=torch.long),
        "edge_index": torch.tensor([edges.src.tolist(), edges.dst.tolist()], dtype=torch.long),
    }
    if (payload.get("pipeline") != PIPELINE or payload["id"] != record["id"]
            or payload["config_sha256"] != config_hash
            or payload["model_revision"] != model_revision
            or payload["desc"] != expected_description
            or len(payload["node_features"]) != len(nodes)
            or len(payload["edge_features"]) != len(relations)
            or payload["centers"].numel() or payload["scores"].numel()
            or any(payload[key] != record[key] for key in
                   ("question", "label", "source_index", "source_id",
                    "source_split", "source_split_index"))
            or any(not torch.equal(payload[key], value) for key, value in expected_tensors.items())):
        raise ValueError(f"Full graph lost, reordered or changed data for sample {record['id']}.")


def worker(output, shard):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    if (config.get("pipeline") != PIPELINE
            or config["implementation_sha256"] != preprocessing.digest(__file__)
            or config["shared_helper_sha256"] != preprocessing.digest(preprocessing.__file__)):
        raise ValueError("Restore the original full-graph preprocessing code or use a new run.")
    metadata, records = preprocessing.read_prepared(config["prepared"])
    if (metadata["dataset"] != "expla_graphs"
            or preprocessing.digest(Path(config["prepared"]) / "prepared.json") != config["prepared_sha256"]
            or len(records) != config["count"]):
        raise ValueError("Prepared ExplaGraphs data differs from this full-graph run.")
    if not 0 <= shard < config["num_shards"]:
        raise ValueError("Shard must be within the configured shard count.")
    if not torch.cuda.is_available():
        raise RuntimeError("Each embedding worker requires one visible Kaggle GPU.")
    torch.set_num_threads(2)
    config_hash = preprocessing.digest(output / "config.json")
    with preprocessing.exclusive_lock(output / f"worker-{shard}.lock"):
        encoder = preprocessing.Encoder(config, "cuda:0")
        if encoder.revision != config["model_revision"]:
            raise ValueError("Loaded encoder revision differs from the pinned commit.")
        for index in tqdm(range(shard, len(records), config["num_shards"]), desc=f"Full-graph shard {shard}"):
            record = records[index]
            nodes, edges = preprocessing.load_tables(config["prepared"], record, "expla_graphs")
            destination = output / "samples" / f"{index}.pt"
            if destination.exists():
                validate_record(preprocessing.load_payload(destination), record, nodes, edges,
                                config_hash, config["model_revision"])
                continue
            payload = encode_record(record, nodes, edges, encoder, config_hash, encoder.revision)
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            if shutil.disk_usage(output).free < buffer.tell() + 2 * 2**30:
                raise OSError("Less than 2 GiB free after this sample; preserve outputs before resuming.")
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            try:
                temporary.write_bytes(buffer.getbuffer())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def validate_dataset(dataset):
    if (dataset.config.get("pipeline") != PIPELINE
            or dataset.metadata["dataset"] != "expla_graphs"
            or dataset.metadata.get("excluded_count") != 0):
        raise ValueError("Expected complete full-graph ExplaGraphs data, not a retrieval snapshot.")
    for record in tqdm(dataset.records, desc="Verifying complete graphs"):
        nodes, edges = preprocessing.load_tables(dataset.prepared, record, "expla_graphs")
        validate_record(dataset.read_payload(record["id"]), record, nodes, edges,
                        dataset.config_hash, dataset.config["model_revision"])
    print(f"Verified every node and edge, full CSV text and source split for {len(dataset)} graphs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", required=True, type=int)
    args = parser.parse_args()
    worker(args.output, args.shard)
