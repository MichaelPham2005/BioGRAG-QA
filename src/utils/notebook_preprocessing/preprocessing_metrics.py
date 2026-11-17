"""Measure the notebook's encoder and retrieval workload without changing its caches."""

import argparse
import importlib.util
import io
import json
from pathlib import Path
import time

import torch


SPEC = importlib.util.spec_from_file_location(
    "kaggle_preprocess", Path(__file__).with_name("kaggle_preprocess.py")
)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def size_buckets(records, shard, num_shards, count=3):
    indices = sorted(
        range(shard, len(records), num_shards),
        key=lambda index: (records[index]["nodes"] + records[index]["edges"], index),
    )
    count = min(count, len(indices))
    if count < 1:
        raise ValueError("This shard has no samples to benchmark.")
    return [indices[i * len(indices) // count:(i + 1) * len(indices) // count]
            for i in range(count)]


def benchmark(output, shard):
    output = Path(output)
    config, _, records = pipeline.read_config(output)
    if not 0 <= shard < config["num_shards"]:
        raise ValueError("Shard is outside the configured range.")
    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires the configured Kaggle CUDA GPUs.")
    destination = output / "benchmark"
    destination.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    with pipeline.exclusive_lock(destination / f"shard-{shard}.lock"):
        start = time.perf_counter()
        encoder = pipeline.Encoder(config, "cuda:0")
        encoder(["Preprocessing timing warmup."])
        torch.cuda.synchronize()
        startup_seconds = time.perf_counter() - start
        parameter_count = sum(parameter.numel() for parameter in encoder.model.parameters())
        weight_bytes = sum(parameter.numel() * parameter.element_size()
                           for parameter in encoder.model.parameters())
        observations = []
        for bucket in size_buckets(records, shard, config["num_shards"]):
            index = bucket[len(bucket) // 2]
            record = records[index]
            start = time.perf_counter()
            nodes, edges = pipeline.load_tables(config["prepared"], record, config["dataset"])
            relation_text = list(dict.fromkeys(edges.edge_attr.tolist()))
            lookup = {text: i for i, text in enumerate(relation_text)}
            relation_ids = torch.tensor([lookup[text] for text in edges.edge_attr], dtype=torch.long)
            table_seconds = time.perf_counter() - start
            torch.cuda.synchronize()
            start = time.perf_counter()
            query = encoder([record["query"]])[0]
            x, relations = encoder(nodes.node_attr.tolist()), encoder(relation_text)
            torch.cuda.synchronize()
            embedding_seconds = time.perf_counter() - start
            start = time.perf_counter()
            payload = pipeline.retrieve_compact(
                nodes, edges, x, relations, relation_ids, query, config
            )
            retrieval_seconds = time.perf_counter() - start
            payload.update(id=index, question=record["question"], label=record["label"])
            start = time.perf_counter()
            pipeline.validate_payload(payload)
            with io.BytesIO() as buffer:
                torch.save(payload, buffer)
                serialized_bytes = buffer.tell()
            serialization_seconds = time.perf_counter() - start
            observations.append({
                "sample_id": index, "bucket_ids": bucket,
                "table_seconds": table_seconds, "embedding_seconds": embedding_seconds,
                "retrieval_seconds": retrieval_seconds,
                "validation_serialization_seconds": serialization_seconds,
                "total_seconds": table_seconds + embedding_seconds + retrieval_seconds + serialization_seconds,
                "serialized_bytes": serialized_bytes,
            })
            print(f"benchmark shard={shard} sample={index} "
                  f"embedding={embedding_seconds:.2f}s retrieval={retrieval_seconds:.2f}s", flush=True)
            del payload, x, relations, query
        report = {
            "config_sha256": pipeline.digest(output / "config.json"),
            "shard": shard, "num_shards": config["num_shards"],
            "gpu": torch.cuda.get_device_name(0),
            "parameter_count_per_encoder": parameter_count,
            "optimized_parameters": 0, "weight_bytes_per_gpu": weight_bytes,
            "startup_warmup_seconds": startup_seconds, "observations": observations,
        }
        pipeline.atomic_json(report, destination / f"shard-{shard}.json")
        print(f"benchmark shard={shard}: {parameter_count:,} parameters, "
              "inference only (0 optimized parameters)", flush=True)
        return report


def estimate_remaining(output):
    output = Path(output)
    config, _, records = pipeline.read_config(output)
    reports = [
        json.loads((output / "benchmark" / f"shard-{shard}.json").read_text())
        for shard in range(config["num_shards"])
    ]
    config_hash = pipeline.digest(output / "config.json")
    pending = {i for i in range(len(records)) if not (output / "samples" / f"{i}.pt").is_file()}
    estimates = []
    counts = set()
    for shard, report in enumerate(reports):
        if (report["config_sha256"] != config_hash or report["shard"] != shard
                or report["num_shards"] != config["num_shards"]):
            raise ValueError("Benchmark settings differ from this run; rerun the benchmark.")
        expected = set(range(shard, len(records), config["num_shards"]))
        bucket_ids = [i for observation in report["observations"] for i in observation["bucket_ids"]]
        if len(bucket_ids) != len(expected) or set(bucket_ids) != expected:
            raise ValueError("Benchmark buckets do not partition this shard.")
        counts.add(report["parameter_count_per_encoder"])
        seconds = sum(
            len(pending.intersection(observation["bucket_ids"])) * observation["total_seconds"]
            for observation in report["observations"]
        )
        if expected.intersection(pending):
            seconds += report["startup_warmup_seconds"]
        estimates.append(seconds)
    if len(counts) != 1:
        raise ValueError("Workers counted different encoder architectures.")
    return {
        "parameters_per_encoder": counts.pop(),
        "optimized_parameters": 0,
        "gpus": [report["gpu"] for report in reports],
        "weight_gib_per_gpu": [report["weight_bytes_per_gpu"] / 2**30 for report in reports],
        "observations": [dict(shard=report["shard"], **item)
                         for report in reports for item in report["observations"]],
        "remaining_samples": len(pending),
        "estimated_seconds_per_shard": estimates,
        "estimated_wall_seconds": max(estimates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", required=True, type=int)
    args = parser.parse_args()
    benchmark(args.output, args.shard)


if __name__ == "__main__":
    main()
