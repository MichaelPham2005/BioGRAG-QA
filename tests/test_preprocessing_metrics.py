import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preprocessing_metrics", ROOT / "src/utils/notebook_preprocessing/preprocessing_metrics.py"
)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class PreprocessingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        (self.output / "samples").mkdir()
        (self.output / "benchmark").mkdir()
        (self.output / "config.json").write_text("{}")
        self.records = [
            {"id": i, "nodes": i + 2, "edges": i + 1,
             "query": f"query {i}", "question": f"question {i}", "label": "support"}
            for i in range(8)
        ]
        self.config = {
            "num_shards": 2, "prepared": str(self.output), "dataset": "expla_graphs",
            "k": 2, "topk": 20, "topk_entity": 10,
        }
        self.read = patch.object(metrics.pipeline, "read_config",
                                 return_value=(self.config, {}, self.records))
        self.read.start()

    def tearDown(self):
        self.read.stop()
        self.temporary.cleanup()

    def reports(self):
        for shard in range(2):
            report = {
                "config_sha256": metrics.pipeline.digest(self.output / "config.json"),
                "shard": shard, "num_shards": 2, "gpu": "T4 fixture",
                "parameter_count_per_encoder": 100, "weight_bytes_per_gpu": 400,
                "startup_warmup_seconds": 10,
                "observations": [
                    {"sample_id": shard, "bucket_ids": [shard, shard + 2],
                     "total_seconds": 2 + shard},
                    {"sample_id": shard + 4, "bucket_ids": [shard + 4, shard + 6],
                     "total_seconds": 5 - shard},
                ],
            }
            (self.output / "benchmark" / f"shard-{shard}.json").write_text(json.dumps(report))

    def test_size_buckets_partition_modulo_shards(self):
        all_ids = []
        for shard in range(2):
            buckets = metrics.size_buckets(self.records, shard, 2)
            self.assertEqual(len(buckets), 3)
            ids = [i for bucket in buckets for i in bucket]
            self.assertEqual(sorted(ids), list(range(shard, 8, 2)))
            all_ids.extend(ids)
        self.assertEqual(sorted(all_ids), list(range(8)))

    def test_estimate_uses_remaining_queries_and_slowest_shard_not_perfect_speedup(self):
        self.reports()
        (self.output / "samples/0.pt").touch()
        summary = metrics.estimate_remaining(self.output)
        self.assertEqual(summary["remaining_samples"], 7)
        self.assertEqual(summary["parameters_per_encoder"], 100)
        self.assertEqual(summary["estimated_seconds_per_shard"], [22, 24])
        self.assertEqual(summary["estimated_wall_seconds"], 24)
        for index in range(8):
            (self.output / "samples" / f"{index}.pt").touch()
        self.assertEqual(metrics.estimate_remaining(self.output)["estimated_wall_seconds"], 0)

    def test_changed_config_is_rejected(self):
        self.reports()
        (self.output / "config.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "settings differ"):
            metrics.estimate_remaining(self.output)

    def test_benchmark_counts_actual_weights_and_never_replaces_sample_cache(self):
        class FakeEncoder:
            def __init__(self, config, device):
                self.model = torch.nn.Linear(2, 2)

            def __call__(self, texts):
                value = torch.zeros((len(texts), 1024))
                value[:, 0] = 1
                return value

        nodes, edges = metrics.pipeline.text_tables([("a", "r", "b")], "expla_graphs")
        edges = edges.reset_index(names="source_edge_id")
        (self.output / "samples/0.pt").write_bytes(b"existing sample must not change")
        with patch.object(metrics.pipeline, "Encoder", FakeEncoder), \
                patch.object(metrics.pipeline, "load_tables", return_value=(nodes, edges)), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_name", return_value="T4 fixture"), \
                patch.object(torch.cuda, "synchronize"):
            report = metrics.benchmark(self.output, 0)
        self.assertEqual(report["parameter_count_per_encoder"], 6)
        self.assertEqual(report["weight_bytes_per_gpu"], 24)
        self.assertEqual(report["optimized_parameters"], 0)
        self.assertEqual(len(report["observations"]), 3)
        for observation in report["observations"]:
            self.assertGreaterEqual(observation["embedding_seconds"], 0)
            self.assertGreaterEqual(observation["retrieval_seconds"], 0)
            self.assertGreater(observation["serialized_bytes"], 0)
        self.assertEqual((self.output / "samples/0.pt").read_bytes(),
                         b"existing sample must not change")
        self.assertEqual([path.name for path in (self.output / "samples").iterdir()], ["0.pt"])
        self.assertTrue((self.output / "benchmark/shard-0.json").is_file())


if __name__ == "__main__":
    unittest.main()
