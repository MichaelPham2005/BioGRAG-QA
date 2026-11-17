import importlib.util
import io
import ast
from contextlib import redirect_stdout
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kaggle_preprocess", ROOT / "src/utils/notebook_preprocessing/kaggle_preprocess.py"
)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class CompactRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.edges = pipeline.text_tables(
            [("a", "r", "b"), ("b", "s", "c"), ("c", "r", "d")], "expla_graphs"
        )
        self.edges = self.edges.reset_index(names="source_edge_id")
        self.x = torch.eye(4, 1024)
        self.relations = torch.eye(2, 1024)
        self.relation_ids = torch.tensor([0, 1, 0])
        self.query = self.x[2]
        self.config = {"k": 1, "topk": 3, "topk_entity": 2}

    def retrieve(self, config=None):
        payload = pipeline.retrieve_compact(
            self.nodes, self.edges, self.x, self.relations, self.relation_ids,
            self.query, self.config if config is None else config,
        )
        payload.update(id=0, question="question", label="support")
        return payload

    def test_round_trip_matches_explicit_ego_concatenation(self):
        payload = self.retrieve()
        sample = pipeline.materialize(payload)
        graph = sample["graph"]
        edge_index = torch.tensor([self.edges.src.tolist(), self.edges.dst.tolist()])
        xs, es, indices, offset, expected_scores = [], [], [], 0, []
        for center in range(len(self.nodes)):
            subset, _, _, mask = k_hop_subgraph(
                center, 1, edge_index, relabel_nodes=True, num_nodes=4
            )
            pooled = torch.cat((self.x[subset], self.relations[self.relation_ids[mask]])).mean(0)
            expected_scores.append(torch.nn.functional.cosine_similarity(self.query, pooled, dim=0))
        torch.testing.assert_close(payload["scores"], torch.stack(expected_scores))
        for center in payload["centers"].tolist():
            subset, edges, _, mask = k_hop_subgraph(
                center, 1, edge_index, relabel_nodes=True, num_nodes=4
            )
            xs.append(self.x[subset])
            es.append(self.relations[self.relation_ids[mask]])
            indices.append(edges + offset)
            offset += len(subset)
        torch.testing.assert_close(graph.x, torch.cat(xs))
        torch.testing.assert_close(graph.edge_attr, torch.cat(es))
        torch.testing.assert_close(graph.edge_index, torch.cat(indices, dim=1))
        torch.testing.assert_close(graph.question_node, self.query.repeat(offset, 1))
        self.assertLess(len(payload["node_features"]), len(graph.x))
        self.assertLessEqual(len(payload["edge_features"]), 2)
        self.assertEqual(graph.x.dtype, torch.float32)

    def test_single_center_edgeless_and_oversized_topk(self):
        for topk in (1, 20):
            payload = self.retrieve({"k": 0, "topk": topk, "topk_entity": 10})
            graph = pipeline.materialize(payload)["graph"]
            self.assertEqual(graph.edge_index.shape, (2, 0))
            self.assertEqual(graph.edge_attr.shape, (0, 1024))
            self.assertEqual(graph.question_edge.shape, (0, 1024))
            self.assertEqual(graph.question_node.shape, graph.x.shape)
            self.assertEqual(len(payload["centers"]), min(topk, 4))

    def test_truly_edgeless_graph_and_isolated_node(self):
        self.edges = self.edges.iloc[:0]
        self.relations = torch.empty((0, 1024))
        self.relation_ids = torch.empty(0, dtype=torch.long)
        graph = pipeline.materialize(self.retrieve())["graph"]
        self.assertEqual(graph.edge_index.shape, (2, 0))
        self.assertEqual(graph.x.shape, (3, 1024))

    def test_duplicate_endpoint_filter_keeps_matching_relation_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nodes").mkdir()
            (root / "edges").mkdir()
            nodes, edges = pipeline.text_tables(
                [("a", "remove-1", "b"), ("a", "remove-2", "b"), ("b", "keep", "c")],
                "webqsp",
            )
            nodes.to_csv(root / "nodes/0.csv", index=False)
            edges.to_csv(root / "edges/0.csv", index=False)
            record = {"id": 0, "nodes_sha256": pipeline.digest(root / "nodes/0.csv"),
                      "edges_sha256": pipeline.digest(root / "edges/0.csv")}
            _, filtered = pipeline.load_tables(root, record, "webqsp")
            self.assertEqual(filtered.edge_attr.tolist(), ["keep"])
            self.assertEqual(filtered.source_edge_id.tolist(), [2])
            _, unfiltered = pipeline.load_tables(root, record, "expla_graphs")
            self.assertEqual(len(unfiltered), 3)

    def test_numeric_entity_and_relation_strings_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nodes").mkdir()
            (root / "edges").mkdir()
            nodes, edges = pipeline.text_tables([("001", "002", "003")], "webqsp")
            nodes.to_csv(root / "nodes/0.csv", index=False)
            edges.to_csv(root / "edges/0.csv", index=False)
            record = {"id": 0, "nodes_sha256": pipeline.digest(root / "nodes/0.csv"),
                      "edges_sha256": pipeline.digest(root / "edges/0.csv")}
            nodes, edges = pipeline.load_tables(root, record, "webqsp")
            self.assertEqual(nodes.node_attr.tolist(), ["001", "003"])
            self.assertEqual(edges.edge_attr.tolist(), ["002"])

    def test_corrupt_feature_and_endpoint_indices_are_rejected(self):
        payload = self.retrieve()
        payload["edge_index"][0, 0] = len(payload["node_ids"])
        with self.assertRaisesRegex(ValueError, "endpoints"):
            pipeline.materialize(payload)
        payload = self.retrieve()
        payload["node_ids"][0] = len(payload["node_features"])
        with self.assertRaisesRegex(ValueError, "node_ids"):
            pipeline.materialize(payload)

    def test_compact_serialization_removes_repeated_feature_storage(self):
        triples = [(f"n{i}", "relation", f"n{j}") for i in range(10) for j in range(10) if i != j]
        self.nodes, self.edges = pipeline.text_tables(triples, "expla_graphs")
        self.edges = self.edges.reset_index(names="source_edge_id")
        self.x, self.relations = torch.eye(10, 1024), torch.eye(1, 1024)
        self.relation_ids = torch.zeros(len(triples), dtype=torch.long)
        payload = self.retrieve({"k": 2, "topk": 20, "topk_entity": 10})
        compact, expanded = io.BytesIO(), io.BytesIO()
        torch.save(payload, compact)
        torch.save(pipeline.materialize(payload), expanded)
        self.assertLess(compact.tell() * 10, expanded.tell())
        restored = torch.load(io.BytesIO(compact.getvalue()), weights_only=True)
        torch.testing.assert_close(pipeline.materialize(restored)["graph"].x,
                                   pipeline.materialize(payload)["graph"].x)


class EncoderTests(unittest.TestCase):
    def test_batched_masked_pooling_deduplication_and_normalization(self):
        from transformers import BatchEncoding, RobertaConfig, RobertaModel

        calls = []

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                calls.append((texts, kwargs))
                rows = [[1, 3, 2] if text == "short" else [1, 4, 5, 2] for text in texts]
                width = max(map(len, rows))
                return BatchEncoding({
                    "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                    "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
                })

        torch.manual_seed(42)
        model = RobertaModel(RobertaConfig(
            vocab_size=10, hidden_size=1024, num_hidden_layers=1,
            num_attention_heads=16, intermediate_size=16,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        )).eval()
        config = {"model": pipeline.MODEL, "model_revision": "local-test",
                  "batch_size": 1, "max_length": 512, "fp16": False}
        with patch("transformers.AutoModel.from_pretrained", return_value=model), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
            encoder = pipeline.Encoder(config, "cpu")
            output = encoder(["short", "long", "short"])
        self.assertEqual([texts for texts, _ in calls], [["short"], ["long"]])
        self.assertTrue(all(settings["max_length"] == 512 for _, settings in calls))
        self.assertEqual(output.shape, (3, 1024))
        torch.testing.assert_close(output[0], output[2])
        torch.testing.assert_close(output.norm(dim=1), torch.ones(3))
        tokens = Tokenizer()(["short", "long"])
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            expected = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1)
        torch.testing.assert_close(output[:2], expected, atol=1e-5, rtol=1e-4)


class StreamingTests(unittest.TestCase):
    def test_webqsp_preparation_preserves_split_offsets_without_embedding(self):
        class Rows(list):
            _fingerprint = "fixture"

        source = {
            "train": Rows([{"question": "first", "answer": ["A"],
                            "graph": [("Head", "Relation", "Tail")]}]),
            "validation": Rows([{"question": "second", "answer": ["B", "C"],
                                 "graph": [("X", "R", "Y")]}]),
            "test": Rows([{"question": "third", "answer": ["D"],
                           "graph": [("K", "R", "L")]}]),
        }
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict("sys.modules", {"datasets": types.SimpleNamespace(
                    load_dataset=lambda *args, **kwargs: source)}):
            pipeline.prepare("webqsp", temporary, revision="fixture-revision")
            metadata, records = pipeline.read_prepared(temporary)
            self.assertEqual(metadata["splits"], {"train": [0], "val": [1], "test": [2]})
            self.assertEqual([row["query"] for row in records], ["first", "second", "third"])
            self.assertEqual(records[1]["label"], "b|c")
            nodes, edges = pipeline.load_tables(temporary, records[0], "webqsp")
            self.assertEqual(nodes.node_attr.tolist(), ["head", "tail"])
            self.assertEqual(edges.edge_attr.tolist(), ["Relation"])

    def test_exclusive_shard_lock_rejects_duplicate_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker-0.lock"
            with pipeline.exclusive_lock(path):
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    with pipeline.exclusive_lock(path):
                        self.fail("Duplicate worker acquired a locked shard")
            with pipeline.exclusive_lock(path):
                pass

    def test_prepare_shards_resume_and_batched_model_input(self):
        class FakeEncoder:
            def __init__(self, config, device):
                self.revision = "test-model"

            def __call__(self, texts):
                output = torch.zeros((len(texts), 1024))
                for index, text in enumerate(texts):
                    output[index, sum(text.encode()) % 1024] = 1
                return output

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, output = root / "train.tsv", root / "prepared", root / "run"
            pd.DataFrame([
                {"arg1": f"argument {i}", "arg2": f"response {i}", "label": "support",
                 "graph": "(a; relates; b)(b; causes; c)"}
                for i in range(10)
            ]).to_csv(raw, sep="\t", index=False)
            pipeline.prepare("expla_graphs", prepared, raw)
            pipeline.configure_run(prepared, output, model_revision="test-model")
            _, records = pipeline.read_prepared(prepared)
            self.assertIn("Argument 1: argument 0\nArgument 2: response 0", records[0]["query"])
            self.assertNotEqual(records[0]["query"], records[1]["query"])
            with patch.object(pipeline, "Encoder", FakeEncoder), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "get_device_name", return_value="fake GPU"), \
                    patch.object(pipeline.shutil, "disk_usage", return_value=type("Disk", (), {"free": 20 * 2**30})()):
                pipeline.worker(output, 0, stop=2)
                self.assertEqual(sorted(p.name for p in (output / "samples").glob("*.pt")), ["0.pt"])
                pipeline.worker(output, 1, stop=2)
                original = (output / "samples/0.pt").read_bytes()
                for shard in range(2):
                    pipeline.worker(output, shard)
                self.assertEqual((output / "samples/0.pt").read_bytes(), original)
            pipeline.validate_run(output)
            dataset = pipeline.CompactDataset(output)
            self.assertEqual({k: len(v) for k, v in dataset.get_idx_split().items()},
                             {"train": 6, "val": 2, "test": 2})
            batch = Batch.from_data_list([dataset[0]["graph"], dataset[1]["graph"]])
            result = GATConv(1024, 4, edge_dim=1024)(batch.x, batch.edge_index, batch.edge_attr)
            self.assertEqual(result.shape, (batch.num_nodes, 4))
            with self.assertRaisesRegex(ValueError, "settings changed"):
                pipeline.configure_run(prepared, output, topk=1, model_revision="test-model")
            (prepared / "nodes/0.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Text table changed"):
                pipeline.load_tables(prepared, records[0], "expla_graphs")


class NotebookLauncherTests(unittest.TestCase):
    def test_notebook_launcher_assigns_gpus_and_stops_its_workers_on_failure(self):
        notebook = json.loads((ROOT / "notebook/kaggle/grag-webqsp-data-processing.ipynb").read_text())
        definition = next(
            node for cell in notebook["cells"] if cell["cell_type"] == "code"
            for node in ast.parse("".join(cell["source"])).body
            if isinstance(node, ast.FunctionDef) and node.name == "run_shards"
        )
        real_popen, real_sleep = subprocess.Popen, time.sleep
        processes = []

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "samples").mkdir()
            script = root / "worker.py"
            script.write_text(
                "import os, pathlib, sys, time\n"
                "shard = sys.argv[sys.argv.index('--shard') + 1]\n"
                "root = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "(root / f'gpu-{shard}').write_text(os.environ['CUDA_VISIBLE_DEVICES'])\n"
                "if (root / 'fail').exists():\n"
                "    if shard == '1':\n"
                "        raise SystemExit(7)\n"
                "    time.sleep(60)\n"
                "(root / 'samples' / f'{shard}.pt').write_text('finished')\n"
            )
            namespace = {
                "subprocess": subprocess, "os": os, "PY": sys.executable,
                "SCRIPT": script, "ROOT": root, "RUN": root,
                "config": {"num_shards": 2}, "records": [0, 1],
            }
            exec(compile(ast.Module(body=[definition], type_ignores=[]), "<notebook launcher>", "exec"),
                 namespace)
            with patch.object(subprocess, "Popen", side_effect=capture_process), \
                    patch.object(time, "sleep", side_effect=lambda delay: real_sleep(0.01)), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(namespace["run_shards"](stop=2), [0, 0])
                self.assertEqual((root / "gpu-0").read_text(), "0")
                self.assertEqual((root / "gpu-1").read_text(), "1")
                metrics_script = root / "metrics.py"
                metrics_script.write_text(script.read_text())
                self.assertEqual(namespace["run_shards"](script=metrics_script), [0, 0])
                self.assertTrue((root / "benchmark-shard-0.log").is_file())
                self.assertTrue((root / "benchmark-shard-1.log").is_file())
                (root / "fail").touch()
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    namespace["run_shards"]()
            self.assertTrue(all(process.poll() is not None for process in processes))


if __name__ == "__main__":
    unittest.main()
