import importlib.util
import io
import ast
from contextlib import redirect_stdout
import json
from pathlib import Path
import os
import re
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


class FullGraphTests(unittest.TestCase):
    def setUp(self):
        from src.utils.notebook_preprocessing import expla_full_graph
        from src.utils.notebook_downstream import artifacts

        self.full_graph, self.artifacts = expla_full_graph, artifacts
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared, self.run = self.root / "prepared", self.root / "full-run"
        # A long chain, a parallel edge, an exact repeated edge and a self-loop.
        self.graph = "".join(f"(N{i}; R; N{i+1})" for i in range(13))
        self.graph += "(N0; S; N1)(N0; R; N1)(N2; R; N2)"
        self.raw = self.root / "train_dev.tsv"
        pd.DataFrame([
            {"arg1": f"first {i}", "arg2": f"second {i}",
             "label": "support" if i % 2 else "counter", "graph": self.graph}
            for i in range(10)
        ]).to_csv(self.raw, sep="\t", index=False)
        pipeline.prepare("expla_graphs", self.prepared, raw=self.raw)
        self.config = expla_full_graph.configure_run(
            self.prepared, self.run, model_revision="a" * 40,
        )
        self.metadata, self.records = pipeline.read_prepared(self.prepared)
        self.config_hash = pipeline.digest(self.run / "config.json")

    @staticmethod
    def encode(texts):
        ids = torch.tensor([sum(text.encode()) % 1024 for text in texts], dtype=torch.long)
        return torch.nn.functional.one_hot(ids, num_classes=1024).float()

    def write_samples(self):
        for record in self.records:
            nodes, edges = pipeline.load_tables(self.prepared, record, "expla_graphs")
            payload = self.full_graph.encode_record(
                record, nodes, edges, self.encode, self.config_hash, "a" * 40,
            )
            torch.save(payload, self.run / "samples" / f"{record['id']}.pt")
        return self.artifacts.CompactTrainingDataset(self.prepared, self.run)

    def test_full_graph_matches_author_tables_description_splits_and_per_row_query(self):
        # Execute only the author's pure textualizer, avoiding its top-level I/O/model imports.
        author = ast.parse((ROOT / "src/dataset/preprocess/expla_graphs.py").read_text())
        textualizer = next(node for node in author.body
                           if isinstance(node, ast.FunctionDef) and node.name == "textualize_graph")
        namespace = {"pd": pd, "re": re}
        exec(compile(ast.Module(body=[textualizer], type_ignores=[]), "<author-textualizer>", "exec"),
             namespace)
        author_nodes, author_edges = namespace["textualize_graph"](self.graph)
        author_split = self.artifacts.load_source("full_graph_author_split", ROOT / "src/utils/generate_split.py")
        author_split.generate_split(10, self.root / "author-splits")
        for name in ("train", "val", "test"):
            expected = [int(line) for line in
                        (self.root / "author-splits" / f"{name}_indices.txt").read_text().splitlines()]
            self.assertEqual(self.metadata["splits"][name], expected)
        with patch.object(pipeline, "retrieve_compact", side_effect=AssertionError("No retrieval")), \
                patch("src.utils.notebook_preprocessing.kaggle_preprocess.retrieve_compact",
                      side_effect=AssertionError("No retrieval")):
            dataset = self.write_samples()
            self.full_graph.validate_dataset(dataset)
        sample = dataset[0]
        graph = sample["graph"]
        expected_description = (author_nodes.to_csv(index=False, columns=["node_id", "node_attr"])
                                + "\n" + author_edges.to_csv(index=False))
        self.assertEqual(sample["desc"], expected_description)
        self.assertEqual((graph.num_nodes, graph.num_edges), (14, 16))
        torch.testing.assert_close(graph.edge_index, torch.tensor([author_edges.src, author_edges.dst]))
        torch.testing.assert_close(graph.x, self.encode(author_nodes.node_attr.tolist()))
        torch.testing.assert_close(graph.edge_attr, self.encode(author_edges.edge_attr.tolist()))
        torch.testing.assert_close(graph.question_node, self.encode([self.records[0]["question"]]).repeat(14, 1))
        self.assertFalse(torch.equal(dataset[0]["graph"].question_node[0], dataset[1]["graph"].question_node[0]))
        self.assertEqual(dataset.config["pipeline"], "expla_full_graph")
        self.assertEqual(self.metadata["excluded_count"], 0)

    def test_full_graph_snapshot_round_trip_and_mutation_rejection(self):
        from src.utils.notebook_preprocessing.drive_backup import collect_files, write_archive

        self.write_samples()
        manifest, files = collect_files(self.prepared, self.run)
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        archive = snapshot / "part-00000.tar.gz"
        write_archive(files, archive)
        manifest["archives"] = [{"name": archive.name, "size": archive.stat().st_size,
                                 "sha256": pipeline.digest(archive)}]
        self.artifacts.atomic_json(manifest, snapshot / "manifest.json")
        prepared, run, _ = self.artifacts.restore_snapshot(snapshot, self.root / "restored", "expla_graphs")
        restored = self.artifacts.CompactTrainingDataset(prepared, run)
        self.full_graph.validate_dataset(restored)
        changed = restored.read_payload(0)
        changed["desc"] = "only some nodes"
        torch.save(changed, run / "samples/0.pt")
        with self.assertRaisesRegex(ValueError, "lost, reordered or changed"):
            self.full_graph.validate_dataset(restored)
        changed["pipeline"] = "retrieval"
        nodes, edges = pipeline.load_tables(prepared, self.records[0], "expla_graphs")
        with self.assertRaisesRegex(ValueError, "lost, reordered or changed"):
            self.full_graph.validate_record(changed, self.records[0], nodes, edges,
                                            self.config_hash, "a" * 40)

    def test_resume_rejects_settings_or_source_changes(self):
        self.assertEqual(self.full_graph.configure_run(
            self.prepared, self.run, model_revision="a" * 40,
        ), self.config)
        with self.assertRaisesRegex(ValueError, "settings/data/code changed"):
            self.full_graph.configure_run(self.prepared, self.run, model_revision="b" * 40)
        with patch.object(self.full_graph.preprocessing, "digest", return_value="changed-code"):
            with self.assertRaisesRegex(ValueError, "original full-graph preprocessing code"):
                self.full_graph.worker(self.run, 0)

    def test_notebook_receipt_restores_exact_snapshot_without_reembedding(self):
        from src.utils.notebook_downstream import drive_sync, training
        from src.utils.notebook_preprocessing import drive_backup
        from tests.test_drive_sync import Download, MutableDrive

        class Media(bytes):
            def execute(self):
                return bytes(self)

        class NotebookDrive(MutableDrive):
            def get_media(self, fileId):
                return Media(super().get_media(fileId))

        class LocalEncoder:
            revision = "a" * 40

            def __init__(self, config, device):
                pass

            __call__ = staticmethod(self.encode)

        checkout = self.root / "checkout"
        self.artifacts.backup_file(self.raw, checkout / "dataset/expla_graphs/train_dev.tsv")
        drive = NotebookDrive()
        store = drive_sync.DriveFolder(drive, "training", self.root / "drive-cache")
        output = checkout / "output/downstream/expla_graphs/full-graph"
        output.mkdir(parents=True)
        spawned = []

        def launch(command, **kwargs):
            shard = int(command[-1])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(shard))
            spawned.append(shard)
            self.full_graph.worker(Path(command[command.index("--output") + 1]), shard)
            return types.SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)

        def notebook_path(value):
            # Keep the notebook's Kaggle caches inside this test's temporary directory.
            if str(value).startswith("/kaggle/temp/"):
                return self.root / str(value).removeprefix("/")
            return Path(value)

        notebook = json.loads((ROOT / "notebook/kaggle/grag-expla-graphs-full-graph.ipynb").read_text())
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "preprocessing")
        code = compile("".join(cell["source"]), "<full-graph-preprocessing-cell>", "exec")
        namespace = {
            "Path": notebook_path, "ROOT": checkout, "OUTPUT": output, "DRIVE_OUTPUT": store.cache,
            "drive_store": store, "service": drive, "DATASET": "expla_graphs", "SMOKE_TEST": True,
            "full_graph": self.full_graph, "preprocessing": self.full_graph.preprocessing,
            "artifacts": self.artifacts, "training": training, "json": json, "os": os, "sys": sys,
            "subprocess": types.SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT),
            "HfApi": lambda: types.SimpleNamespace(model_info=lambda model: types.SimpleNamespace(sha="a" * 40)),
            "COMMIT": "test-code", "requirements": {}, "version": lambda name: "test",
            "backup_to_drive": drive_backup.backup_to_drive, "find_remote": drive_backup.find_remote,
            "restore_drive_snapshot": drive_sync.restore_drive_snapshot,
        }
        with patch.object(drive_backup, "upload_media", side_effect=lambda path: path), \
                patch.object(drive_sync, "upload_media", side_effect=lambda path: path), \
                patch("googleapiclient.http.MediaIoBaseDownload", Download), \
                patch.object(self.full_graph.preprocessing, "Encoder", LocalEncoder), \
                patch.object(torch.cuda, "is_available", return_value=True):
            exec(code, namespace)
            first_hash = namespace["dataset"].config_hash
            self.assertEqual(spawned, [0, 1])
            self.assertEqual(len(namespace["dataset"]), 10)
            receipt = json.loads((store.cache / "preprocessing.json").read_text())
            self.assertEqual(receipt["snapshot_sha256"], namespace["SNAPSHOT_HASH"])
            fresh = drive_sync.DriveFolder(drive, "training", self.root / "fresh-cache")
            fresh.download("preprocessing.json")
            namespace.update(DRIVE_OUTPUT=fresh.cache, drive_store=fresh)
            namespace["subprocess"].Popen = lambda *args, **kwargs: self.fail("Snapshot restore reran embeddings")
            exec(code, namespace)
            self.assertEqual(namespace["dataset"].config_hash, first_hash)
            self.assertEqual(namespace["SNAPSHOT_HASH"], receipt["snapshot_sha256"])


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
            "train": Rows([{"id": "train-first", "question": "first", "answer": ["A"],
                            "graph": [("Head", "Relation", "Tail")]}]),
            "validation": Rows([{"id": "val-second", "question": "second", "answer": ["B", "C"],
                                 "graph": [("X", "R", "Y")]}]),
            "test": Rows([{"id": "test-third", "question": "third", "answer": ["D"],
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

    def test_empty_webqsp_graphs_are_audited_without_losing_source_or_split_mapping(self):
        class Rows(list):
            _fingerprint = "fixture"

        def row(index, empty=False):
            return {"id": f"native-{index}", "question": f"question {index}",
                    "answer": ["Answer"], "graph": [] if empty else [("A", "R", "B")]}

        source = {
            "train": Rows([row(0, empty=True), row(1)]),
            "validation": Rows([row(2, empty=True), row(3)]),
            "test": Rows([row(4), row(5, empty=True)]),
        }
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict("sys.modules", {"datasets": types.SimpleNamespace(
                    load_dataset=lambda *args, **kwargs: source)}):
            root = Path(temporary)
            pipeline.prepare("webqsp", root, revision="fixture-revision")
            metadata, records = pipeline.read_prepared(root)
            excluded = json.loads((root / "excluded.json").read_text())
            self.assertEqual((metadata["source_count"], metadata["count"], metadata["excluded_count"]),
                             (6, 3, 3))
            self.assertEqual(metadata["splits"], {"train": [0], "val": [1], "test": [2]})
            self.assertEqual(metadata["source_splits"], {"train": [0, 1], "val": [2, 3], "test": [4, 5]})
            self.assertEqual([record["id"] for record in records], [0, 1, 2])
            self.assertEqual([record["source_index"] for record in records], [1, 3, 4])
            self.assertEqual([record["source_id"] for record in records],
                             ["native-1", "native-3", "native-4"])
            self.assertEqual(records[1]["source_split"], "val")
            self.assertEqual(records[1]["source_split_index"], 1)
            self.assertEqual([record["source_index"] for record in excluded], [0, 2, 5])
            self.assertTrue(all(record["reason"] == "empty_graph" for record in excluded))
            self.assertEqual(sorted(path.name for path in (root / "nodes").glob("*.csv")),
                             ["0.csv", "1.csv", "2.csv"])
            (root / "excluded.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Exclusion report changed"):
                pipeline.read_prepared(root)

    def test_all_empty_webqsp_graphs_fail_with_audit_instead_of_an_empty_dataset(self):
        class Rows(list):
            _fingerprint = "fixture"

        source = {split: Rows([{"id": split, "question": "question", "answer": ["a"], "graph": []}])
                  for split in ("train", "validation", "test")}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict("sys.modules", {"datasets": types.SimpleNamespace(
                    load_dataset=lambda *args, **kwargs: source)}):
            with self.assertRaisesRegex(ValueError, "No nonempty graphs"):
                pipeline.prepare("webqsp", temporary, revision="fixture-revision")
            self.assertEqual(len(json.loads((Path(temporary) / "excluded.json").read_text())), 3)
            self.assertFalse((Path(temporary) / "prepared.json").exists())

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
            self.assertEqual(dataset[0]["source_index"], 0)
            self.assertEqual(dataset[0]["source_id"], None)
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
