import ast
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.notebook_downstream import artifacts, training
from src.utils.notebook_preprocessing import kaggle_preprocess as preprocessing


class Tokenizer:
    pad_token_id, eos_token_id = 0, 2

    def encode(self, text, add_special_tokens=False):
        fixed = {"<s>[INST]": [1, 3], "[/INST]": [4], "support": [5], "counter": [6]}
        return fixed.get(text, [10 + ord(char) % 70 for char in text])

    def batch_decode(self, values, skip_special_tokens=True):
        self.decoded_width = values.shape[1]
        return [" ".join(str(token) for token in row.tolist()) for row in values]


class ScalarModel(nn.Module):
    def __init__(self, settings, interrupt=False):
        super().__init__()
        self.settings = settings
        self.weight = nn.Parameter(torch.tensor(0.))
        self.device = torch.device("cpu")
        self.calls = 0
        self.interrupt = interrupt

    def forward(self, samples):
        if self.training:
            self.calls += 1
            if self.interrupt and self.calls == 3:
                raise RuntimeError("Injected interruption after the first checkpoint.")
        return (self.weight - samples["id"][0] - 1).square()

    def learned_state(self):
        return {"weight": self.weight.detach().clone()}

    def load_learned_state(self, state):
        self.load_state_dict(state)


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared, self.run = self.root / "prepared", self.root / "run"
        self.prepared.mkdir()
        (self.run / "samples").mkdir(parents=True)
        self.splits = {"train": [0, 1, 2], "val": [3], "test": [4]}
        records = [{"id": i, "question": f"question {i}", "label": "support"}
                   for i in range(5)]
        artifacts.atomic_json(records, self.prepared / "records.json")
        artifacts.atomic_json({
            "dataset": "expla_graphs", "count": 5, "splits": self.splits,
            "records_sha256": artifacts.checksum(self.prepared / "records.json"),
        }, self.prepared / "prepared.json")
        # A previous producer implementation and a path that does not exist in Colab.
        artifacts.atomic_json({
            "format": 1, "dataset": "expla_graphs", "count": 5,
            "prepared": "/kaggle/working/old-checkout/output/prepared",
            "prepared_sha256": artifacts.checksum(self.prepared / "prepared.json"),
            "implementation_sha256": "earlier-producer-source", "model_revision": "encoder-sha",
        }, self.run / "config.json")
        nodes, edges = preprocessing.text_tables([("a", "r", "b")], "expla_graphs")
        edges = edges.reset_index(names="source_edge_id")
        for row in records:
            payload = preprocessing.retrieve_compact(
                nodes, edges, torch.randn(2, 1024), torch.randn(1, 1024),
                torch.tensor([0]), torch.randn(1024), {"k": 1, "topk": 2, "topk_entity": 2},
            )
            payload.update(row, config_sha256=artifacts.checksum(self.run / "config.json"),
                           model_revision="encoder-sha")
            torch.save(payload, self.run / "samples" / f"{row['id']}.pt")
        self.dataset = artifacts.CompactTrainingDataset(self.prepared, self.run)
        self.settings = training.Settings(
            model_revision="model-sha", epochs=1, grad_accum=2, checkpoint_steps=1,
            learning_rate=0.05, weight_decay=0., max_sequence_tokens=96,
            gnn_hidden=8, gnn_layers=2, gnn_heads=2, projector_hidden=12,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot(self, extra=None, omit=None):
        snapshot = self.root / "snapshot"
        snapshot.mkdir(exist_ok=True)
        entries = []
        part = snapshot / "part-00000.tar.gz"
        with tarfile.open(part, "w:gz") as archive:
            for folder in (self.prepared, self.run):
                for path in sorted(folder.rglob("*")):
                    if not path.is_file():
                        continue
                    name = f"output/{folder.name}/{path.relative_to(folder).as_posix()}"
                    entries.append({"path": name, "size": path.stat().st_size,
                                    "sha256": artifacts.checksum(path)})
                    if name != omit:
                        archive.add(path, arcname=name, recursive=False)
            if extra:
                info = tarfile.TarInfo(extra)
                info.size = 3
                archive.addfile(info, io.BytesIO(b"bad"))
        manifest = {
            "format": 1, "dataset": "expla_graphs", "sample_count": 5,
            "restore_paths": {"prepared": "/kaggle/output/prepared", "run": "/kaggle/output/run"},
            "files": entries,
            "archives": [{"name": part.name, "size": part.stat().st_size,
                          "sha256": artifacts.checksum(part)}],
        }
        artifacts.atomic_json(manifest, snapshot / "manifest.json")
        return snapshot

    def test_restore_old_producer_and_verify_samples(self):
        snapshot = self.snapshot()
        prepared, run, _ = artifacts.restore_snapshot(snapshot, self.root / "restored", "expla_graphs")
        restored = artifacts.CompactTrainingDataset(prepared, run)
        self.assertEqual(restored[0]["label"], "support")
        self.assertNotIn("source_id", restored[0])
        torch.testing.assert_close(restored[0]["graph"].x, self.dataset[0]["graph"].x)
        # A repeated restore validates local file hashes, without needing the archives again.
        (snapshot / "part-00000.tar.gz").unlink()
        artifacts.restore_snapshot(snapshot, self.root / "restored", "expla_graphs")
        changed = torch.load(run / "samples/0.pt", weights_only=True)
        changed["config_sha256"] = "different-run"
        torch.save(changed, run / "samples/0.pt")
        with self.assertRaisesRegex(ValueError, "another run"):
            restored[0]
        with self.assertRaisesRegex(ValueError, "differs"):
            artifacts.restore_snapshot(snapshot, self.root / "restored", "expla_graphs")

    def test_restore_rejects_bad_hash_unknown_traversal_and_missing_members(self):
        for i, extra in enumerate(("../escaped", "output/run/unknown", "output/run/config.json")):
            with self.subTest(extra=extra):
                snapshot = self.snapshot(extra=extra)
                with self.assertRaises(ValueError):
                    artifacts.restore_snapshot(snapshot, self.root / f"bad-{i}", "expla_graphs")
        snapshot = self.snapshot(omit="output/run/samples/4.pt")
        with self.assertRaisesRegex(ValueError, "exactly"):
            artifacts.restore_snapshot(snapshot, self.root / "missing", "expla_graphs")
        snapshot = self.snapshot()
        (snapshot / "part-00000.tar.gz").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum/size"):
            artifacts.restore_snapshot(snapshot, self.root / "corrupt", "expla_graphs")

    def tiny_model(self):
        llm = LlamaForCausalLM(LlamaConfig(
            vocab_size=96, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
        ))
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=False)
        llm = get_peft_model(llm, LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", lora_dropout=0.,
        ))
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.config.use_cache = False
        return training.ColabGRAG(llm, Tokenizer(), self.settings)

    def test_tiny_llama_gradient_flow_masking_no_gold_leakage_and_checkpoint(self):
        model = self.tiny_model()
        samples = artifacts.collate([self.dataset[0], self.dataset[1]])
        original = samples["graph"].x.clone()
        model.eval()
        inputs = model.input_batch(samples, supervised=True)
        for labels in inputs["labels"]:
            self.assertEqual(labels[labels != -100].tolist(), [5, 2])
        inference = model.input_batch(samples, supervised=False)
        other = model.input_batch({**samples, "label": ["counter", "counter"]}, supervised=False)
        torch.testing.assert_close(inference["inputs_embeds"], other["inputs_embeds"])
        self.assertNotIn("labels", inference)
        model.train()
        loss = model(samples)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.projector[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.graph_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.llm.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.llm.parameters() if not p.requires_grad))
        torch.testing.assert_close(samples["graph"].x, original)
        model.eval()
        generated = model.predict(samples)
        self.assertEqual(len(generated), 2)
        self.assertLessEqual(model.tokenizer.decoded_width, self.settings.max_answer_tokens)
        model.train()
        path = self.root / "learned.pt"
        training.atomic_checkpoint(model.learned_state(), path)
        state = torch.load(path, weights_only=True)
        self.assertTrue(all("lora_" in key for key in state["lora"]))
        model.load_learned_state(state)
        one = self.dataset[0]
        graph = one["graph"]
        graph.x, graph.question_node = graph.x[:1], graph.question_node[:1]
        graph.num_nodes = 1
        graph.edge_index, graph.edge_attr = graph.edge_index[:, :0], graph.edge_attr[:0]
        graph.question_edge = graph.question_edge[:0]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            singleton_loss = model(artifacts.collate([one]))
        self.assertTrue(any("Singleton" in str(warning.message) for warning in captured))
        self.assertTrue(torch.isfinite(singleton_loss))
        self.assertTrue(all(module.training for module in model.graph_encoder.bns))

    def fit_scalar(self, model, name, norms=None):
        output, drive = self.root / name, self.root / f"{name}-drive"
        signature = training.initialize_run(
            self.settings, self.dataset, self.splits, "snapshot-sha", output, drive,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=self.settings.learning_rate)
        original_clip = torch.nn.utils.clip_grad_norm_

        def capture(parameters, *args, **kwargs):
            if norms is not None:
                norms.append(model.weight.grad.item())
            return original_clip(parameters, *args, **kwargs)

        with patch.object(training.torch.optim, "AdamW", return_value=optimizer):
            with patch.object(training.torch.nn.utils, "clip_grad_norm_", side_effect=capture):
                history = training.fit(model, self.dataset, self.splits, signature, output, drive)
        return output, drive, signature, history

    def test_accumulation_remainder_and_mid_epoch_resume(self):
        model = ScalarModel(self.settings)
        norms = []
        output, drive, signature, history = self.fit_scalar(model, "complete", norms)
        self.assertEqual(history[0]["optimizer_steps"], 2)
        # Seed 3 orders training IDs as [1, 2, 0]: targets 2, 3, then 1.
        self.assertAlmostEqual(norms[0], -5., places=5)
        self.assertAlmostEqual(norms[1], 2 * (0.05 * (2 / 3) * .1 - 1), places=5)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            self.fit_scalar(ScalarModel(self.settings, interrupt=True), "interrupted")
        interrupted = torch.load(self.root / "interrupted/last.pt", weights_only=True)
        self.assertEqual(interrupted["state"]["next_batch"], 2)
        resumed = ScalarModel(self.settings)
        self.fit_scalar(resumed, "interrupted")
        torch.testing.assert_close(model.weight, resumed.weight)
        restored = self.root / "restored-training"
        artifacts.restore_training_outputs(drive, restored)
        self.assertEqual(artifacts.checksum(restored / "last.pt"), artifacts.checksum(output / "last.pt"))
        with self.assertRaisesRegex(ValueError, "changed"):
            training.initialize_run(replace(self.settings, lora_rank=16), self.dataset,
                                    self.splits, "snapshot-sha", self.root / "fresh-local", drive)
        with self.assertRaisesRegex(ValueError, "overlap"):
            training.initialize_run(self.settings, self.dataset,
                                    {**self.splits, "test": [0]}, "sha", output, drive)

    def test_external_backup_callback_covers_initialization_training_and_prediction(self):
        output, drive = self.root / "remote-output", self.root / "remote-cache"
        backup = Mock(side_effect=artifacts.backup_file)
        signature = training.initialize_run(
            self.settings, self.dataset, self.splits, "snapshot", output, drive, backup=backup,
        )
        training.fit(ScalarModel(self.settings), self.dataset, self.splits, signature,
                     output, drive, backup=backup)
        names = {call.args[1].name for call in backup.call_args_list}
        self.assertTrue({"run.json", "last.pt", "best.pt", "history.json"}.issubset(names))
        model = self.tiny_model()
        training.save_and_backup({"signature": signature, "learned": model.learned_state()},
                                 "best.pt", output, drive, checkpoint=True, backup=backup)
        training.predict_test(model, self.dataset, self.splits["test"], signature,
                              output, drive, backup=backup)
        names = {call.args[1].name for call in backup.call_args_list}
        self.assertTrue({"prediction-config.json", "predictions.json", "metrics.json"}.issubset(names))
        with patch.object(model, "predict", side_effect=AssertionError("Unexpected generation")):
            training.predict_test(model, self.dataset, self.splits["test"], signature,
                                  output, drive, backup=backup)
        failing_backup = Mock(side_effect=OSError("Remote upload failed"))
        with self.assertRaisesRegex(OSError, "Remote upload failed"):
            training.save_and_backup({"kept": True}, "retained.json", output, drive,
                                     backup=failing_backup)
        self.assertTrue((output / "retained.json").is_file())

    def test_sharded_loader_forwards_map_without_changing_single_gpu_default(self):
        tokenizer = Tokenizer()
        llm = Mock()
        device_map = {"model.embed_tokens": 0, "model.layers.0": 0,
                      "model.layers.1": 1, "model.norm": 1, "lm_head": 1}
        with patch.object(training.torch.cuda, "is_available", return_value=True), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=llm) as load, \
                patch("transformers.BitsAndBytesConfig"), \
                patch("peft.prepare_model_for_kbit_training", return_value=llm), \
                patch("peft.get_peft_model", return_value=llm), \
                patch.object(training, "ColabGRAG", return_value=llm):
            training.load_qlora(self.settings, "test-token", device_map=device_map)
            self.assertEqual(load.call_args.kwargs["device_map"], device_map)
            training.load_qlora(self.settings, "test-token")
            self.assertEqual(load.call_args.kwargs["device_map"], {"": 0})

    def test_webqsp_extended_description_and_answer_budget(self):
        notebook = json.loads((ROOT / "notebook/kaggle/grag-webqsp-downstream.ipynb").read_text())
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "model")
        constructor = next(node for node in ast.walk(ast.parse("".join(cell["source"])))
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == "Settings")
        budgets = {keyword.arg: ast.literal_eval(keyword.value) for keyword in constructor.keywords
                   if keyword.arg.startswith("max_")}
        model = self.tiny_model()
        model.settings = replace(self.settings, dataset="webqsp", **budgets)
        model.eval()
        samples = artifacts.collate([self.dataset[0]])
        samples["desc"], samples["label"] = ["d" * 800], ["a" * 100]
        inputs = model.input_batch(samples, supervised=True)
        prefix = len(model.tokenizer.encode("<s>[INST]", add_special_tokens=False))
        suffix = len(model.tokenizer.encode("[/INST]", add_special_tokens=False))
        question = len(model.tokenizer.encode(samples["question"][0], add_special_tokens=False))
        self.assertEqual(inputs["inputs_embeds"].shape[1], prefix + 1 + 512 + question + suffix + 65)
        self.assertEqual(int((inputs["labels"] != -100).sum()), 65)
        samples["question"] = ["q" * 500]
        self.assertEqual(model.input_batch(samples, supervised=True)["inputs_embeds"].shape[1], 1024)

    def test_full_graph_notebook_keeps_description_and_rejects_any_truncation(self):
        from types import SimpleNamespace

        path = ROOT / "notebook/kaggle/grag-expla-graphs-full-graph.ipynb"
        notebook = json.loads(path.read_text())
        tree = ast.parse("".join(next(c for c in notebook["cells"] if c["id"] == "model")["source"]))
        constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "Settings")
        settings = eval(compile(ast.Expression(constructor), "<notebook-settings>", "eval"), {
            "training": training, "DATASET": "expla_graphs", "MODEL_ID": "local-test",
            "model_revision": "model-sha", "SMOKE_TEST": False,
            "config": SimpleNamespace(max_position_embeddings=4096),
        })
        guard = next(node for node in tree.body if isinstance(node, ast.If)
                     and "split_report" in ast.unparse(node.test))
        check = compile(ast.Module(body=[guard], type_ignores=[]), "<full-text-guard>", "exec")
        model = self.tiny_model()
        model.settings = settings
        samples = artifacts.collate([self.dataset[0]])
        samples["desc"] = ["d" * 700]
        row = {key: value[0] for key, value in samples.items() if key != "graph"}
        dataset = SimpleNamespace(read_payload=lambda index: row)
        report = training.token_report(model, dataset, [0])
        self.assertEqual(report, {"answer_targets_truncated": 0, "descriptions_truncated": 0})
        exec(check, {"report": {"train": report}})
        inputs = model.input_batch(samples, supervised=True)
        question = len(model.tokenizer.encode(samples["question"][0]))
        self.assertEqual(inputs["inputs_embeds"].shape[1], 2 + 1 + 700 + question + 1 + 2)
        row["desc"] = "d" * 4096
        report = training.token_report(model, dataset, [0])
        self.assertEqual(report["descriptions_truncated"], 1)
        with self.assertRaisesRegex(ValueError, "every graph to fit intact"):
            exec(check, {"report": {"test": report}})

    def test_colab_retrieval_snapshot_supports_untruncated_text_without_reprocessing(self):
        from types import SimpleNamespace

        config = {**self.dataset.config, "k": 2, "topk": 20, "topk_entity": 10}
        artifacts.atomic_json(config, self.run / "config.json")
        for index in range(len(self.dataset)):
            path = self.run / "samples" / f"{index}.pt"
            payload = torch.load(path, weights_only=True)
            payload.update(config_sha256=artifacts.checksum(self.run / "config.json"), desc="d" * 700)
            torch.save(payload, path)
        snapshot = self.snapshot()
        prepared, run, snapshot_hash = artifacts.restore_snapshot(snapshot, self.root / "colab", "expla_graphs")
        dataset = artifacts.CompactTrainingDataset(prepared, run)
        original_config_bytes = (run / "config.json").read_bytes()
        original_payload_bytes = (run / "samples/0.pt").read_bytes()
        self.assertEqual(dataset.config["prepared"], "/kaggle/working/old-checkout/output/prepared")
        self.assertEqual(dataset.get_idx_split(), self.splits)
        self.assertEqual(dataset[0]["desc"], "d" * 700)

        notebook = json.loads((ROOT / "notebook/colab/grag_expla_graphs_downstream.ipynb").read_text())
        cells = {cell["id"]: ast.parse("".join(cell["source"]))
                 for cell in notebook["cells"] if cell["cell_type"] == "code"}
        retrieval_guard = next(node for node in cells["cell-06"].body if isinstance(node, ast.If)
                               and "dataset.config" in ast.unparse(node.test))
        exec(compile(ast.Module(body=[retrieval_guard], type_ignores=[]), "<retrieval-guard>", "exec"),
             {"dataset": dataset})
        setting_call = next(node for node in ast.walk(cells["cell-10"]) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute) and node.func.attr == "Settings")
        run_name = next(node for node in cells["cell-08"].body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "RUN_NAME" for target in node.targets))
        guard = next(node for node in cells["cell-10"].body if isinstance(node, ast.If)
                     and "split_report" in ast.unparse(node.test))
        guard_code = compile(ast.Module(body=[guard], type_ignores=[]), "<no-truncation-guard>", "exec")
        model = self.tiny_model()
        model.eval()
        namespace = {
            "training": training, "DATASET": "expla_graphs", "MODEL_ID": "local-test",
            "resolved_revision": "model-sha", "SMOKE_TEST": False,
            "native_context": 4096,
        }
        settings = eval(compile(ast.Expression(setting_call), "<colab-settings>", "eval"), namespace)
        name_code = compile(ast.Module(body=[run_name], type_ignores=[]), "<run-name>", "exec")
        exec(name_code, namespace)
        full_run = namespace["RUN_NAME"]
        namespace["SMOKE_TEST"] = True
        exec(name_code, namespace)
        self.assertNotEqual(full_run, namespace["RUN_NAME"])
        self.assertNotEqual(full_run, "expla-graphs-chat-qlora-full-v2")
        output, drive = self.root / full_run, self.root / "drive" / full_run
        training.initialize_run(settings, dataset, self.splits, snapshot_hash, output, drive)
        model.settings = settings
        report = {name: training.token_report(model, dataset, ids) for name, ids in self.splits.items()}
        exec(guard_code, {"report": report})
        self.assertEqual(sum(r["descriptions_truncated"] for r in report.values()), 0)
        samples = artifacts.collate([dataset[0]])
        inputs = model.input_batch(samples, supervised=True)
        question = len(model.tokenizer.encode(samples["question"][0]))
        self.assertEqual(inputs["inputs_embeds"].shape[1], 2 + 1 + 700 + question + 1 + 2)
        kaggle = json.loads((ROOT / "notebook/kaggle/grag-expla-graphs-full-graph.ipynb").read_text())
        tree = ast.parse("".join(next(c for c in kaggle["cells"] if c["id"] == "model")["source"]))
        kaggle_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "Settings")
        kaggle_settings = eval(compile(ast.Expression(kaggle_call), "<kaggle-settings>", "eval"), {
            "training": training, "DATASET": "expla_graphs", "MODEL_ID": "local-test",
            "model_revision": "model-sha", "SMOKE_TEST": False,
            "config": SimpleNamespace(max_position_embeddings=4096),
        })
        self.assertEqual(settings, kaggle_settings)
        self.assertEqual((run / "config.json").read_bytes(), original_config_bytes)
        self.assertEqual((run / "samples/0.pt").read_bytes(), original_payload_bytes)

    def test_colab_untruncated_preflight_stops_before_model_load_and_rejects_full_graph(self):
        from types import SimpleNamespace

        notebook = json.loads((ROOT / "notebook/colab/grag_expla_graphs_downstream.ipynb").read_text())
        cells = {cell["id"]: ast.parse("".join(cell["source"]))
                 for cell in notebook["cells"] if cell["cell_type"] == "code"}
        guard = next(node for node in cells["cell-10"].body if isinstance(node, ast.If)
                     and "split_report" in ast.unparse(node.test))
        loads = [node for node in cells["cell-10"].body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                 and node.value.func.attr in ("initialize_run", "load_qlora")]
        self.assertTrue(all(cells["cell-10"].body.index(guard) < cells["cell-10"].body.index(node)
                            for node in loads))
        code = compile(ast.Module(body=[guard, *loads], type_ignores=[]), "<preflight-before-loading>", "exec")
        for key in ("descriptions_truncated", "answer_targets_truncated"):
            training_stub = Mock()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cannot fit intact"):
                exec(code, {"report": {"test": {key: 1}},
                            "training": training_stub})
            training_stub.initialize_run.assert_not_called()
            training_stub.load_qlora.assert_not_called()
        retrieval_guard = next(node for node in cells["cell-06"].body if isinstance(node, ast.If)
                               and "dataset.config" in ast.unparse(node.test))
        for config in ({"pipeline": "expla_full_graph"}, {}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "original retrieval snapshot"):
                exec(compile(ast.Module(body=[retrieval_guard], type_ignores=[]), "<retrieval-guard>", "exec"),
                     {"dataset": SimpleNamespace(config=config)})

    def test_newer_best_checkpoint_survives_older_progress_and_retries_backup(self):
        output, drive, _, history = self.fit_scalar(ScalarModel(self.settings), "ahead")
        best = torch.load(output / "best.pt", weights_only=True)
        best["validation_loss"] = 0.
        training.atomic_checkpoint(best, output / "best.pt")
        progress = torch.load(output / "last.pt", weights_only=True)
        progress["state"].update(epoch=0, next_batch=3, best_loss=float("inf"),
                                 best_epoch=-1, history=[], train_batches=3,
                                 train_loss_sum=history[0]["train_loss"] * 3)
        training.atomic_checkpoint(progress, output / "last.pt")
        self.fit_scalar(ScalarModel(self.settings), "ahead")
        remote_best = torch.load(drive / "best.pt", weights_only=True)
        remote_last = torch.load(drive / "last.pt", weights_only=True)
        self.assertEqual(remote_best["validation_loss"], 0.)
        self.assertEqual(remote_last["state"]["best_loss"], 0.)
        self.assertEqual(artifacts.checksum(output / "history.json"),
                         artifacts.checksum(drive / "history.json"))

    def test_metrics_and_notebook_syntax(self):
        scores = training.score_predictions("expla_graphs", [
            {"prediction": "Support.", "label": "support"},
            {"prediction": "COUNTER", "label": "counter"},
            {"prediction": "unsupported", "label": "support"},
        ])
        self.assertEqual(scores["accuracy"], 2 / 3)
        self.assertEqual(scores["invalid_outputs"], 1)
        web = training.score_predictions("webqsp", [{"prediction": "Paris", "label": "paris|london"}])
        self.assertEqual(web["hit"], 100)
        self.assertEqual(web["answer_coverage"], 50)
        expla = ROOT / "notebook/colab/grag-expla-graphs-downstream.ipynb"
        if not expla.exists():
            expla = ROOT / "notebook/colab/grag_expla_graphs_downstream.ipynb"
        for path in (expla, ROOT / "notebook/colab/grag-webqsp-downstream.ipynb",
                     ROOT / "notebook/kaggle/grag-webqsp-downstream.ipynb",
                     ROOT / "notebook/kaggle/grag-expla-graphs-full-graph.ipynb"):
            notebook = json.loads(path.read_text())
            import nbformat
            nbformat.validate(nbformat.from_dict(notebook))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]), filename=path.name)

    def test_audited_data_test_predictions_backup_resume_and_denominators(self):
        records = json.loads((self.prepared / "records.json").read_text())
        for split, indices in self.splits.items():
            for position, index in enumerate(indices):
                records[index].update(source_index=index, source_id=f"source-{index}",
                                      source_split=split, source_split_index=position)
        artifacts.atomic_json(records, self.prepared / "records.json")
        artifacts.atomic_json([{"source_index": 5, "reason": "empty_graph"}],
                              self.prepared / "excluded.json")
        metadata = self.dataset.metadata.copy()
        metadata.update(source_count=6, excluded_count=1,
                        source_splits={**self.splits, "test": [4, 5]},
                        records_sha256=artifacts.checksum(self.prepared / "records.json"),
                        exclusions_sha256=artifacts.checksum(self.prepared / "excluded.json"))
        artifacts.atomic_json(metadata, self.prepared / "prepared.json")
        config = self.dataset.config.copy()
        config["prepared_sha256"] = artifacts.checksum(self.prepared / "prepared.json")
        artifacts.atomic_json(config, self.run / "config.json")
        for record in records:
            path = self.run / "samples" / f"{record['id']}.pt"
            payload = torch.load(path, weights_only=True)
            payload.update(record, config_sha256=artifacts.checksum(self.run / "config.json"))
            torch.save(payload, path)
        dataset = artifacts.CompactTrainingDataset(self.prepared, self.run)
        output, drive = self.root / "predictions", self.root / "predictions-drive"
        signature = training.initialize_run(self.settings, dataset, self.splits, "sha", output, drive)
        model = self.tiny_model()
        training.save_and_backup({"signature": signature, "learned": model.learned_state()},
                                 "best.pt", output, drive, checkpoint=True)
        rows, metrics = training.predict_test(model, dataset, [4], signature, output, drive)
        self.assertEqual(rows[0]["source_id"], "source-4")
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["source_test_count"], 2)
        self.assertEqual(metrics["excluded_test_count"], 1)
        self.assertEqual(metrics["scope"], "full_retained_dataset")
        self.assertEqual(artifacts.checksum(output / "metrics.json"),
                         artifacts.checksum(drive / "metrics.json"))
        with patch.object(model, "predict", side_effect=AssertionError("Should reuse predictions")):
            repeated, _ = training.predict_test(model, dataset, [4], signature, output, drive)
        self.assertEqual(rows, repeated)
        rows[0]["label"] = "counter"
        artifacts.atomic_json(rows, output / "predictions.json")
        with self.assertRaisesRegex(ValueError, "differs"):
            training.predict_test(model, dataset, [4], signature, output, drive)


if __name__ == "__main__":
    unittest.main()
