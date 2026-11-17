"""Colab GRAG + QLoRA training, with lazy graphs and resumable Drive checkpoints."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch_geometric.nn import global_mean_pool

from src.utils.notebook_downstream.artifacts import (
    atomic_json, backup_file, checksum, collate, load_source,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
GNN = load_source("colab_original_gnn", SOURCE_ROOT / "model/gnn.py")
EVALUATION = load_source("colab_original_evaluation", SOURCE_ROOT / "utils/evaluate.py")


@dataclass
class Settings:
    dataset: str = "expla_graphs"
    model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_revision: str = ""
    seed: int = 3
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: float = 1
    patience: int = 3
    checkpoint_steps: int = 50
    max_sequence_tokens: int = 512
    max_description_tokens: int = 256
    max_answer_tokens: int = 8
    gnn_hidden: int = 1024
    gnn_layers: int = 4
    gnn_heads: int = 4
    projector_hidden: int = 2048
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def load_qlora(settings, token):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU runtime before loading the 4-bit model.")
    if not settings.model_revision:
        raise ValueError("Pin the gated model revision before loading/training.")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id, revision=settings.model_revision, token=token, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    llm = AutoModelForCausalLM.from_pretrained(
        settings.model_id, revision=settings.model_revision, token=token,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        ),
        torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
    )
    llm = prepare_model_for_kbit_training(
        llm, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    llm = get_peft_model(llm, LoraConfig(
        r=settings.lora_rank, lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    llm.config.use_cache = False
    return ColabGRAG(llm, tokenizer, settings)


class ColabGRAG(nn.Module):
    def __init__(self, llm, tokenizer, settings):
        super().__init__()
        self.llm, self.tokenizer, self.settings = llm, tokenizer, settings
        self.graph_encoder = GNN.GAT(
            in_channels=1024, hidden_channels=settings.gnn_hidden,
            out_channels=settings.gnn_hidden, num_layers=settings.gnn_layers,
            dropout=0, mlp_layers=3, num_heads=settings.gnn_heads, operator="euclidean",
        )
        self.projector = nn.Sequential(
            nn.Linear(settings.gnn_hidden, settings.projector_hidden), nn.Sigmoid(),
            nn.Linear(settings.projector_hidden, llm.config.hidden_size),
        )
        self.graph_encoder.to(self.device)
        self.projector.to(self.device)

    @property
    def device(self):
        return self.llm.get_input_embeddings().weight.device

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()

    def graph_tokens(self, graph):
        # The repository's GAT scales features in-place; do not mutate a reused batch.
        graph = graph.clone().to(self.device)
        batch_norms = [module for module in self.graph_encoder.modules()
                       if isinstance(module, nn.BatchNorm1d)]
        states = [module.training for module in batch_norms]
        if self.training and graph.num_nodes == 1:
            warnings.warn("Singleton graph batch: BatchNorm uses its stored running statistics.")
            for module in batch_norms:
                module.eval()
        try:
            x, _ = self.graph_encoder(
                graph.x, graph.edge_index.long(), graph.question_node,
                graph.edge_attr, graph.question_edge,
            )
        finally:
            for module, state in zip(batch_norms, states):
                module.train(state)
        pooled = global_mean_pool(x, graph.batch, size=graph.num_graphs)
        return self.projector(pooled)

    def input_batch(self, samples, supervised):
        tokenizer, config = self.tokenizer, self.settings
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        bos, end_user = encode("<s>[INST]"), encode("[/INST]")
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer needs explicit EOS and padding tokens.")
        embedding = self.llm.get_input_embeddings()
        graph_tokens = self.graph_tokens(samples["graph"]).to(embedding.weight.dtype)
        inputs, masks, targets = [], [], []
        for i, question in enumerate(samples["question"]):
            question_ids = encode(question)
            budget = (config.max_sequence_tokens - config.max_answer_tokens - 1
                      - len(bos) - 1 - len(question_ids) - len(end_user))
            if budget < 0:
                raise ValueError(f"Question {samples['id'][i]} exceeds the configured context budget.")
            description = encode(samples["desc"][i])[:min(config.max_description_tokens, budget)]
            prefix = bos
            suffix = description + question_ids + end_user
            if supervised:
                answer = encode(samples["label"][i])[:config.max_answer_tokens] + [tokenizer.eos_token_id]
            else:
                answer = []
            before = embedding(torch.tensor(prefix, dtype=torch.long, device=self.device))
            after = embedding(torch.tensor(suffix + answer, dtype=torch.long, device=self.device))
            value = torch.cat((before, graph_tokens[i:i + 1], after), dim=0)
            inputs.append(value)
            masks.append([1] * len(value))
            targets.append([-100] * (len(value) - len(answer)) + answer)
        width = max(map(len, inputs))
        pad = embedding(torch.tensor([tokenizer.pad_token_id], device=self.device))
        for i, value in enumerate(inputs):
            padding = width - len(value)
            inputs[i] = torch.cat((pad.repeat(padding, 1), value))
            masks[i] = [0] * padding + masks[i]
            targets[i] = [-100] * padding + targets[i]
        result = {
            "inputs_embeds": torch.stack(inputs),
            "attention_mask": torch.tensor(masks, device=self.device),
        }
        if supervised:
            result["labels"] = torch.tensor(targets, device=self.device)
            positions = result["attention_mask"].cumsum(-1) - 1
            result["position_ids"] = positions.clamp_min(0)
        return result

    def forward(self, samples):
        with self.autocast():
            return self.llm(**self.input_batch(samples, supervised=True), use_cache=False).loss

    @torch.inference_mode()
    def predict(self, samples):
        with self.autocast():
            inputs = self.input_batch(samples, supervised=False)
            generated = self.llm.generate(
                **inputs, max_new_tokens=self.settings.max_answer_tokens,
                do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def adapter_state(self):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.llm, save_embedding_layers=False)

    def load_adapter(self, state):
        from peft import set_peft_model_state_dict

        current = self.adapter_state()
        if set(current) != set(state) or any(current[key].shape != state[key].shape for key in current):
            raise ValueError("Checkpoint LoRA adapter keys/shapes do not match this model.")
        result = set_peft_model_state_dict(self.llm, state)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected adapter checkpoint keys: {result.unexpected_keys}")

    def learned_state(self):
        return cpu_copy({
            "graph_encoder": self.graph_encoder.state_dict(),
            "projector": self.projector.state_dict(), "lora": self.adapter_state(),
        })

    def load_learned_state(self, state):
        self.graph_encoder.load_state_dict(state["graph_encoder"], strict=True)
        self.projector.load_state_dict(state["projector"], strict=True)
        self.load_adapter(state["lora"])


def token_report(model, dataset, indices):
    config, tokenizer = model.settings, model.tokenizer
    report = {"answer_targets_truncated": 0, "descriptions_truncated": 0}
    for index in indices:
        row = dataset.read_payload(index)
        question = tokenizer.encode(row["question"], add_special_tokens=False)
        special = (len(tokenizer.encode("<s>[INST]", add_special_tokens=False)) + 1
                   + len(tokenizer.encode("[/INST]", add_special_tokens=False)))
        budget = config.max_sequence_tokens - config.max_answer_tokens - 1 - len(question) - special
        if budget < 0:
            raise ValueError(f"Question {row['id']} needs a larger context budget.")
        report["answer_targets_truncated"] += int(
            len(tokenizer.encode(row["label"], add_special_tokens=False)) > config.max_answer_tokens
        )
        report["descriptions_truncated"] += int(
            len(tokenizer.encode(row["desc"], add_special_tokens=False))
            > min(config.max_description_tokens, budget)
        )
    return report


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def atomic_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_run(settings, dataset, splits, snapshot_hash, output, drive_output):
    if (settings.epochs < 1 or settings.batch_size < 1 or settings.grad_accum < 1
            or settings.checkpoint_steps < 1 or settings.gnn_layers < 2
            or settings.patience < 1 or settings.warmup_epochs < 0
            or not 0 < settings.learning_rate or settings.max_answer_tokens < 1
            or settings.max_description_tokens < 0):
        raise ValueError("Epochs/batch/accumulation/checkpoint interval must be positive; GNN needs two layers.")
    if settings.dataset != dataset.metadata["dataset"] or not settings.model_revision:
        raise ValueError("Dataset must match the restored data and model revision must be pinned.")
    if set(splits) != {"train", "val", "test"} or any(not ids for ids in splits.values()):
        raise ValueError("Select nonempty train, validation and test subsets.")
    if any(len(ids) != len(set(ids)) for ids in splits.values()):
        raise ValueError("Split subsets contain duplicate IDs.")
    ids = [index for values in splits.values() for index in values]
    if len(ids) != len(set(ids)) or any(not 0 <= index < len(dataset) for index in ids):
        raise ValueError("Split subsets overlap or contain invalid IDs.")
    original = dataset.get_idx_split()
    if any(not set(splits[name]).issubset(original[name]) for name in splits):
        raise ValueError("A subset changed the original train/val/test membership.")
    output, drive_output = Path(output), Path(drive_output)
    output.mkdir(parents=True, exist_ok=True)
    drive_output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1, "settings": asdict(settings), "snapshot_sha256": snapshot_hash,
        "data_config_sha256": dataset.config_hash, "splits": splits,
        "training_helper_sha256": checksum(__file__),
        "artifact_helper_sha256": checksum(Path(__file__).with_name("artifacts.py")),
        "gnn_helper_sha256": checksum(SOURCE_ROOT / "model/gnn.py"),
        "evaluation_helper_sha256": checksum(SOURCE_ROOT / "utils/evaluate.py"),
        "scope": "subset_smoke" if any(len(splits[name]) != len(original[name]) for name in splits)
        else "full_retained_dataset",
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["signature"] = signature
    path = output / "run.json"
    for existing in (path, drive_output / path.name):
        if existing.exists() and json.loads(existing.read_text()) != manifest:
            raise ValueError("Run settings/data/code changed. Choose a new RUN_NAME; do not mix checkpoints.")
    if (drive_output / "last.pt").exists() and not (output / "last.pt").exists():
        raise ValueError("Restore the existing Drive checkpoints before resuming this run.")
    if not path.exists() and any((output / name).exists() for name in ("last.pt", "best.pt")):
        raise ValueError("Checkpoint metadata is missing; choose a fresh run directory.")
    atomic_json(manifest, path)
    backup_file(path, drive_output / path.name)
    return signature


def save_and_backup(value, name, output, drive_output, checkpoint=False):
    path = Path(output) / name
    if checkpoint:
        atomic_checkpoint(value, path)
    else:
        atomic_json(value, path)
    backup_file(path, Path(drive_output) / name)


def checkpoint_data(model, optimizer, scaler, signature, state):
    return {
        "format": 1, "signature": signature, "learned": model.learned_state(),
        "optimizer": cpu_copy(optimizer.state_dict()), "scaler": scaler.state_dict(),
        "state": state, "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else [],
        "python_rng": random.getstate(),
    }


def load_best(model, output, signature):
    checkpoint = torch.load(Path(output) / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["signature"] != signature:
        raise ValueError("Best checkpoint belongs to another run.")
    model.load_learned_state(checkpoint["learned"])


def validation_loss(model, dataset, indices, batch_size):
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, collate_fn=collate, num_workers=0)
    total = 0.
    with torch.inference_mode():
        for batch in loader:
            loss = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss is not finite.")
            total += loss.item()
    return total / len(loader)


def fit(model, dataset, splits, signature, output, drive_output):
    config = model.settings
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                 weight_decay=config.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda")
    state = {"epoch": 0, "next_batch": 0, "optimizer_steps": 0,
             "train_loss_sum": 0., "train_batches": 0, "best_loss": float("inf"),
             "best_epoch": -1, "history": []}
    last = Path(output) / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume checkpoint belongs to another run.")
        model.load_learned_state(checkpoint["learned"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if model.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        backup_file(last, Path(drive_output) / last.name)
        print(f"Resuming epoch {state['epoch']}, next microbatch {state['next_batch']}.")
    best_path = Path(output) / "best.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if best["signature"] != signature:
            raise ValueError("Best checkpoint belongs to another run.")
        # A disconnect can leave the best backup newer than the progress backup.
        if best["validation_loss"] < state["best_loss"]:
            state["best_loss"], state["best_epoch"] = best["validation_loss"], best["epoch"]
        del best
        backup_file(best_path, Path(drive_output) / best_path.name)
    while state["epoch"] < config.epochs:
        epoch = state["epoch"]
        if state["best_epoch"] >= 0 and epoch - 1 - state["best_epoch"] >= config.patience:
            break
        ordered = list(splits["train"])
        random.Random(config.seed + epoch).shuffle(ordered)
        total_batches = math.ceil(len(ordered) / config.batch_size)
        cursor = state["next_batch"]
        if cursor != total_batches and cursor % config.grad_accum:
            raise ValueError("Resume cursor is not at an optimizer-step boundary.")
        loader = DataLoader(
            Subset(dataset, ordered[cursor * config.batch_size:]),
            batch_size=config.batch_size, collate_fn=collate, num_workers=0,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader, start=cursor):
            group_start = batch_index // config.grad_accum * config.grad_accum
            group_size = min(config.grad_accum, total_batches - group_start)
            loss = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}.")
            scaler.scale(loss / group_size).backward()
            state["train_loss_sum"] += loss.item()
            state["train_batches"] += 1
            if (batch_index + 1) % config.grad_accum == 0 or batch_index + 1 == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 0.1, error_if_nonfinite=True)
                progress = epoch + (batch_index + 1) / total_batches
                if progress < config.warmup_epochs:
                    factor = progress / config.warmup_epochs
                else:
                    factor = 0.5 * (1 + math.cos(math.pi * (progress - config.warmup_epochs)
                                                / max(1e-9, config.epochs - config.warmup_epochs)))
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * factor
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                state["optimizer_steps"] += 1
                state["next_batch"] = batch_index + 1
                if state["optimizer_steps"] % config.checkpoint_steps == 0:
                    save_and_backup(checkpoint_data(model, optimizer, scaler, signature, state),
                                    "last.pt", output, drive_output, checkpoint=True)
                    print(f"Backed up optimizer step {state['optimizer_steps']}.", flush=True)
        val_loss = validation_loss(model, dataset, splits["val"], config.batch_size)
        entry = {"epoch": epoch, "train_loss": state["train_loss_sum"] / state["train_batches"],
                 "validation_loss": val_loss, "optimizer_steps": state["optimizer_steps"]}
        state["history"].append(entry)
        if val_loss < state["best_loss"]:
            state["best_loss"], state["best_epoch"] = val_loss, epoch
            save_and_backup({"signature": signature, "learned": model.learned_state(),
                             "epoch": epoch, "validation_loss": val_loss},
                            "best.pt", output, drive_output, checkpoint=True)
        state.update(epoch=epoch + 1, next_batch=0, train_loss_sum=0., train_batches=0)
        save_and_backup(checkpoint_data(model, optimizer, scaler, signature, state),
                        "last.pt", output, drive_output, checkpoint=True)
        save_and_backup(state["history"], "history.json", output, drive_output)
        print(entry, flush=True)
    load_best(model, output, signature)
    save_and_backup(state["history"], "history.json", output, drive_output)
    return state["history"]


def score_predictions(dataset, rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set.")
    if dataset == "expla_graphs":
        parsed = []
        for row in rows:
            match = re.search(r"\b(support|counter)\b", row["prediction"], flags=re.I)
            parsed.append(match.group(1).lower() if match else None)
            if row["label"] not in ("support", "counter"):
                raise ValueError("ExplaGraphs labels must be support or counter.")
        correct = sum(pred == row["label"] for pred, row in zip(parsed, rows))
        f1 = []
        for label in ("support", "counter"):
            tp = sum(pred == label and row["label"] == label for pred, row in zip(parsed, rows))
            fp = sum(pred == label and row["label"] != label for pred, row in zip(parsed, rows))
            fn = sum(pred != label and row["label"] == label for pred, row in zip(parsed, rows))
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        confusion = {
            gold: {prediction: sum(row["label"] == gold and pred == prediction
                                   for pred, row in zip(parsed, rows))
                   for prediction in ("support", "counter", None)}
            for gold in ("support", "counter")
        }
        confusion = {gold: {key if key is not None else "invalid": count
                            for key, count in counts.items()}
                     for gold, counts in confusion.items()}
        return {"count": len(rows), "accuracy": correct / len(rows),
                "macro_f1": sum(f1) / 2, "invalid_outputs": parsed.count(None),
                "confusion": confusion}
    if dataset != "webqsp":
        raise ValueError("Unsupported evaluation dataset.")
    values = []
    for row in rows:
        answers = row["label"].split("|")
        prediction = row["prediction"].replace("|", "\n").split("\n")
        f1, precision, recall = EVALUATION.eval_f1(prediction, answers)
        joined = " ".join(prediction)
        values.append((EVALUATION.eval_hit(joined, answers),
                       EVALUATION.eval_acc(joined, answers), f1, precision, recall))
    names = ("hit", "answer_coverage", "f1", "precision", "recall")
    return {"count": len(rows), **{
        name: sum(value[i] for value in values) * 100 / len(values)
        for i, name in enumerate(names)
    }}


def predict_test(model, dataset, indices, signature, output, drive_output):
    load_best(model, output, signature)
    prediction_config = {
        "signature": signature, "best_checkpoint_sha256": checksum(Path(output) / "best.pt"),
        "split": "test", "ids": indices,
    }
    config_path = Path(output) / "prediction-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != prediction_config:
        raise ValueError("Existing predictions used another checkpoint or test subset.")
    save_and_backup(prediction_config, config_path.name, output, drive_output)
    path = Path(output) / "predictions.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    seen = {row["id"] for row in rows}
    if len(seen) != len(rows) or not seen.issubset(indices):
        raise ValueError("Prediction cache has duplicate or unexpected IDs.")
    if any(not isinstance(row["prediction"], str)
           or any(row[key] != dataset.records[row["id"]][key] for key in ("question", "label"))
           for row in rows):
        raise ValueError("Prediction cache question/label differs from the source records.")
    model.eval()
    for index in indices:
        if index in seen:
            continue
        sample = dataset[index]
        prediction = model.predict(collate([sample]))[0]
        rows.append({
            "id": index, "source_index": sample.get("source_index"),
            "source_id": sample.get("source_id"), "source_split": sample.get("source_split"),
            "source_split_index": sample.get("source_split_index"), "question": sample["question"],
            "label": sample["label"], "prediction": prediction,
        })
        if len(rows) % 10 == 0:
            save_and_backup(rows, "predictions.json", output, drive_output)
            print(f"Backed up {len(rows)}/{len(indices)} test predictions.", flush=True)
    save_and_backup(rows, "predictions.json", output, drive_output)
    scores = score_predictions(model.settings.dataset, rows)
    manifest = json.loads((Path(output) / "run.json").read_text())
    scores.update(scope=manifest["scope"], evaluation_split="test",
                  source_count=dataset.metadata.get("source_count", dataset.metadata["count"]),
                  retained_dataset_count=dataset.metadata["count"],
                  excluded_count=dataset.metadata.get("excluded_count", 0))
    source_test_count = len(dataset.metadata.get("source_splits", dataset.metadata["splits"])["test"])
    retained_test_count = len(dataset.metadata["splits"]["test"])
    scores.update(source_test_count=source_test_count, retained_test_count=retained_test_count,
                  excluded_test_count=source_test_count - retained_test_count)
    save_and_backup(scores, "metrics.json", output, drive_output)
    return rows, scores
