"""Verified Drive snapshot restore and lazy compact-data access for Colab."""

import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

from torch.utils.data import Dataset
from torch_geometric.data import Batch

from src.utils.notebook_preprocessing import kaggle_preprocess as preprocessing
from src.utils.notebook_preprocessing.drive_backup import checksum


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(value, path):
    preprocessing.atomic_json(value, path)


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def restore_snapshot(snapshot, destination, dataset):
    snapshot, destination = Path(snapshot), Path(destination).resolve()
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != 1 or manifest["dataset"] != dataset:
        raise ValueError("This snapshot has the wrong format or dataset.")
    manifest_hash = checksum(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".snapshot-sha256"
    if marker.exists() and marker.read_text().strip() != manifest_hash:
        raise ValueError("Restore directory belongs to another snapshot; choose a fresh directory.")
    expected = {}
    for item in manifest["files"]:
        relative = safe_relative(item["path"])
        if relative.parts[0] != "output" or item["path"] in expected or item["size"] < 0:
            raise ValueError("Invalid or duplicate snapshot file.")
        expected[item["path"]] = item
        target = destination.joinpath(*relative.parts)
        if destination not in target.resolve().parents:
            raise ValueError(f"Restore path escapes its destination: {relative}")
        if target.exists() and (not target.is_file() or checksum(target) != item["sha256"]):
            raise ValueError(f"Existing restored file differs: {target}; choose a fresh directory.")
    missing_bytes = sum(item["size"] for name, item in expected.items()
                        if not (destination / name).is_file())
    if shutil.disk_usage(destination).free < missing_bytes + 2**30:
        raise OSError("Insufficient Colab disk space for restored files plus 1 GiB reserve.")
    if not marker.exists() or any(not (destination / name).is_file() for name in expected):
        seen = set()
        for part in manifest["archives"]:
            relative = safe_relative(part["name"])
            if len(relative.parts) != 1:
                raise ValueError("Archive parts must be direct children of the snapshot folder.")
            path = snapshot / part["name"]
            if path.stat().st_size != part["size"] or checksum(path) != part["sha256"]:
                raise ValueError(f"Archive checksum/size mismatch: {path.name}")
            with tarfile.open(path, mode="r|gz") as archive:
                for member in archive:
                    name = safe_relative(member.name).as_posix()
                    if not member.isfile() or name not in expected or name in seen:
                        raise ValueError(f"Unexpected, duplicate, or non-regular archive member: {name}")
                    item = expected[name]
                    if member.size != item["size"]:
                        raise ValueError(f"Archive member size differs: {name}")
                    seen.add(name)
                    target = destination / name
                    if target.is_file():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                        temporary = Path(handle.name)
                        try:
                            with archive.extractfile(member) as source:
                                shutil.copyfileobj(source, handle, length=1024 * 1024)
                            handle.flush()
                            os.fsync(handle.fileno())
                            if checksum(temporary) != item["sha256"]:
                                raise ValueError(f"Restored file checksum mismatch: {name}")
                            os.replace(temporary, target)
                        finally:
                            temporary.unlink(missing_ok=True)
        if seen != set(expected):
            raise ValueError("Snapshot archive parts do not contain exactly the manifest files.")
        marker.write_text(manifest_hash + "\n")
    prepared = destination / "output" / Path(manifest["restore_paths"]["prepared"]).name
    run = destination / "output" / Path(manifest["restore_paths"]["run"]).name
    if f"output/{prepared.name}/prepared.json" not in expected or f"output/{run.name}/config.json" not in expected:
        raise ValueError("Manifest restore paths do not identify its prepared/run data.")
    print(f"Verified snapshot: {dataset}, {manifest['sample_count']} retained samples.")
    return prepared, run, manifest_hash


class CompactTrainingDataset(Dataset):
    """Consume format-1 artifacts without requiring the producer's old Python path/code."""

    def __init__(self, prepared, run):
        self.prepared, self.run = Path(prepared), Path(run)
        self.config = json.loads((self.run / "config.json").read_text())
        if self.config["format"] != 1:
            raise ValueError("Unsupported compact preprocessing format.")
        if checksum(self.prepared / "prepared.json") != self.config["prepared_sha256"]:
            raise ValueError("Prepared metadata differs from the producing run.")
        self.metadata, self.records = preprocessing.read_prepared(self.prepared)
        if (self.metadata["count"] != self.config["count"]
                or self.metadata["dataset"] != self.config["dataset"]):
            raise ValueError("Prepared/run sample counts or datasets differ.")
        self.config_hash = checksum(self.run / "config.json")
        expected = {f"{i}.pt" for i in range(len(self.records))}
        if {path.name for path in (self.run / "samples").glob("*.pt")} != expected:
            raise ValueError("Compact run is incomplete or contains unexpected sample files.")

    def __len__(self):
        return len(self.records)

    def read_payload(self, index):
        payload = preprocessing.load_payload(self.run / "samples" / f"{index}.pt")
        if payload["id"] != index or payload["config_sha256"] != self.config_hash:
            raise ValueError(f"Sample {index} belongs to another run or row.")
        if any(payload[key] != self.records[index][key] for key in ("question", "label")):
            raise ValueError(f"Sample {index} question/label differs from the prepared row.")
        return payload

    def __getitem__(self, index):
        payload = self.read_payload(index)
        sample = preprocessing.materialize(payload)
        for key in ("source_index", "source_id", "source_split", "source_split_index"):
            if key in self.records[index]:
                if key in payload and payload[key] != self.records[index][key]:
                    raise ValueError(f"Sample {index} has an inconsistent {key}.")
                sample[key] = self.records[index][key]
        return sample

    def get_idx_split(self):
        return self.metadata["splits"]


def collate(samples):
    result = {key: [sample[key] for sample in samples] for key in ("id", "question", "label", "desc")}
    result["graph"] = Batch.from_data_list([sample["graph"] for sample in samples])
    return result


def backup_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = checksum(source)
    if destination.exists() and checksum(destination) == digest:
        return
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if checksum(temporary) != digest:
            raise OSError(f"Drive backup checksum differs: {source.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_training_outputs(drive_output, local_output):
    drive_output, local_output = Path(drive_output), Path(local_output)
    local_output.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "last.pt", "best.pt", "history.json",
                 "prediction-config.json", "predictions.json", "metrics.json"):
        source = drive_output / name
        if source.is_file():
            target = local_output / name
            if target.exists() and checksum(source) != checksum(target):
                raise ValueError(f"Local and Drive {name} differ; inspect before restoring.")
            backup_file(source, target)
