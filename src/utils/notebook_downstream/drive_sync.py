"""Verified Google Drive API transfers for runtimes without a Drive filesystem mount."""

import json
import os
from pathlib import Path
import shutil
import tempfile

from src.utils.notebook_downstream import artifacts
from src.utils.notebook_preprocessing.drive_backup import find_remote, upload_media


class DriveFolder:
    def __init__(self, service, folder_id, cache):
        self.service, self.folder_id = service, folder_id
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)

    def local_path(self, name):
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"Drive cache entries must be plain filenames: {name}")
        path = self.cache / name
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink in the Drive cache: {name}")
        return path

    def download(self, name, *, required=True):
        from googleapiclient.http import MediaIoBaseDownload

        path = self.local_path(name)
        remote = find_remote(self.service, name, self.folder_id)
        if remote is None:
            if required:
                raise FileNotFoundError(f"Missing Drive file: {name}")
            return False
        if not remote.get("md5Checksum") or "size" not in remote:
            raise ValueError(f"Drive entry is not a checksummed binary file: {name}")
        size = int(remote["size"])
        if path.exists():
            if path.stat().st_size != size or artifacts.checksum(path, "md5") != remote["md5Checksum"]:
                raise ValueError(f"Local/Drive {name} differ; inspect or choose a fresh cache directory.")
            return True
        if shutil.disk_usage(self.cache).free < size + 2**30:
            raise OSError(f"Insufficient disk space to download {name} plus a 1 GiB reserve.")
        with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                request = self.service.files().get_media(
                    fileId=remote["id"], supportsAllDrives=True,
                )
                download = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _, done = download.next_chunk(num_retries=5)
                handle.flush()
                os.fsync(handle.fileno())
                if (temporary.stat().st_size != size
                        or artifacts.checksum(temporary, "md5") != remote["md5Checksum"]):
                    raise OSError(f"Drive download checksum/size mismatch: {name}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Drive downloaded and verified: {name}", flush=True)
        return True

    def restore_training_outputs(self):
        self.download("run.json", required=False)
        for name in ("last.pt", "best.pt", "history.json", "token-report.json",
                     "prediction-config.json", "predictions.json", "metrics.json"):
            self.download(name, required=False)
        if not (self.cache / "run.json").exists() and any(
            (self.cache / name).exists() for name in ("last.pt", "best.pt")
        ):
            raise ValueError("Drive checkpoints are missing their run.json; do not start a new run over them.")

    def backup_file(self, source, destination):
        source, destination = Path(source), Path(destination)
        if destination.parent.resolve() != self.cache or self.local_path(destination.name) != destination:
            raise ValueError("Checkpoint backup destination must be inside this Drive folder's cache.")
        artifacts.backup_file(source, destination)
        local_md5 = artifacts.checksum(destination, "md5")
        existing = find_remote(self.service, destination.name, self.folder_id)
        if existing is not None and existing.get("md5Checksum") == local_md5:
            remote_id = existing["id"]
        else:
            media = upload_media(destination)
            if existing is None:
                request = self.service.files().create(
                    body={"name": destination.name, "parents": [self.folder_id]},
                    media_body=media, fields="id", supportsAllDrives=True,
                )
            else:
                request = self.service.files().update(
                    fileId=existing["id"], media_body=media, fields="id",
                    supportsAllDrives=True,
                )
            response = None
            while response is None:
                _, response = request.next_chunk(num_retries=5)
            remote_id = response["id"]
        remote = self.service.files().get(
            fileId=remote_id, fields="id,size,md5Checksum,parents",
            supportsAllDrives=True,
        ).execute()
        if (remote.get("md5Checksum") != local_md5
                or int(remote["size"]) != destination.stat().st_size
                or self.folder_id not in remote["parents"]):
            raise OSError(f"Drive upload checksum/size/parent mismatch: {destination.name}; local files retained.")
        print(f"Drive uploaded and verified: {destination.name}", flush=True)


def restore_drive_snapshot(service, folder_id, cache, destination, dataset):
    """Download immutable archive parts only when extraction is needed; discard verified parts afterward."""
    folder = DriveFolder(service, folder_id, cache)
    folder.download("manifest.json")
    manifest_path = folder.cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != 1 or manifest["dataset"] != dataset:
        raise ValueError("Drive snapshot has the wrong format or dataset.")
    destination = Path(destination).resolve()
    marker = destination / ".snapshot-sha256"
    if marker.exists() and marker.read_text().strip() != artifacts.checksum(manifest_path):
        raise ValueError("Restore directory belongs to another snapshot; choose a fresh directory.")
    missing = any(not (destination / artifacts.safe_relative(item["path"])).is_file()
                  for item in manifest["files"])
    if not marker.exists() or missing:
        for part in manifest["archives"]:
            folder.download(part["name"])
    result = artifacts.restore_snapshot(folder.cache, destination, dataset)
    for part in manifest["archives"]:
        folder.local_path(part["name"]).unlink(missing_ok=True)
    return result
