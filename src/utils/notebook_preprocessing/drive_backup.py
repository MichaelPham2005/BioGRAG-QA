"""Optional Google Drive backup of completed compact preprocessing runs."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def checksum(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect_files(prepared, run):
    """Whitelist data/provenance, never upload credentials, logs, or model caches."""
    prepared, run = Path(prepared).resolve(), Path(run).resolve()
    if prepared.parent != run.parent or prepared == run:
        raise ValueError("Prepared and run directories must be distinct siblings under output/.")
    metadata = json.loads((prepared / "prepared.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if (config["count"] != metadata["count"]
            or Path(config["prepared"]).resolve() != prepared
            or config["prepared_sha256"] != checksum(prepared / "prepared.json")
            or metadata["records_sha256"] != checksum(prepared / "records.json")):
        raise ValueError("Run configuration does not match the prepared data.")
    expected = {f"{i}.pt" for i in range(metadata["count"])}
    if {path.name for path in (run / "samples").glob("*.pt")} != expected:
        raise ValueError("Backup requires a complete run, not a smoke subset.")
    records = json.loads((prepared / "records.json").read_text())
    if "exclusions_sha256" in metadata:
        if checksum(prepared / "excluded.json") != metadata["exclusions_sha256"]:
            raise ValueError("Exclusion report changed.")
    if len(records) != metadata["count"]:
        raise ValueError("Prepared record count is inconsistent.")
    for index, record in enumerate(records):
        if record["id"] != index:
            raise ValueError("Prepared row IDs are inconsistent.")
        for folder in ("nodes", "edges"):
            if checksum(prepared / folder / f"{index}.csv") != record[f"{folder}_sha256"]:
                raise ValueError(f"Prepared {folder} table changed for sample {index}.")
    files = []
    for root, patterns in (
        (prepared, ["prepared.json", "records.json", "excluded.json", "nodes/*.csv", "edges/*.csv",
                    "split/*_indices.txt"]),
        (run, ["config.json", "samples/*.pt", "session-*-provenance.json", "benchmark/*.json"]),
    ):
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path.is_symlink() or root not in path.resolve().parents:
                    raise ValueError(f"Refusing a symlink or escaped backup path: {path}")
                if not path.is_file():
                    raise ValueError(f"Not a regular backup file: {path}")
                files.append({
                    "path": f"output/{root.name}/{path.relative_to(root).as_posix()}",
                    "size": path.stat().st_size, "sha256": checksum(path),
                    "_local": path,
                })
    for folder, suffix in (("nodes", ".csv"), ("edges", ".csv")):
        actual = {p.name for p in (prepared / folder).glob(f"*{suffix}")}
        if actual != {f"{i}{suffix}" for i in range(metadata["count"])}:
            raise ValueError(f"Prepared {folder} tables are incomplete or contain unexpected files.")
    for split in ("train", "val", "test"):
        path = prepared / "split" / f"{split}_indices.txt"
        if [int(line) for line in path.read_text().splitlines()] != metadata["splits"][split]:
            raise ValueError(f"Prepared {split} split indices changed.")
    manifest = {
        "format": 1, "dataset": config["dataset"], "sample_count": config["count"],
        "restore_paths": {"prepared": str(prepared), "run": str(run)},
        "files": [{key: value for key, value in item.items() if key != "_local"} for item in files],
    }
    if "source_count" in metadata:
        manifest.update(source_count=metadata["source_count"],
                        excluded_count=metadata["excluded_count"])
    return manifest, files


def archive_groups(files, target_bytes):
    if target_bytes < 1:
        raise ValueError("Archive target size must be positive.")
    group, size = [], 0
    for item in files:
        estimate = item["size"] + 1024
        if group and size + estimate > target_bytes:
            yield group
            group, size = [], 0
        group.append(item)
        size += estimate
    if group:
        yield group


def write_archive(files, destination):
    estimate = sum(item["size"] + 1024 for item in files)
    required = int(estimate * 1.01) + 65536 + 2**30
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError("Insufficient space for one backup part plus 1 GiB reserve.")
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as archive:
                for item in files:
                    info = tarfile.TarInfo(item["path"])
                    info.size, info.mode = item["size"], 0o644
                    with item["_local"].open("rb") as source:
                        archive.addfile(info, source)
    for item in files:
        if checksum(item["_local"]) != item["sha256"]:
            raise ValueError(f"Source changed during backup: {item['path']}")


def find_remote(service, name, parent):
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{escaped}' and '{parent}' in parents and trashed = false"
    response = service.files().list(
        q=query, spaces="drive", fields="files(id,name,mimeType,md5Checksum,size)",
        pageSize=2,
    ).execute()
    matches = response.get("files", [])
    if len(matches) > 1:
        raise ValueError(f"Multiple Drive entries named {name}; resolve duplicates before retrying.")
    return matches[0] if matches else None


def ensure_folder(service, name, parent="root"):
    existing = find_remote(service, name, parent)
    if existing is not None:
        if existing["mimeType"] != "application/vnd.google-apps.folder":
            raise ValueError(f"Drive entry {name} is not a folder.")
        return existing["id"]
    return service.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder",
              "parents": [parent]},
        fields="id",
    ).execute()["id"]


def upload_media(path):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(
        str(path), mimetype="application/octet-stream",
        chunksize=8 * 1024 * 1024, resumable=True,
    )


def upload_file(service, path, parent):
    local_md5 = checksum(path, "md5")
    existing = find_remote(service, path.name, parent)
    if existing is None:
        request = service.files().create(
            body={"name": path.name, "parents": [parent]},
            media_body=upload_media(path), fields="id",
        )
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=5)
        remote_id = response["id"]
    else:
        remote_id = existing["id"]
    remote = service.files().get(fileId=remote_id, fields="id,md5Checksum,size").execute()
    if remote.get("md5Checksum") != local_md5 or int(remote["size"]) != path.stat().st_size:
        raise ValueError(f"Drive checksum/size mismatch for {path.name}; local outputs are retained.")
    print(f"Drive verified: {path.name} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return remote_id


def backup_to_drive(prepared, run, service, *, part_bytes=256 * 1024 * 1024):
    """Upload bounded parts, then publish the manifest only after every part verifies.

    Call the preprocessing pipeline's validate_run() first. Re-running reuses
    verified parts in the same snapshot; it never deletes or mirrors Drive data.
    """
    manifest, files = collect_files(prepared, run)
    if part_bytes < 1:
        raise ValueError("Archive target size must be positive.")
    manifest["part_target_bytes"] = part_bytes
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    root = ensure_folder(service, "GRAG preprocessing backups")
    snapshot = ensure_folder(service, f"{manifest['dataset']}-{identity}", root)
    manifest["archives"] = []
    with tempfile.TemporaryDirectory(prefix=".drive-backup-", dir=Path(run).parent) as temporary:
        temporary = Path(temporary)
        for index, group in enumerate(archive_groups(files, part_bytes)):
            path = temporary / f"part-{index:05d}.tar.gz"
            try:
                write_archive(group, path)
                remote_id = upload_file(service, path, snapshot)
                manifest["archives"].append({
                    "name": path.name, "id": remote_id, "size": path.stat().st_size,
                    "sha256": checksum(path),
                })
            finally:
                path.unlink(missing_ok=True)
        path = temporary / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        upload_file(service, path, snapshot)
    url = f"https://drive.google.com/drive/folders/{snapshot}"
    print(f"Backup complete: {url}", flush=True)
    return url
