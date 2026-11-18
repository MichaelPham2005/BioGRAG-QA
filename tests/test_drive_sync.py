import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.utils.notebook_downstream import artifacts, drive_sync
from tests.test_drive_backup import FakeDrive, Request


class MutableDrive(FakeDrive):
    def __init__(self):
        super().__init__()
        self.downloads = []
        self.corrupt_download = False
        self.bad_upload_metadata = False

    def describe(self, key):
        return {**super().describe(key), "parents": self.entries[key]["parents"]}

    def get_media(self, fileId):
        self.downloads.append(fileId)
        return self.entries[fileId]["data"] + (b"corrupt" if self.corrupt_download else b"")

    def update(self, fileId, media_body, **kwargs):
        drive = self

        class Upload:
            def next_chunk(self, num_retries):
                entry = drive.entries[fileId]
                if entry["name"] == drive.fail_upload:
                    raise OSError("Simulated network interruption")
                entry["data"] = Path(media_body).read_bytes()
                drive.uploads.append(entry["name"])
                return None, {"id": fileId}

        return Upload()

    def get(self, fileId, **kwargs):
        metadata = self.describe(fileId)
        if self.bad_upload_metadata:
            metadata["md5Checksum"] = "invalid"
        return Request(metadata)


class Download:
    def __init__(self, handle, request, **kwargs):
        self.handle, self.data = handle, request

    def next_chunk(self, num_retries):
        self.handle.write(self.data)
        return None, True


class DriveSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.drive = MutableDrive()
        self.folder = drive_sync.DriveFolder(self.drive, "run-folder", self.root / "cache")
        self.media = patch.object(drive_sync, "upload_media", side_effect=lambda path: path)
        self.downloader = patch("googleapiclient.http.MediaIoBaseDownload", Download)
        self.media.start()
        self.downloader.start()

    def tearDown(self):
        self.downloader.stop()
        self.media.stop()
        self.temporary.cleanup()

    def remote_file(self, name, data, parent="run-folder"):
        key = str(len(self.drive.entries))
        self.drive.entries[key] = {
            "name": name, "data": data, "parents": [parent],
            "mimeType": "application/octet-stream",
        }
        return key

    def test_checkpoint_upsert_retry_and_verified_download(self):
        source = self.root / "last.pt"
        source.write_bytes(b"first checkpoint")
        destination = self.folder.cache / source.name
        self.folder.backup_file(source, destination)
        self.folder.backup_file(source, destination)
        self.assertEqual(self.drive.uploads, ["last.pt"])
        source.write_bytes(b"newer checkpoint")
        self.drive.fail_upload = "last.pt"
        with self.assertRaisesRegex(OSError, "network interruption"):
            self.folder.backup_file(source, destination)
        self.assertEqual(source.read_bytes(), destination.read_bytes())
        self.assertEqual(self.drive.entries["0"]["data"], b"first checkpoint")
        self.drive.fail_upload = None
        self.folder.backup_file(source, destination)
        self.assertEqual(len(self.drive.entries), 1)
        restored = drive_sync.DriveFolder(self.drive, "run-folder", self.root / "restored")
        restored.download("last.pt")
        self.assertEqual((restored.cache / "last.pt").read_bytes(), source.read_bytes())
        restored.download("last.pt")
        self.assertEqual(self.drive.downloads, ["0"])
        self.drive.bad_upload_metadata = True
        with self.assertRaisesRegex(OSError, "upload checksum"):
            self.folder.backup_file(source, destination)

    def test_corruption_unsafe_cache_names_and_orphan_checkpoints(self):
        self.remote_file("last.pt", b"checkpoint")
        self.drive.corrupt_download = True
        with self.assertRaisesRegex(OSError, "download checksum"):
            self.folder.download("last.pt")
        self.assertEqual(list(self.folder.cache.iterdir()), [])
        self.drive.corrupt_download = False
        with self.assertRaisesRegex(ValueError, "missing their run.json"):
            self.folder.restore_training_outputs()
        (self.folder.cache / "last.pt").write_bytes(b"conflicting local checkpoint")
        with self.assertRaisesRegex(ValueError, "Local/Drive"):
            self.folder.download("last.pt")
        for name in ("../last.pt", "/last.pt", "x\\last.pt", "..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.folder.local_path(name)
        with self.assertRaisesRegex(ValueError, "destination"):
            self.folder.backup_file(self.root / "unused", self.root / "escaped.pt")

    def test_snapshot_restore_deletes_parts_only_after_verified_extraction(self):
        contents = {"output/prepared/prepared.json": b"{}", "output/run/config.json": b"{}"}
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for name, data in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        data = raw.getvalue()
        manifest = {
            "format": 1, "dataset": "webqsp", "sample_count": 1,
            "restore_paths": {"prepared": "/kaggle/output/prepared", "run": "/kaggle/output/run"},
            "files": [{"path": name, "size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                      for name, value in contents.items()],
            "archives": [{"name": "part-00000.tar.gz", "size": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()}],
        }
        self.remote_file("manifest.json", json.dumps(manifest).encode(), "snapshot")
        self.remote_file("part-00000.tar.gz", data, "snapshot")
        cache, output = self.root / "parts", self.root / "data"
        prepared, run, _ = drive_sync.restore_drive_snapshot(
            self.drive, "snapshot", cache, output, "webqsp",
        )
        self.assertEqual((prepared / "prepared.json").read_bytes(), b"{}")
        self.assertEqual((run / "config.json").read_bytes(), b"{}")
        self.assertFalse((cache / "part-00000.tar.gz").exists())
        count = len(self.drive.downloads)
        drive_sync.restore_drive_snapshot(self.drive, "snapshot", cache, output, "webqsp")
        self.assertEqual(len(self.drive.downloads), count)
        (prepared / "prepared.json").write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "differs"):
            drive_sync.restore_drive_snapshot(self.drive, "snapshot", cache, output, "webqsp")


if __name__ == "__main__":
    unittest.main()
