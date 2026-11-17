import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "drive_backup", ROOT / "src/utils/notebook_preprocessing/drive_backup.py"
)
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


class Request:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class FakeDrive:
    def __init__(self):
        self.entries = {}
        self.uploads = []
        self.fail_upload = None

    def files(self):
        return self

    def list(self, q, **kwargs):
        name = re.search(r"name = '([^']+)'", q).group(1)
        parent = re.search(r"and '([^']+)' in parents", q).group(1)
        return Request({"files": [
            self.describe(key) for key, entry in self.entries.items()
            if entry["name"] == name and parent in entry["parents"]
        ]})

    def describe(self, key):
        entry = self.entries[key]
        result = {"id": key, "name": entry["name"], "mimeType": entry["mimeType"]}
        if "data" in entry:
            result.update(md5Checksum=hashlib.md5(entry["data"]).hexdigest(),
                          size=str(len(entry["data"])))
        return result

    def create(self, body, media_body=None, **kwargs):
        if media_body is None:
            key = str(len(self.entries))
            self.entries[key] = dict(body)
            return Request({"id": key})
        drive = self

        class Upload:
            def next_chunk(self, num_retries):
                if body["name"] == drive.fail_upload:
                    raise OSError("Simulated network interruption")
                key = str(len(drive.entries))
                drive.entries[key] = {
                    **body, "mimeType": "application/octet-stream",
                    "data": Path(media_body).read_bytes(),
                }
                drive.uploads.append(body["name"])
                return None, {"id": key}

        return Upload()

    def get(self, fileId, **kwargs):
        return Request(self.describe(fileId))


class DriveBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared, self.run = self.root / "example-prepared", self.root / "example-run"
        for path in (self.prepared / "nodes", self.prepared / "edges",
                     self.prepared / "split", self.run / "samples"):
            path.mkdir(parents=True)
        records = []
        for index in range(2):
            node = self.prepared / "nodes" / f"{index}.csv"
            edge = self.prepared / "edges" / f"{index}.csv"
            node.write_text("node_id,node_attr\n0,a\n1,b\n")
            edge.write_text("src,edge_attr,dst\n0,relation,1\n")
            records.append({"id": index, "nodes_sha256": backup.checksum(node),
                            "edges_sha256": backup.checksum(edge)})
            (self.run / "samples" / f"{index}.pt").write_bytes(b"sample-data" * 100)
        (self.prepared / "records.json").write_text(json.dumps(records))
        (self.prepared / "excluded.json").write_text('[{"source_index": 2, "reason": "empty_graph"}]')
        splits = {"train": [0], "val": [], "test": [1]}
        for split, indices in splits.items():
            (self.prepared / "split" / f"{split}_indices.txt").write_text(
                "\n".join(map(str, indices))
            )
        (self.prepared / "prepared.json").write_text(json.dumps({
            "count": 2, "splits": splits,
            "source_count": 3, "excluded_count": 1,
            "exclusions_sha256": backup.checksum(self.prepared / "excluded.json"),
            "records_sha256": backup.checksum(self.prepared / "records.json"),
        }))
        (self.run / "config.json").write_text(json.dumps({
            "count": 2, "dataset": "expla_graphs", "prepared": str(self.prepared),
            "prepared_sha256": backup.checksum(self.prepared / "prepared.json"),
        }))
        (self.run / "shard-0.log").write_text("private log should not be uploaded")
        (self.run / "session-1-packages.txt").write_text("environment should not be uploaded")
        (self.run / "credentials.json").write_text("synthetic decoy, not a credential")
        (self.run / "benchmark").mkdir()
        (self.run / "benchmark/shard-0.json").write_text('{"parameter_count": 100}')
        self.drive = FakeDrive()
        self.media = patch.object(backup, "upload_media", side_effect=lambda path: path)
        self.media.start()

    def tearDown(self):
        self.media.stop()
        self.temporary.cleanup()

    def upload(self):
        return backup.backup_to_drive(self.prepared, self.run, self.drive, part_bytes=4096)

    def test_round_trip_manifest_exclusions_and_idempotent_retry(self):
        expected, _ = backup.collect_files(self.prepared, self.run)
        url = self.upload()
        self.assertIn("drive.google.com/drive/folders/", url)
        self.assertEqual(self.drive.uploads[-1], "manifest.json")
        manifest_data = next(entry["data"] for entry in self.drive.entries.values()
                             if entry["name"] == "manifest.json")
        manifest = json.loads(manifest_data)
        self.assertGreater(len(manifest["archives"]), 1)
        restored = {}
        for part in manifest["archives"]:
            raw = self.drive.entries[part["id"]]["data"]
            self.assertEqual(hashlib.sha256(raw).hexdigest(), part["sha256"])
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
                for member in archive:
                    restored[member.name] = archive.extractfile(member).read()
        self.assertEqual(set(restored), {item["path"] for item in expected["files"]})
        for item in expected["files"]:
            self.assertEqual(hashlib.sha256(restored[item["path"]]).hexdigest(), item["sha256"])
        self.assertFalse(any("credentials" in path or ".log" in path or "packages" in path
                             for path in restored))
        self.assertIn(f"output/{self.run.name}/benchmark/shard-0.json", restored)
        self.assertIn(f"output/{self.prepared.name}/excluded.json", restored)
        self.assertEqual((manifest["source_count"], manifest["excluded_count"]), (3, 1))
        count = len(self.drive.uploads)
        self.assertEqual(self.upload(), url)
        self.assertEqual(len(self.drive.uploads), count)
        self.assertFalse(list(self.root.glob(".drive-backup-*")))
        self.assertEqual(len(list((self.run / "samples").glob("*.pt"))), 2)

    def test_interruption_has_no_success_manifest_and_can_resume(self):
        self.drive.fail_upload = "part-00001.tar.gz"
        with self.assertRaisesRegex(OSError, "network interruption"):
            self.upload()
        self.assertNotIn("manifest.json", self.drive.uploads)
        self.assertFalse(list(self.root.glob(".drive-backup-*")))
        self.assertEqual(self.drive.uploads.count("part-00000.tar.gz"), 1)
        self.drive.fail_upload = None
        self.upload()
        self.assertEqual(self.drive.uploads.count("part-00000.tar.gz"), 1)
        self.assertEqual(self.drive.uploads[-1], "manifest.json")

    def test_remote_corruption_is_reported_instead_of_claiming_success(self):
        self.upload()
        part = next(entry for entry in self.drive.entries.values()
                    if entry["name"] == "part-00000.tar.gz")
        part["data"] = b"corrupted"
        with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
            self.upload()

    def test_incomplete_or_changed_inputs_are_rejected_before_upload(self):
        (self.run / "samples/1.pt").unlink()
        with self.assertRaisesRegex(ValueError, "complete run"):
            self.upload()
        self.assertEqual(self.drive.entries, {})
        (self.run / "samples/1.pt").write_bytes(b"sample")
        (self.prepared / "nodes/0.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "table changed"):
            self.upload()
        self.assertEqual(self.drive.entries, {})

    def test_insufficient_scratch_space_preserves_local_outputs(self):
        with patch.object(backup.shutil, "disk_usage",
                          return_value=type("Disk", (), {"free": 1})()):
            with self.assertRaisesRegex(OSError, "Insufficient space"):
                self.upload()
        self.assertFalse(self.drive.uploads)
        self.assertFalse(list(self.root.glob(".drive-backup-*")))
        self.assertTrue((self.run / "samples/0.pt").is_file())


if __name__ == "__main__":
    unittest.main()
