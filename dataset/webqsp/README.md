# WebQSP preparation on Kaggle (2 x T4)

## Recommended: compact streaming notebooks

For visualization and storage-conscious preprocessing, use the updated
[WebQSP notebook](../../notebook/kaggle/grag-webqsp-data-processing.ipynb).
The matching
[ExplaGraphs notebook](../../notebook/kaggle/grag-expla-graphs-data-processing.ipynb)
deliberately applies WebQSP-style retrieval to the supplied explanation graphs;
this is **not the original ExplaGraphs baseline**.

Both notebooks use Kaggle's current Python/PyTorch/CUDA and install only missing
preprocessing dependencies. No virtual environment, Llama weights, Hugging Face
token, `gensim`, `peft`, or `torch_scatter` is required for this new path.
Enable Internet and select **GPU T4 x2** manually in Kaggle Settings.
Commit/push or upload the updated notebooks and
[shared helper](../../src/utils/notebook_preprocessing/kaggle_preprocess.py) before running them remotely.
Notebook-specific preprocessing helpers live in
[`src/utils/notebook_preprocessing/`](../../src/utils/notebook_preprocessing),
separate from the model's other utilities; their regression tests remain under
[`tests/`](../../tests).

### Why these outputs are smaller

Each worker embeds one query graph at a time, retrieves its ego-graphs immediately,
then releases the full embeddings. It does **not** write `graphs/*.pt`,
`cached_graphs/*.pt`, or repeated query-feature matrices. Compact samples store:

- unique selected node/relation features as FP32 tensors;
- integer mappings that reproduce overlapping ego components and their edges;
- one query vector, ranking scores, source IDs, CSV description, question and label.

`pipeline.CompactDataset(RUN)` in each notebook reconstructs the
`id/question/label/desc/graph` sample contract, including repeated query fields,
only when loading a sample. Storage sharing is lossless relative to the embeddings
computed in that run. It is an explicit adapter: the legacy dataset registry and
training entry point do **not** automatically read compact outputs. No training
behavior or existing caches are changed by these notebooks.

Retrieval keeps `k=2`, `topk=20`, `topk_entity=10`, `augment="none"`; top-k text
selection does not prune the soft graph. Directed neighborhoods and concatenated
overlapping egos follow the existing pipeline. WebQSP retains the author's
removal of **all** edges with repeated endpoint pairs, but keeps filtered
relation text aligned with its features. Single-center outputs, isolated nodes
and edgeless graphs receive valid tensor shapes instead of silent fallbacks.
ExplaGraphs query embeddings use the current row's arguments, correcting the
legacy preprocessor's full-column interpolation in this new path only.
These corrections and native dependency versions mean this is not bit-for-bit
reproduction of the authors' pipeline.

### Run order and limits

1. Clone/reuse the checkout and validate both GPUs.
2. Prepare text tables and original split IDs; inspect size distributions.
3. Configure the run (model/source revisions are recorded) and smoke-test two
   samples, one per GPU.
4. Inspect compact versus expanded sizes and retrieval/model-input shapes.
5. Set `RUN_FULL=True` only after reviewing disk/runtime estimates. Successful
   samples are resumed; changed inputs/settings require a new directory.
6. Validate the full run and save both prepared/run directories under `output/`.

The **6a** cell in both notebooks counts parameters from the actual loaded
SentenceBERT encoder and benchmarks preprocessing on both GPUs concurrently.
It reports per-GPU weight-only memory, embedding/retrieval timings, and remaining
runtime estimated from small/medium/large query-size buckets in each modulo
shard. The estimate uses the slower shard's projected duration, not a presumed
perfect 2x speedup, and excludes already-saved samples. It is a rough point
estimate, not an upper bound: token lengths, neighborhood topology and hardware
load can change throughput. Downloads, CSV preparation/plots, final full-run
validation, disk writes, Drive upload, and benchmark/monitor overhead are excluded.
The benchmark itself recomputes up to three representative queries per GPU and
adds runtime, but never modifies sample caches. Small timing reports are kept in
`RUN/benchmark/` and included in Drive backups.
Preprocessing optimizes **zero** parameters; no downstream Llama model is
counted or timed. Two GPU copies still represent one logical encoder model.
The separate [metrics helper](../../src/utils/notebook_preprocessing/preprocessing_metrics.py)
does not change the main pipeline's source hash or invalidate existing caches.

Workers use disjoint modulo shards and per-shard locks. The blocking monitor
stops its owned processes on interruption or worker failure; it never kills
processes by name. Retrieval runs on CPU between GPU embedding batches, so
continuous 100% utilization on both T4s is not expected.

No graph-size cap or lossy precision/storage change is imposed. Compact storage
reduces duplication but cannot guarantee that arbitrary graphs fit in RAM or
Kaggle's disk/time quota. Estimates from two examples are not upper bounds.
Workers refuse writes that would leave less than 2 GiB free. Old legacy caches
are not deleted automatically. Save outputs and restore them to the same working
paths to resume after a session reset; see the notebooks' final sections.

Focused local regression checks (with torch, PyG, pandas, scikit-learn and
transformers available):

```bash
python -m unittest discover -s tests -p test_kaggle_preprocess.py
```

### Google Drive backup (optional)

Both notebooks now end with an optional automatic backup to **personal Google
Drive**. Set `ENABLE_DRIVE_BACKUP=True` in the final cell before Run All, after
configuring the secrets below. This cell first validates the **complete** run;
two-sample smoke outputs are not accepted as a finished dataset.
Kaggle cannot use Colab's `google.colab.drive.mount()`.

The [Drive helper](../../src/utils/notebook_preprocessing/drive_backup.py)
creates a private `GRAG preprocessing backups` folder using the limited
[`drive.file` OAuth scope](https://developers.google.com/workspace/drive/api/guides/api-specific-auth).
It does not request access to every file in your Drive, make files public, or
delete/overwrite unrelated Drive files. It creates versioned snapshots rather
than performing a destructive mirror.

#### One-time OAuth setup (on your own computer, not inside Kaggle)

1. In [Google Cloud Console](https://console.cloud.google.com/), create/select a
   project and enable **Google Drive API**.
2. Configure Google Auth Platform's consent screen/audience for personal use.
   If the app is in Testing, add your own Google account as a test user. Declare
   `https://www.googleapis.com/auth/drive.file` under Data Access.
3. Create an OAuth client with application type **Desktop app**. Download its
   client JSON to a private location **outside this repository**. Keep the
   notebook private while configuring credentials.
4. With `google-auth-oauthlib` available in your local Python, run the following
   **locally**, adjusting the private client-file path. It opens your local browser
   for consent and receives the redirect on localhost. No password/token should
   be entered into a notebook cell or sent in chat.

```python
import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    str(Path.home() / "private-google-oauth" / "client_secret.json"),
    scopes=["https://www.googleapis.com/auth/drive.file"],
)
credentials = flow.run_local_server(
    port=0, access_type="offline", prompt="consent",
)
if not credentials.refresh_token:
    raise RuntimeError("No refresh token returned; repeat offline consent.")
token_file = Path.home() / "private-google-oauth" / "drive-token.json"
descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    handle.write(credentials.to_json())
print("Saved credentials privately. Do not upload this file to a notebook or Git.")
```

The destination directory must already exist and be private. The example refuses
to overwrite an existing token file. Install `google-auth-oauthlib` only if that
local import is missing. This local authorization step does not recreate the
authors' environment or change Kaggle's Python.
See Google's [desktop OAuth flow](https://developers.google.com/identity/protocols/oauth2/native-app)
for the consent/redirect requirements.

5. In Kaggle **Add-ons > Secrets**, add and enable for your notebook:

   | Kaggle Secret | Value from your private OAuth files |
   |---|---|
   | `GDRIVE_CLIENT_ID` | `client_id` |
   | `GDRIVE_CLIENT_SECRET` | `client_secret` |
   | `GDRIVE_REFRESH_TOKEN` | `refresh_token` from the generated token file |

   Never paste these values into source code, outputs, Git, or chat. The notebook
   reads them through `UserSecretsClient` and refreshes the access token in memory.
   OAuth apps in Testing commonly receive refresh tokens that expire after seven
   days for Drive access; revoked/expired credentials need renewed consent.
   Follow your account/organization's OAuth restrictions rather than bypassing them.

#### What is saved, and how to restore it for the LLM task

- Prepared node/edge CSVs, records, source metadata, and original split IDs.
- All compact retrieved samples: their node/relation features, query embeddings,
  expansion maps, graph edges, hard descriptions, questions, and gold labels.
- Run configuration and session provenance. The source commit/helper hash lets
  you restore the correct code; model weights and source code are not uploaded.

Logs, package listings, credentials, HF caches, and legacy full-graph caches are
excluded. The helper builds/uploads one compressed archive part at a time,
targeting 256 MiB of source data per part; an individual larger file can exceed
that target. It requires space for that part plus 1 GiB reserve, not a second
complete dataset copy. Each upload is resumable within the request and its remote
size/MD5 is verified. Parts also receive SHA256 checksums in `manifest.json`,
uploaded **last**. Re-running after a failure reuses matching finished parts
(an interrupted in-flight part may restart). Source data are never deleted.

Download every `part-*.tar.gz` plus `manifest.json` from the printed Drive folder.
Verify each archive's SHA256 against `manifest["archives"]` before extraction.
Extract the verified archives at the repository root; their paths start with
`output/<prepared-folder>/...` and `output/<run-folder>/...`.
Use the same original Kaggle working paths listed in `manifest["restore_paths"]`
and the matching helper version because resume checks deliberately reject changed
paths/code/settings. Then `pipeline.CompactDataset(RUN)` provides the inputs for
the downstream GraphLLM classifier (ExplaGraphs) or QA task (WebQSP).
The existing training registry still needs that explicit adapter.

The upload happens only when the final cell executes; it cannot protect an
unfinished run if Kaggle kills the session earlier. Keep Kaggle Save Version /
Dataset outputs as an additional backup. Drive quota or authentication/network
errors stop explicitly and leave the local outputs intact for retry.
Live OAuth/Drive access requires your account and cannot be verified locally
without it. The offline upload/archive regression tests use a fake Drive service:

```bash
python -m unittest discover -s tests -p 'test_*preprocess.py'
python -m unittest discover -s tests -p test_drive_backup.py
```

## Legacy full-cache workflow (reference only)

The remaining sections describe the older, much larger on-disk format and the
original training path. **Do not run Stages A-D below after the compact notebook**
unless you intentionally want a separate full-cache experiment. They are not
required by either updated notebook.

## Scope and reproducibility

These instructions move dataset download and preprocessing to Kaggle; no local
Mac training environment is required. They use **Kaggle's preinstalled Python
environment** and install only the packages that are missing, instead of
rebuilding the authors' environment (Python 3.9, PyTorch 2.0.1, CUDA 11.8; see
[requirements.txt](../../requirements.txt)). Package versions therefore differ
from the paper's setup and Kaggle updates them over time. Preprocessing is
simple tensor/CSV work and should be insensitive to this, but nothing here is a
GPU-tested environment. [requirements-kaggle.txt](../../requirements-kaggle.txt)
is an optional pinned set for training, **not an authors-provided lockfile**.

The paper used four NVIDIA A10G GPUs. Two 16 GB T4s are not one 32 GB GPU, and
training with the repository unchanged is not guaranteed to fit or run.
This guide prepares data first; see the training blockers at the end.
Full SentenceBERT embedding and ego-graph retrieval can take a long time and use
substantial disk space. Start with the one-sample smoke check, not the full job.

## 1. Configure the Kaggle notebook

- Enable **Internet** and select **GPU T4 x2**.
- Use a **private notebook** while configuring access to the gated Llama-2 model.
- Accept the Llama-2 access conditions on Hugging Face with your own account.
  Add its read token as a Kaggle Secret named `HF_TOKEN`; never paste it into a
  cell or commit it. The public WebQSP and SentenceBERT downloads do not need it.
- Put the workspace and generated artifacts under `/kaggle/working`, not the
  read-only `/kaggle/input`.

The following are notebook cells. They use subprocesses with `check=True` so a
failed command stops explicitly. GRAG commands run in subprocesses with the
notebook's own Python (`sys.executable`).

### Clone the working branch

First commit/push the setup files in this change, or upload them to the Kaggle
checkout separately. A remote clone cannot see uncommitted local files.

```python
import subprocess
from pathlib import Path

ROOT = Path("/kaggle/working/BioGRAG-QA")
subprocess.run(
    ["git", "clone", "--branch", "GRAG/reproduce/WebQSP",
     "https://github.com/MichaelPham2005/BioGRAG-QA.git", str(ROOT)],
    check=True,
)
print(subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip())
```

Run the clone once in a fresh session. If the folder already exists, inspect it
rather than deleting it or cloning over it.

### Use Kaggle's environment

```python
import subprocess, sys

PY = sys.executable
subprocess.run(
    [PY, "-m", "pip", "install", "-q", "torch_geometric", "gensim"],
    check=True,
)

def run(code):
    subprocess.run([PY, "-u", "-c", code], cwd=ROOT, check=True)
```

Kaggle already provides `torch`, `transformers`, `datasets`, `pandas`,
`scikit-learn`, `networkx` and `joblib`. Preprocessing (Stages A-D) also needs
`torch_geometric` and `gensim` (imported by
[lm_modeling.py](../../src/utils/lm_modeling.py)), installed above. Training
additionally needs `peft`, `torch_scatter` and `wandb`; install those only when
you reach training (see the end of this guide), because `torch_scatter` wheels
must match Kaggle's torch/CUDA versions.

The IPython warning `History will not be written to the database` concerns
notebook input-history storage, not the dataset or packages. If it persists,
save the notebook and restart the session.

### Validate the environment before downloading models

```python
run("""
import torch, datasets, transformers, gensim, networkx, joblib, sklearn
from torch_geometric.nn import GATConv
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__, "datasets:", datasets.__version__)
assert torch.cuda.is_available(), "Enable the Kaggle GPU accelerator."
print("GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 2**30, 2), "GiB")
conv = GATConv(4, 4, heads=2, concat=False).cuda()
out = conv(torch.ones(2, 4, device="cuda"),
           torch.tensor([[0, 1], [1, 0]], device="cuda"))
assert out.shape == (2, 4)
print("Imports and basic CUDA graph operations passed.")
""")
```

Do not continue if this fails. Save the error and the printed versions.

## 2. Download and inspect the raw dataset

```python
run("""
from datasets import load_dataset

ds = load_dataset("rmanluo/RoG-webqsp")
print(ds)
for split, rows in ds.items():
    print(split, len(rows), rows.column_names)
    assert {"question", "answer", "graph"} <= set(rows.column_names)
    assert isinstance(rows[0]["question"], str)
    assert isinstance(rows[0]["answer"], list)
    assert all(len(triple) == 3 for triple in rows[0]["graph"])
ds.save_to_disk("dataset/webqsp/raw")
row = ds["train"][0]
print("question:", row["question"])
print("answer:", row["answer"])
print("first three graph triples:", row["graph"][:3])
""")
```

This both populates the Hugging Face cache (used by the original scripts) and
saves a convenient Arrow snapshot under `dataset/webqsp/raw`. The original
scripts call `load_dataset` themselves; they **do not** read that saved snapshot.
Keep the cache in the same Kaggle session when running the next stages.

Upstream dataset revision observed during setup:
`c0632533135a06f8c5d536b420deec5fcb5c58f3`.
The original scripts do not pin a dataset revision. Record the downloaded
split lengths and fingerprints, rather than assuming a future `main` is
identical. For a strict frozen-data experiment, revision pinning must be made
consistent in **all** dataset-loading call sites.

```python
run("""
import json
from datasets import load_from_disk
ds = load_from_disk("dataset/webqsp/raw")
summary = {s: {"rows": len(d), "fingerprint": d._fingerprint}
           for s, d in ds.items()}
with open("dataset/webqsp/raw/download-summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(summary)
""")
```

## 3. Preprocess in stages

The actual source flow is:

```text
HF question/answer/graph triples
    |
    +-- step_one --> nodes/{i}.csv + edges/{i}.csv
    |
    +-- generate_split --> split/*_indices.txt
    |
    +-- step_two --> q_embs.pt + graphs/{i}.pt
    |
    +-- retrieval preprocess --> sims_2hop_none/{i}.pt
                             + cached_graphs/{i}.pt
                             + cached_desc/{i}.txt
```

Sample index `i` is the row index after concatenating **train, validation,
test in that order**. The preserved split offsets are created by
[generate_split](../../src/dataset/preprocess/webqsp.py#L59).
No graph is extracted from question text: the input dataset already supplies
the graph triples. `answer` becomes the training `label`; `q_entity` and
`a_entity` are not used in this pipeline.

### Stage A: text tables and split indices

```python
run("""
from src.dataset.preprocess.webqsp import step_one, generate_split
step_one()
generate_split()
""")
```

[step_one](../../src/dataset/preprocess/webqsp.py#L33) lowercases head/tail
entity text, assigns IDs in first-seen order, and saves:

```text
nodes/{i}.csv                edges/{i}.csv
node_id,node_attr            src,edge_attr,dst
0,entity a                  0,relation.name,1
1,entity b
```

**Import caveat:** importing `src.dataset` also imports
[WebQSPDataset](../../src/dataset/__init__.py#L2), which imports retrieval
utilities and [loads SentenceBERT at module scope](../../src/utils/graph_retrieval.py#L12).
Even Stage A can therefore download/load the encoder. This is existing code
behavior, not a requirement of CSV construction itself.

#### Optional: chart average nodes/edges per query

Run after Stage A, in the notebook itself (not through `run()`), so it uses
the absolute `ROOT` path. It reads only the CSVs (no SBERT, no GPU) and saves a
figure plus a summary table to `dataset/webqsp/stats/`.

```python
import glob, os
import pandas as pd
import matplotlib.pyplot as plt

root = str(ROOT / "dataset/webqsp")  # ROOT from the clone cell; notebook cwd is not the repo
n = len(glob.glob(f"{root}/nodes/*.csv"))
split = {}
for name in ["train", "val", "test"]:
    with open(f"{root}/split/{name}_indices.txt") as f:
        for line in f:
            if line.strip():
                split[int(line)] = name

def count(path):
    try:
        return len(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return 0

df = pd.DataFrame({
    "idx": range(n),
    "nodes": [count(f"{root}/nodes/{i}.csv") for i in range(n)],
    "edges": [count(f"{root}/edges/{i}.csv") for i in range(n)],
})
df["split"] = df["idx"].map(split)

summary = df.groupby("split")[["nodes", "edges"]].mean()
summary.loc["all"] = df[["nodes", "edges"]].mean()
print(summary.round(1))
print(df[["nodes", "edges"]].describe().round(1))

os.makedirs(f"{root}/stats", exist_ok=True)
summary.to_csv(f"{root}/stats/avg_nodes_edges.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
summary.plot.bar(ax=axes[0], rot=0)
axes[0].set_title("Average nodes / edges per query")
axes[0].set_ylabel("count")
axes[1].hist([df["nodes"], df["edges"]], bins=50, label=["nodes", "edges"])
axes[1].set_title("Distribution per query")
axes[1].set_xlabel("count")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{root}/stats/avg_nodes_edges.png", dpi=150)
plt.show()
```

Rows in `nodes/{i}.csv` / `edges/{i}.csv` are the unique entities and
triples of the full per-question graph (before retrieval). The `val` split
name must match the file `val_indices.txt` written by `generate_split`.

### Stage B: check one graph and SentenceBERT before the full embedding job

```python
run("""
import pandas as pd
import torch
from src.utils.lm_modeling import load_model, load_text2embedding

nodes = pd.read_csv("dataset/webqsp/nodes/0.csv")
edges = pd.read_csv("dataset/webqsp/edges/0.csv")
model, tokenizer, device = load_model["sbert"]()
encode = load_text2embedding["sbert"]
texts = nodes.node_attr.fillna("").head(3).tolist()
x = encode(model, tokenizer, device, texts)
assert x.shape == (len(texts), 1024), x.shape
assert torch.isfinite(x).all()
assert torch.allclose(x.norm(dim=1), torch.ones(len(texts)), atol=1e-4)
assert edges[["src", "dst"]].to_numpy().min() >= 0
assert edges[["src", "dst"]].to_numpy().max() < len(nodes)
print("Text encoding and graph-0 endpoint checks passed:", x.shape)
""")
```

These assertions matter because
[sber_text2embedding](../../src/utils/lm_modeling.py#L106) catches errors and
returns an empty tensor instead of raising.

### Stage C: embed all questions, nodes and edges

**Recommended on Kaggle (2 GPUs, resumable).** The original `step_two` is a
single process on one GPU and re-embeds every repeated relation string, which
measured ~52 s per graph (~67 h) on a T4. The sharded variant
[webqsp_sharded.py](../../src/dataset/preprocess/webqsp_sharded.py) runs one
process per GPU, embeds each unique string once per graph, and skips finished
graphs so it can be restarted. Stop the running `step_two` cell first.

**1. Launch (run once).** It refuses to start if workers are already running,
because two workers on the same shard corrupt each other's output.

```python
import os, subprocess

stale = subprocess.run(["pgrep", "-f", "webqsp_sharded"],
                       capture_output=True, text=True).stdout.split()
if stale:
    raise RuntimeError(f"Workers already running: PIDs {stale}. "
                       "Use the stop cell below first.")

NUM_SHARDS = 2
procs = []
for shard in range(NUM_SHARDS):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
    log = open(f"/kaggle/working/embed_shard{shard}.log", "w")
    procs.append(subprocess.Popen(
        [PY, "-u", "-m", "src.dataset.preprocess.webqsp_sharded",
         "--shard", str(shard), "--num_shards", str(NUM_SHARDS),
         "--batch_size", "64", "--fp16"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    ))
print("launched PIDs:", [p.pid for p in procs])
```

**2. Live status.** Prints progress every 30 s until both workers finish (or
one fails). Interrupting this cell does not stop the workers; rerun it to
resume watching.

```python
import re, time
from pathlib import Path

total = 4700
graphs = Path(ROOT) / "dataset/webqsp/graphs"
while True:
    codes = [p.poll() for p in procs]
    done = len(list(graphs.glob("*.pt")))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")
    last = []
    for s in range(NUM_SHARDS):
        txt = open(f"/kaggle/working/embed_shard{s}.log", errors="replace").read()
        bars = re.findall(r"shard \d+:[^\r\n]*", txt.replace("\r", "\n"))
        last.append(bars[-1] if bars else "(no progress yet)")
    print(time.strftime("%H:%M:%S"), f"graphs {done}/{total}", "exit:", codes)
    for s, l in enumerate(last):
        print(f"  {l}   GPU: {gpu[s].strip()}")
    if any(c not in (None, 0) for c in codes):
        raise RuntimeError("A worker failed; read /kaggle/working/embed_shard*.log")
    if all(c == 0 for c in codes):
        break
    time.sleep(30)
print("all workers finished")
```

Both exit codes must be `0`. If the kernel restarts, `procs` is lost but any
workers keep running: check with `pgrep -f webqsp_sharded`, count
`graphs/*.pt` yourself, or stop and relaunch (finished graphs are skipped).

**Stop stray workers** (e.g. after launching twice, or before relaunching):

```python
subprocess.run(["pkill", "-f", "webqsp_sharded"])
```

Then run the validation cell below, then Stage D.

`--fp16` changes embeddings by small floating-point amounts; drop it for
strict numerical fidelity (slower). Outputs are the same files as `step_two`.

Original single-GPU version:

```python
run("""
import src.utils.lm_modeling as lm
from src.dataset.preprocess.webqsp import step_two
lm.batch_size = 16
step_two()
""")
```

The original embedding batch size is 256. Here it is **explicitly reduced to
16 for T4 memory**, in this subprocess only. This changes execution batching,
not the encoder architecture or retrieval settings; small floating-point
differences remain possible. If it still runs out of memory, stop and lower
the embedding batch size further. Do not silently accept empty embeddings.

[step_two](../../src/dataset/preprocess/webqsp.py#L85) uses
`sentence-transformers/all-roberta-large-v1`, not Llama:

- `q_embs.pt`: `[number_of_questions, 1024]`
- `graphs/{i}.pt`: `x [V,1024]`, `edge_index [2,E]`, `edge_attr [E,1024]`
- Raw graphs do not yet have repeated `question_node`/`question_edge` fields.
  Retrieval adds those later.

Preserve the author's preprocessing when targeting their results:
[remove_repeated_edges_tensor](../../src/dataset/preprocess/webqsp.py#L17)
removes **all** edges for endpoint pairs occurring more than once, not merely
all but one duplicate. This affects the graph; do not silently "correct" it
within a baseline run.

Validate every generated graph before retrieval:

```python
run("""
from pathlib import Path
import torch
from datasets import load_from_disk

ds = load_from_disk("dataset/webqsp/raw")
count = sum(len(d) for d in ds.values())
q = torch.load("dataset/webqsp/q_embs.pt", map_location="cpu")
assert q.shape == (count, 1024), q.shape
assert torch.isfinite(q).all()
root = Path("dataset/webqsp/graphs")
assert len(list(root.glob("*.pt"))) == count
for i in range(count):
    g = torch.load(root / f"{i}.pt", map_location="cpu")
    assert g.x.shape == (g.num_nodes, 1024), i
    assert g.edge_index.shape[0] == 2, i
    assert g.edge_attr.shape == (g.edge_index.shape[1], 1024), i
    assert torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all(), i
    if g.edge_index.numel():
        assert int(g.edge_index.min()) >= 0, i
        assert int(g.edge_index.max()) < g.num_nodes, i
print("Validated", count, "base graphs.")
""")
```

The full embedding stage has no built-in per-graph resume check; rerunning
`step_two` redoes the job. Do not start it casually near the session time limit.

### Stage D: retrieve query-specific ego-graphs

```python
subprocess.run(
    [PY, "-u", "-m", "src.dataset.webqsp"], cwd=ROOT, check=True
)
```

The [entry point](../../src/dataset/webqsp.py#L102) uses:

| Parameter | Value | Meaning |
|---|---|---|
| `k` | 2 | Hops around each ego-node |
| `topk` | 20 | Number of ego-graph centers selected |
| `topk_entity` | 10 | Node/edge text rows selected per ego-graph |
| `augment` | `"none"` | Mean-pool existing text embeddings for ranking |

Important: `topk_entity` selects text for the hard prompt; it does not delete
those unselected nodes/edges from the soft graph. The selected ego-graphs are
concatenated with shifted node indices, not deduplicated into a set union.
Learned soft pruning happens later in the GNN.
See [find_topk_subgraph](../../src/utils/graph_retrieval.py#L154) and
[merge_graphs](../../src/utils/graph_retrieval.py#L110).

Retrieval saves final graph/text pairs and skips an index if its graph file
already exists. Cache folder names do not encode all parameters; changing
`topk` or `topk_entity` with old caches present will not reliably recompute the
results. Keep separate saved output versions for each experiment.

### Stage E: validate training inputs

```python
run("""
import torch
from src.dataset.webqsp import WebQSPDataset

ds = WebQSPDataset()
split = ds.get_idx_split()
ids = [i for name in ("train", "val", "test") for i in split[name]]
assert sorted(ids) == list(range(len(ds))), "Splits must partition all samples."
for i in range(len(ds)):
    s = ds[i]
    g = s["graph"]
    assert g.x.shape == (g.num_nodes, 1024), i
    assert g.edge_attr.shape == (g.edge_index.shape[1], 1024), i
    assert g.question_node.shape == g.x.shape, i
    assert g.question_edge.shape == g.edge_attr.shape, i
    for t in (g.x, g.edge_attr, g.question_node, g.question_edge):
        assert torch.isfinite(t).all(), i
    if g.edge_index.numel():
        assert int(g.edge_index.min()) >= 0, i
        assert int(g.edge_index.max()) < g.num_nodes, i
    assert isinstance(s["desc"], str) and s["desc"].strip(), i
    assert isinstance(s["label"], str) and s["label"].strip(), i
print("Validated", len(ds), "cached training samples.")
print({k: len(v) for k, v in split.items()})
""")
```

If a validation assertion fails, stop and diagnose that sample before training.
The author's retrieval code contains broad exception fallbacks; process exit
status alone is not sufficient to validate the resulting data.

## 4. Preserve data across Kaggle sessions

Save generated data as a **private Kaggle Dataset/output** for the next
session. Retain the node/edge tables, split files, embeddings, retrieval scores,
cached graphs, and cached descriptions together, plus:

- source commit hash;
- dataset summary from Stage 2;
- preprocessing settings and any embedding batch-size change;
- an environment package listing.

```python
with open("/kaggle/working/grag-environment.txt", "w") as f:
    subprocess.run(
        [PY, "-m", "pip", "freeze"],
        stdout=f, check=True,
    )
```

The original loaders use relative paths under `dataset/webqsp`; when restoring
saved outputs from `/kaggle/input`, copy them back to that layout or deliberately
adapt the paths. Do not commit generated data, downloaded model weights, or
tokens to GitHub.

## 5. Training on 2 x T4: blockers before claiming reproduction

The author's baseline invocation, **after** validated preprocessing, is:

```bash
python train.py --dataset webqsp --model_name graph_llm --llm_frozen True --seed 3
```

This is a reference command, **not a verified T4-ready training command**.
Before training, install the extra packages (not needed for preprocessing),
e.g. `pip install peft wandb`, and a `torch_scatter` wheel matching Kaggle's
installed torch/CUDA from <https://data.pyg.org/whl/>; the authors used
`torch-scatter 2.1.2` with torch 2.0.1.
The following need a separate, tested compatibility pass:

1. [GraphLLM model loading](../../src/model/graph_llm.py#L32) hard-codes
   four devices with `20GiB` each. On T4 x2, configure only the available devices
   and leave headroom for graph parameters, gradients, optimizer states and
   activations; a weight-placement budget is not a training-memory limit.
2. [maybe_autocast](../../src/model/graph_llm.py#L100) requests BF16 by
   default; T4 does not provide native BF16 support. FP16/FP32 behavior and
   numerical stability need validation, not just changing the dtype string.
3. Default training batch size is 2 and evaluation batch size is 16. Memory
   limits may require smaller batches/checkpointing. Reducing training batch size
   changes the experiment. `grad_steps` is **not true gradient accumulation**:
   [train.py](../../train.py#L65) zeroes and steps the optimizer every batch.
4. Frozen Llama still needs differentiable computation to train the graph
   encoder/projector. Do not wrap the entire Llama forward pass in `no_grad`.
5. Access to `meta-llama/Llama-2-7b-hf` is gated. To authenticate the subprocesses,
   load the Kaggle Secret into the notebook environment without printing it:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

6. W&B needs configuration. Use `os.environ["WANDB_MODE"] = "offline"` if you do
   not want online logging. Do not embed API keys in the notebook.
7. Quantizing Llama to 4/8 bits or replacing it with another model may improve
   fit, but is a **different configuration**, not exact reproduction.

Use the frozen-LLM baseline first; treat LoRA as a separate experiment after
the baseline has passed a forward/backward smoke test on the actual GPUs.
The repository's `device_map="auto"` is not a DDP/`torchrun` training setup.

No Kaggle runtime was available during preparation of this guide. Neither the
GPU environment, full dataset download, full preprocessing, nor training has
been executed here. The cells above are the steps to execute and validate on
your Kaggle session, not a report of completed cloud execution.
