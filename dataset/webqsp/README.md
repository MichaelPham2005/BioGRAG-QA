# WebQSP preparation on Kaggle (2 x T4)

## Scope and reproducibility

These instructions move dataset download and preprocessing to Kaggle; no local
Mac training environment is required. They propose a Python 3.10 environment
isolated from Kaggle's changing preinstalled packages. The original dependency
instructions use Python 3.9, PyTorch 2.0.1 and CUDA 11.8; see
[requirements.txt](../../requirements.txt). The additional pins in
[requirements-kaggle.txt](../../requirements-kaggle.txt) are a compatibility
starting point, **not an authors-provided lockfile or a GPU-tested environment**.

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
failed command stops explicitly. The notebook kernel stays unchanged; every
GRAG command runs using the separate environment's Python executable.

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

### Create an isolated environment

```python
import shutil
import subprocess
import sys
from pathlib import Path

UV = shutil.which("uv")
if UV is None:
    tools = Path("/kaggle/working/grag-tools")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", str(tools), "uv"],
        check=True,
    )
    UV = str(tools / "bin" / "uv")
    if not Path(UV).is_file():
        raise FileNotFoundError(f"uv installation did not create {UV}")

subprocess.run([UV, "--version"], check=True)

ENV = Path("/kaggle/working/grag-venv")
PY = str(ENV / "bin/python")
subprocess.run(
    [UV, "venv", "--python", "3.10", str(ENV)],
    check=True,
)

def install(*args):
    subprocess.run(
        [UV, "pip", "install", "--python", PY, *args],
        cwd=ROOT, check=True,
    )

install("torch==2.0.1", "--index-url", "https://download.pytorch.org/whl/cu118")
install("-r", "requirements-kaggle.txt")
install(
    "torch-scatter==2.1.2+pt20cu118",
    "--find-links", "https://data.pyg.org/whl/torch-2.0.1+cu118.html",
    "--only-binary", "torch-scatter",
)
subprocess.run(
    [UV, "pip", "check", "--python", PY], check=True
)

def run(code):
    subprocess.run([PY, "-u", "-c", code], cwd=ROOT, check=True)
```

Use the resolved `UV` executable throughout, not `sys.executable -m uv`.
Kaggle may have `uv` on `PATH` without the `uv` Python module being installed
in the notebook kernel's interpreter. If the executable is missing, the cell
installs it into the writable `/kaggle/working/grag-tools` directory rather
than modifying the system Python environment.

The IPython warning `History will not be written to the database` is separate:
it concerns notebook input-history storage, not the GRAG environment or dataset.
It does not explain `No module named uv`. If it persists, save the notebook and
restart the Kaggle session; do not change permissions on shared system files.

This installs the `torch_scatter` extension actually imported by
[GraphLLM](../../src/model/graph_llm.py#L6). The original requirements also list
other PyG extensions; add them only if a chosen operation reports a missing
extension. `torchvision` and `torchaudio` are not used in this WebQSP flow.

### Validate the environment before downloading models

```python
run("""
import torch, datasets, transformers, peft, gensim
from torch_scatter import scatter
from torch_geometric.nn import GATConv
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "Enable the Kaggle GPU accelerator."
assert torch.cuda.device_count() == 2, "Select T4 x2."
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 2**30, 2), "GiB")
x = torch.ones(3, 2, device="cuda")
batch = torch.tensor([0, 0, 1], device="cuda")
assert scatter(x, batch, dim=0, reduce="mean").shape == (2, 2)
conv = GATConv(4, 4, heads=2, concat=False).cuda()
out = conv(torch.ones(2, 4, device="cuda"),
           torch.tensor([[0, 1], [1, 0]], device="cuda"))
assert out.shape == (2, 4)
print("Imports and basic CUDA graph operations passed.")
""")
```

Do not continue if this fails. Save the error and the printed package/GPU
versions; successfully installing packages is not proof of GPU compatibility.

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

```python
import os, subprocess

procs = []
for shard in range(2):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
    log = open(f"/kaggle/working/embed_shard{shard}.log", "w")
    procs.append(subprocess.Popen(
        [PY, "-u", "-m", "src.dataset.preprocess.webqsp_sharded",
         "--shard", str(shard), "--num_shards", "2",
         "--batch_size", "64", "--fp16"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    ))
```

Monitor with `!tail -c 300 /kaggle/working/embed_shard0.log`, and wait with
`[p.wait() for p in procs]` (all return codes must be 0). `--fp16` changes
embeddings by small floating-point amounts; drop it for strict numerical
fidelity (slower). Outputs are the same files as `step_two`.

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
        [UV, "pip", "freeze", "--python", PY],
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
