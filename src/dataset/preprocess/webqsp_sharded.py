"""Resumable, multi-process variant of ``step_two`` in ``webqsp.py``.

Each process handles the indices ``i`` with ``i % num_shards == shard`` on its
own GPU, so two processes with CUDA_VISIBLE_DEVICES=0 / =1 use both GPUs.
Outputs (q_embs.pt, graphs/{i}.pt) match ``step_two``; existing files are skipped.
"""
import argparse
import os

import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from tqdm import tqdm

import src.utils.lm_modeling as lm
from src.dataset.preprocess.webqsp import (
    path, path_nodes, path_edges, path_graphs, remove_repeated_edges_tensor,
)


def embed_unique(text2embedding, model, tokenizer, device, texts):
    # Relation/entity strings repeat a lot inside one graph; embed each once.
    unique = list(dict.fromkeys(texts))
    if not unique:
        return torch.zeros((0, 1024))
    embs = text2embedding(model, tokenizer, device, unique)
    assert embs.shape == (len(unique), 1024), embs.shape
    lookup = {t: j for j, t in enumerate(unique)}
    return embs[[lookup[t] for t in texts]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    lm.batch_size = args.batch_size
    model, tokenizer, device = lm.load_model["sbert"]()
    if args.fp16:
        model.half()
    text2embedding = lm.load_text2embedding["sbert"]

    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )

    if args.shard == 0 and not os.path.exists(f"{path}/q_embs.pt"):
        questions = [d["question"] for d in dataset]
        q_embs = text2embedding(model, tokenizer, device, questions)
        assert q_embs.shape == (len(questions), 1024), q_embs.shape
        torch.save(q_embs.float(), f"{path}/q_embs.pt")

    os.makedirs(path_graphs, exist_ok=True)
    indices = range(args.shard, len(dataset), args.num_shards)
    for index in tqdm(indices, desc=f"shard {args.shard}"):
        out = f"{path_graphs}/{index}.pt"
        if os.path.exists(out):
            continue
        nodes = pd.read_csv(f"{path_nodes}/{index}.csv")
        edges = pd.read_csv(f"{path_edges}/{index}.csv")
        nodes.node_attr = nodes.node_attr.fillna("")

        x = embed_unique(text2embedding, model, tokenizer, device,
                         nodes.node_attr.tolist()).float()
        edge_attr = embed_unique(text2embedding, model, tokenizer, device,
                                 edges.edge_attr.tolist()).float()
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                     num_nodes=len(nodes))
        graph = remove_repeated_edges_tensor(graph)
        tmp = f"{out}.{os.getpid()}.tmp"  # unique per process
        torch.save(graph, tmp)
        os.replace(tmp, out)


if __name__ == "__main__":
    main()
