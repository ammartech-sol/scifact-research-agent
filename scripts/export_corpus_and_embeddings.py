"""One-off export of SciFact data for the agent (run once, in the Project 2 environment).

Writes three files into data/:
  corpus.jsonl         one document per line: doc_id, title, text
  doc_embeddings.npy   one 384-number vector per document, same order as corpus.jsonl
  claims.jsonl         the 300 test claims with their gold document ids

The agent itself never imports ir_datasets and never downloads anything at runtime.
"""

import json
from pathlib import Path

import ir_datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DATA_DIR = Path("data")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    # newline="\n" keeps Linux-style line endings even though this runs on Windows.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    dataset = ir_datasets.load("beir/scifact/test")

    docs = [
        {"doc_id": d.doc_id, "title": d.title, "text": d.text}
        for d in dataset.docs_iter()
    ]
    write_jsonl(DATA_DIR / "corpus.jsonl", docs)
    print("documents:", len(docs))

    gold: dict[str, list[str]] = {}
    for row in dataset.qrels_iter():
        gold.setdefault(row.query_id, []).append(row.doc_id)
    claims = [
        {"query_id": q.query_id, "text": q.text, "gold_doc_ids": gold[q.query_id]}
        for q in dataset.queries_iter()
        if q.query_id in gold
    ]
    write_jsonl(DATA_DIR / "claims.jsonl", claims)
    print("claims with a gold abstract:", len(claims))

    # Same text as Project 2: title, a full stop and a space, then the abstract.
    texts = [d["title"] + ". " + d["text"] for d in docs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("embedding on:", device)
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
    assert vectors.shape == (len(docs), 384), vectors.shape
    np.save(DATA_DIR / "doc_embeddings.npy", vectors)

    norms = np.linalg.norm(vectors, axis=1)
    print("vector shape:", vectors.shape, "norm min/max:", norms.min(), norms.max())
    for name in ["corpus.jsonl", "claims.jsonl", "doc_embeddings.npy"]:
        print(name, (DATA_DIR / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()