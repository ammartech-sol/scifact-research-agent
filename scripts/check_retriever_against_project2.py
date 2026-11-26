"""Check that the new retriever reproduces the Project 2 hybrid results.

Run from the repo root:
    python -m scripts.check_retriever_against_project2 "path\\to\\Project 2 results.json"
The path is optional. With it, every claim is compared with Project 2's per-claim recall.
"""

import json
import sys
import time
from pathlib import Path
from statistics import median

from app.embedder import load_embedder
from app.retriever import REPO_ROOT, load_retriever, read_jsonl

KS = (5, 10, 20)
# Project 2 hybrid, from its results.json summary.
PROJECT2_MEANS = {5: 0.7780555555555555, 10: 0.8587222222222223, 20: 0.8885555555555555}


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def main() -> None:
    project2_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    started = time.perf_counter()
    retriever = load_retriever(load_embedder())
    print(f"loaded model, vectors and BM25 index in {time.perf_counter() - started:.1f} s")

    claims = read_jsonl(REPO_ROOT / "data" / "claims.jsonl")
    mine: dict[int, dict[str, float]] = {k: {} for k in KS}
    search_ms = []
    for claim in claims:
        gold = set(claim["gold_doc_ids"])
        t = time.perf_counter()
        ranked = retriever.search(claim["text"], top_k=max(KS))
        search_ms.append((time.perf_counter() - t) * 1000)
        for k in KS:
            mine[k][claim["query_id"]] = recall_at(ranked, gold, k)

    print("\nmean recall, this retriever vs Project 2 hybrid:")
    for k in KS:
        mean = sum(mine[k].values()) / len(mine[k])
        print(f"  recall@{k}: {mean:.4f} vs {PROJECT2_MEANS[k]:.4f} (difference {mean - PROJECT2_MEANS[k]:+.4f})")
    print(f"median search time: {median(search_ms):.1f} ms on CPU (Project 2: about 10 ms, GPU)")

    if project2_path is None:
        print("no Project 2 results.json given, so the claim-by-claim comparison was skipped")
        return
    theirs = json.loads(project2_path.read_text(encoding="utf-8"))["hybrid"]["per_query"]
    print("\nclaim-by-claim:")
    for k in KS:
        differing = [q for q, v in mine[k].items() if abs(v - theirs[f"recall@{k}"][q]) > 1e-9]
        print(f"  recall@{k}: {len(differing)} of {len(mine[k])} claims differ; first ones: {differing[:10]}")


if __name__ == "__main__":
    main()