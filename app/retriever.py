"""Hybrid retriever: BM25 and dense search merged with reciprocal rank fusion.

Same behaviour as Project 2 (bm25_search.py, dense_search.py, hybrid_search.py), but it
reads the exported data folder instead of ir_datasets, and it loads the document vectors
from a file instead of computing them at start-up.
"""

import json
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class HybridRetriever:
    def __init__(self, documents: list[dict], doc_vectors: np.ndarray, embedder, config: dict):
        if len(documents) != len(doc_vectors):
            raise ValueError(
                f"{len(documents)} documents but {len(doc_vectors)} vectors; the files do not match"
            )
        self._documents = {d["doc_id"]: d for d in documents}
        self._doc_ids = [d["doc_id"] for d in documents]
        self._doc_vectors = doc_vectors
        self._embedder = embedder

        self._query_prefix = config["dense"]["query_prefix"]
        self._normalize = config["dense"]["normalize_embeddings"]
        self._rrf_k = config["fusion"]["rrf_k"]
        self._first_stage_k = config["first_stage_k"]
        self._stopwords = config["bm25"]["stopwords"]
        self._stemmer = Stemmer.Stemmer(config["bm25"]["stemmer"])

        # Same indexed text as Project 2: title, a full stop and a space, then the abstract.
        texts = [d["title"] + ". " + d["text"] for d in documents]
        tokens = bm25s.tokenize(texts, stopwords=self._stopwords, stemmer=self._stemmer)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens)

    def _bm25_ranked(self, query: str) -> list[str]:
        tokens = bm25s.tokenize(query, stopwords=self._stopwords, stemmer=self._stemmer)
        positions, _ = self._bm25.retrieve(tokens, k=self._first_stage_k, show_progress=False)
        return [self._doc_ids[p] for p in positions[0]]

    def _dense_ranked(self, query: str) -> list[str]:
        vector = self._embedder.encode(self._query_prefix + query, normalize_embeddings=self._normalize)
        scores = self._doc_vectors @ vector
        best = np.argsort(-scores)[: self._first_stage_k]
        return [self._doc_ids[i] for i in best]

    def search(self, query: str, top_k: int) -> list[str]:
        """Return the ids of the top_k best documents, best first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        points: dict[str, float] = {}
        # BM25 first, then dense: the same order as Project 2, so ties break the same way.
        for ranked in (self._bm25_ranked(query), self._dense_ranked(query)):
            for position, doc_id in enumerate(ranked, start=1):
                points[doc_id] = points.get(doc_id, 0) + 1 / (self._rrf_k + position)
        merged = sorted(points, key=points.get, reverse=True)
        return merged[:top_k]

    def get_document(self, doc_id: str) -> dict | None:
        """Return {doc_id, title, text} for an id, or None if there is no such document."""
        return self._documents.get(doc_id)


def load_retriever(
    embedder,
    data_dir: Path = REPO_ROOT / "data",
    config_path: Path = REPO_ROOT / "config.json",
) -> HybridRetriever:
    documents = read_jsonl(data_dir / "corpus.jsonl")
    doc_vectors = np.load(data_dir / "doc_embeddings.npy")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return HybridRetriever(documents, doc_vectors, embedder, config)