import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_PATH = "/opt/models/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it for every request."""
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", DEFAULT_MODEL_PATH)
    return SentenceTransformer(model_path, device="cpu")