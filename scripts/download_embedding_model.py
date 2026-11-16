"""Download the embedding model into the image at build time.

The revision is pinned to the snapshot used for the Project 2 results, so the
weights in the image are the ones those numbers came from.
"""

import os

from huggingface_hub import snapshot_download

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def main() -> None:
    target_dir = os.environ["EMBEDDING_MODEL_PATH"]
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=target_dir,
        # Skip duplicate copies of the same weights in other formats.
        ignore_patterns=["*.onnx", "onnx/*", "pytorch_model.bin", "tf_model.h5", "*.msgpack", "*.ot"],
    )
    print(f"model saved to {target_dir}")


if __name__ == "__main__":
    main()