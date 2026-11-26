#!/usr/bin/env python3
"""Download a pinned, public original Granite base model, not a LoRA merge."""
import json
import os
import sys
from pathlib import Path

BASE_REPO = "ibm-granite/granite-4.2-8b"
# IBM's publicly documented 2026-09 model snapshot; override if training metadata requires another revision.
REVISION = os.environ.get(
    "EVIDENCITE_GRANITE_REVISION",
    "7fce579a7fbbad4b1e7703b6850cefd517a4002b",
)


def complete(root: Path) -> bool:
    index = root / "model.safetensors.index.json"
    if not (root / "config.json").is_file() or not (root / "tokenizer.json").is_file() or not index.is_file():
        return False
    try:
        paths = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    except (ValueError, KeyError, OSError):
        return False
    return bool(paths) and all((root / name).is_file() and (root / name).stat().st_size > 0 for name in paths)


def main():
    if len(sys.argv) != 2:
        sys.exit("Uso: download_granite_base.py <directorio>")
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    if complete(root):
        print("[OK] Granite base complete; skipping download", flush=True)
        return
    from huggingface_hub import snapshot_download
    print(f"[HF] Downloading {BASE_REPO}@{REVISION} (about 17.6 GB)", flush=True)
    snapshot_download(
        repo_id=BASE_REPO,
        revision=REVISION,
        local_dir=str(root),
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "*.jinja"],
        max_workers=3,
    )
    if not complete(root):
        sys.exit("[STOP] Incomplete Granite base model (missing shards/tokenizer)")
    print("[OK] Granite base available", flush=True)


if __name__ == "__main__":
    main()
