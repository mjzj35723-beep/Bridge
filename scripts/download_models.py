#!/usr/bin/env python3
"""Download model checkpoints; weights are intentionally outside the release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "Qwen3-14B-Base": "Qwen/Qwen3-14B-Base",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama-2-13B": "NousResearch/Llama-2-13b-hf",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        repo = MODELS[name]
        path = snapshot_download(repo_id=repo, repo_type="model", token=args.hf_token, local_dir=str(root / name), local_dir_use_symlinks=False)
        report[name] = {"repo": repo, "path": path}
    (root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
