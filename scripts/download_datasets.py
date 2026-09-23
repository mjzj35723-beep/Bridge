#!/usr/bin/env python3
"""Download the six benchmark inputs into a reproducible local layout.

SorryBench is gated on Hugging Face.  Pass --hf-token or set HF_TOKEN when
access has been granted; the other five families are public.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def save(name: str, config: str | None, out: Path) -> dict:
    kwargs = {"path": name, "cache_dir": str(out / "_hf_cache"), "download_mode": "reuse_dataset_if_exists"}
    if config is not None:
        kwargs["name"] = config
    ds = load_dataset(**kwargs)
    out.mkdir(parents=True, exist_ok=True)
    for split, table in ds.items():
        table.save_to_disk(str(out / split))
    return {"source": name, "config": config, "splits": {k: len(v) for k, v in ds.items()}, "path": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--skip-sorrybench", action="store_true")
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    report = {"datasets": {}}
    report["datasets"]['truthfulqa'] = save("truthful_qa", "multiple_choice", root / "truthfulqa_multiple_choice")
    report["datasets"]['mmlu'] = save("cais/mmlu", "all", root / "mmlu")
    report["datasets"]['arc_easy'] = save("allenai/ai2_arc", "ARC-Easy", root / "arc_easy")
    report["datasets"]['arc_challenge'] = save("allenai/ai2_arc", "ARC-Challenge", root / "arc_challenge")
    report["datasets"]['bbq'] = save("heegyu/bbq", None, root / "bbq")
    if args.skip_sorrybench:
        report["datasets"]['sorrybench'] = {"status": "skipped", "reason": "gated dataset"}
    else:
        local = snapshot_download(repo_id="sorry-bench/sorry-bench-202503", repo_type="dataset", token=args.hf_token, local_dir=str(root / "sorrybench"), local_dir_use_symlinks=False)
        report["datasets"]['sorrybench'] = {"source": "sorry-bench/sorry-bench-202503", "path": local, "status": "ok"}
    (root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
