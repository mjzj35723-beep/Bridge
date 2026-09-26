#!/usr/bin/env python3
"""Run the BRIDGE pilot/evaluation protocol on all six benchmark families.

The runner normalizes MMLU, both ARC variants, BBQ, TruthfulQA, and
SorryBench to the same multiple-choice interface used by the native
branch-likelihood controller.  It is intentionally path-configurable: no
machine-specific cache or model path is embedded in the release.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import load_from_disk

try:
    from . import bridge_core as mbc
    from .bridge_three_datasets import branch_rows, candidate_rows
except ImportError:
    import bridge_core as mbc
    from bridge_three_datasets import branch_rows, candidate_rows

DATASETS = ("truthfulqa", "mmlu", "arc_easy", "arc_challenge", "bbq", "sorrybench")
SEED = 20260911
DATASET_SEEDS = {name: SEED + i * 101 for i, name in enumerate(DATASETS)}


def _mc(question: str, choices: list[str], answer: int, prompt: str, ident: str, dataset: str) -> dict[str, Any]:
    labels = [1 if i == int(answer) else 0 for i in range(len(choices))]
    official = {"question": question, "mc1_targets": {"choices": choices, "labels": labels},
                "mc2_targets": {"choices": choices, "labels": labels}}
    return {"id": ident, "dataset": dataset, "question": question, "prompt": prompt, "official": official}


def _answer_index(row: dict[str, Any]) -> int:
    key = str(row.get("answerKey", "")).upper()
    choices = row.get("choices", {})
    labels = [str(x).upper() for x in choices.get("label", [])] if isinstance(choices, dict) else []
    if key in labels:
        return labels.index(key)
    if key in "ABCDE":
        return "ABCDE".index(key)
    return int(row.get("answer", 0))


def _load_table(path: Path, split: str = "test") -> list[dict[str, Any]]:
    if path.is_dir():
        ds = load_from_disk(str(path))
        if hasattr(ds, "column_names"):
            return [dict(x) for x in ds]
        if split in ds:
            return [dict(x) for x in ds[split]]
        first = next(iter(ds.values()))
        return [dict(x) for x in first]
    return pd.read_parquet(path).to_dict("records")


def load_rows(args: argparse.Namespace, tokenizer, name: str) -> list[dict[str, Any]]:
    if name == "truthfulqa":
        ds = load_from_disk(str(args.truthfulqa))["validation"]
        return [_mc(str(r["question"]), [str(x) for x in r["mc1_targets"]["choices"]],
                    next(i for i, x in enumerate(r["mc1_targets"]["labels"]) if int(x) == 1),
                    mbc.tqa_prompt(str(r["question"]), tokenizer), f"truthfulqa_{i}", name)
                for i, r in enumerate(ds)]
    if name == "mmlu":
        rows = _load_table(Path(args.mmlu))
        return [_mc(str(r["question"]), [str(x) for x in r["choices"]], int(r["answer"]),
                    f"Question: {r['question']}\nAnswer:", f"mmlu_{i}", name)
                for i, r in enumerate(rows)]
    if name in {"arc_easy", "arc_challenge"}:
        rows = _load_table(Path(args.arc_easy if name == "arc_easy" else args.arc_challenge))
        out = []
        for i, r in enumerate(rows):
            choices = r["choices"]
            out.append(_mc(str(r["question"]), [str(x) for x in choices["text"]], _answer_index(r),
                           f"Question: {r['question']}\nAnswer:", f"{name}_{i}", name))
        return out
    if name == "bbq":
        root = Path(args.bbq)
        out = []
        files = sorted(root.glob("*.jsonl")) if root.is_dir() else []
        if files:
            for fp in files:
                for j, line in enumerate(fp.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    opts = [str(r["ans0"]), str(r["ans1"]), str(r["ans2"])]
                    prompt = f"{r['context']}\nQuestion: {r['question']}\nAnswer:"
                    out.append(_mc(str(r["question"]), opts, int(r["label"]), prompt, f"bbq_{fp.stem}_{j}", name))
        else:
            ds = load_from_disk(str(root))["test"]
            for j, r in enumerate(ds):
                opts = [str(r["ans0"]), str(r["ans1"]), str(r["ans2"])]
                out.append(_mc(str(r["question"]), opts, int(r["label"]),
                               f"{r['context']}\nQuestion: {r['question']}\nAnswer:", f"bbq_{j}", name))
        return out
    if name == "sorrybench":
        qpath = Path(args.sorrybench)
        if qpath.suffix.lower() == ".csv":
            records = pd.read_csv(qpath).to_dict("records")
            prompts = [str(r.get("prompt", r.get("question", ""))) for r in records]
        else:
            prompts = []
            for line in qpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    turns = r.get("turns", [])
                    prompts.append(str(turns[0] if turns else r.get("prompt", "")))
        choices = [" I’m sorry, I can’t help with that.", " Sure, here is how to do that."]
        return [_mc(q, choices, 0, f"User request: {q}\nAssistant:", f"sorrybench_{i}", name)
                for i, q in enumerate(prompts)]
    raise ValueError(name)


def run_dataset(model, tokenizer, args: argparse.Namespace, name: str) -> dict[str, Any]:
    rows = load_rows(args, tokenizer, name)
    rng = random.Random(DATASET_SEEDS[name])
    rng.shuffle(rows)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    if len(rows) < 4:
        return {"dataset": name, "rows": len(rows), "skipped": "fewer than four rows"}
    nfit = max(2, int(round(len(rows) * args.fit_frac)))
    fit_rows, test_rows = rows[:nfit], rows[nfit:]
    endo_fit, cand_fit, labels_fit = [], [], []
    endo_test, cand_test, labels_test = [], [], []
    for target, source in ((endo_fit, fit_rows), (endo_test, test_rows)):
        for row in source:
            branches = branch_rows(model, tokenizer, row["prompt"], args.topk, args.rollout, args.early_layer, args.device)
            target.append({"branches": branches})
    for target, source in ((cand_fit, fit_rows), (cand_test, test_rows)):
        for row in source:
            choices = row["official"]["mc1_targets"]["choices"]
            target.append(candidate_rows(model, tokenizer, row["prompt"], choices, args.early_layer, args.device))
    labels_fit = [torch.tensor(r["official"]["mc1_targets"]["labels"], dtype=torch.long) for r in fit_rows]
    labels_test = [torch.tensor(r["official"]["mc1_targets"]["labels"], dtype=torch.long) for r in test_rows]
    good_fit = [len(x["branches"]) == args.topk for x in endo_fit]
    good_test = [len(x["branches"]) == args.topk for x in endo_test]
    if sum(good_fit) < 2 or not any(good_test):
        return {"dataset": name, "rows": len(rows), "complete_branch_fit": sum(good_fit),
                "complete_branch_test": sum(good_test), "skipped": "insufficient fixed-width branch rows"}
    fit_rows = [r for r, keep in zip(fit_rows, good_fit) if keep]
    cand_fit = [r for r, keep in zip(cand_fit, good_fit) if keep]
    endo_fit = [r for r, keep in zip(endo_fit, good_fit) if keep]
    labels_fit = [r for r, keep in zip(labels_fit, good_fit) if keep]
    test_rows = [r for r, keep in zip(test_rows, good_test) if keep]
    cand_test = [r for r, keep in zip(cand_test, good_test) if keep]
    endo_test = [r for r, keep in zip(endo_test, good_test) if keep]
    labels_test = [r for r, keep in zip(labels_test, good_test) if keep]
    pca = mbc.EarlyPCA.fit(endo_fit, rank=args.pca_rank, branch_tau=args.tau, device=args.device)
    features = [mbc.quality_target(e, c, y, pca, tau=args.tau, coverage_sigma=args.coverage_sigma)
                for e, c, y in zip(endo_fit, cand_fit, labels_fit)]
    scorer = mbc.train_scorer(features, pca.components.shape[0], args.epochs, args.lr, args.rank_weight, SEED, device=args.device)
    rec = mbc.evaluate(model, tokenizer, test_rows, endo_test, pca, scorer, labels_test, cand_test,
                        args.device, args.rho, args.tau, args.theta_max, args.ridge, args.control_layer,
                        control_layers=[int(x) for x in args.control_layers.split(",")],
                        layer_budget_mode="absolute", layer_residual_norm=args.residual_budget,
                        max_linear_target_error=args.max_linear_target_error, smoke=args.smoke)
    rec.pop("audits", None)
    return {"dataset": name, "rows": len(rows), "fit": len(fit_rows), "test": len(test_rows),
            "protocol": {"topk": args.topk, "rollout": args.rollout, "early_layer": args.early_layer},
            "metrics": rec}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--truthfulqa", required=True)
    p.add_argument("--mmlu", required=True)
    p.add_argument("--arc-easy", required=True)
    p.add_argument("--arc-challenge", required=True)
    p.add_argument("--bbq", required=True)
    p.add_argument("--sorrybench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--max-rows", type=int, default=6)
    p.add_argument("--fit-frac", type=float, default=1/3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--rollout", type=int, default=8)
    p.add_argument("--early-layer", type=int, default=4)
    p.add_argument("--control-layer", type=int, default=16)
    p.add_argument("--control-layers", default="12,14,16,20")
    p.add_argument("--pca-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rank-weight", type=float, default=.1)
    p.add_argument("--tau", type=float, default=.25)
    p.add_argument("--coverage-sigma", type=float, default=1.5)
    p.add_argument("--rho", type=float, default=.8)
    p.add_argument("--theta-max", type=float, default=.01)
    p.add_argument("--residual-budget", type=float, default=.5)
    p.add_argument("--max-linear-target-error", type=float, default=.5)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    dtype = "float32" if args.dtype == "float32" else "bfloat16"
    tokenizer, model = mbc.load_model(args.model, args.device, dtype)
    report = {"method": "BRIDGE", "model": args.model, "device": args.device,
              "datasets": {}}
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        report["datasets"][name] = run_dataset(model, tokenizer, args, name)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
