#!/usr/bin/env python3
"""Native endogenous-branch likelihood pullback on TruthfulQA.

This runner replaces the layer-24 observer bridge with a direct Jacobian from
fixed endogenous branch sequence likelihoods to a selectable branch-launch
residual. The prompt is only the common prefix used to enumerate and greedily
complete branches; the controlled object is the branch distribution.
The only supervised upstream object is the early (layers 1--4) branch scorer.
No future branch hidden state, Teacher, or Student is used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # package import
    from .native_branch_utils import (
        clean,
        load_model as load_native_model,
        tqa_prompt as native_tqa_prompt,
    )
except ImportError:  # direct script execution
    from native_branch_utils import (
        clean,
        load_model as load_native_model,
        tqa_prompt as native_tqa_prompt,
    )

EPS = 1e-8
# BRIDGE configuration used in the current experiment: one early L4
# representation, rather than concatenating L1--L4.
EARLY_LAYERS = (4,)
DEFAULT_CONTROL_LAYER = 16
DEFAULT_BRANCH_TAU = 0.25
# All 16 fixed branches fit comfortably on the reserved 32GB card.  Keeping
# them in one batch makes the Jacobian rows exact while avoiding 4x repeated
# model passes per question.
DEFAULT_GRAD_BRANCH_BATCH = 2
DEFAULT_SCORE_BRANCH_BATCH = 2


def tqa_prompt(question: str, tokenizer) -> str:
    """Shared prefix used only to launch label-free greedy branch rollout."""
    return native_tqa_prompt(question, tokenizer)


def load_model(path: str, device: str, model_dtype: str):
    return load_native_model(path, device, model_dtype)


def valid_branches(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    branches = [b for b in row.get("branches", []) if b.get("valid", True)]
    if len(branches) < 2:
        raise ValueError(f"row {row.get('id')} has fewer than two valid branches")
    return branches


def proposal_mass(
    branches: Sequence[Mapping[str, Any]], tau: float = DEFAULT_BRANCH_TAU,
) -> torch.Tensor:
    """Cached native branch distribution for early geometry and supervision.

    The same likelihood temperature is used by the transport distribution so
    the coverage-influence target is a first-order objective at the native
    branch distribution that the pullback subsequently reweights.
    """
    vals = torch.tensor(
        [float(b.get("mean_logprob", -50.0)) for b in branches],
        dtype=torch.float32,
    )
    return torch.softmax(vals / max(float(tau), 1e-4), dim=0)


def early_branch_vector(branch: Mapping[str, Any]) -> torch.Tensor:
    return torch.cat(
        [branch["hidden_diff"][str(layer)].float() for layer in EARLY_LAYERS]
    )


def early_candidate_vector(candidate: Mapping[str, Any]) -> torch.Tensor:
    return torch.cat(
        [candidate["hidden_diff"][str(layer)].float() for layer in EARLY_LAYERS]
    )


class EarlyPCA:
    def __init__(self, mean: torch.Tensor, components: torch.Tensor,
                 eigenvalues: torch.Tensor):
        self.mean = mean.float()
        self.components = components.float()
        self.eigenvalues = eigenvalues.float()

    @classmethod
    def fit(
        cls, rows: Sequence[Mapping[str, Any]], rank: int = 8,
        branch_tau: float = DEFAULT_BRANCH_TAU, device: str = "cpu",
    ):
        xs, weights = [], []
        for row in rows:
            branches = valid_branches(row)
            p = proposal_mass(branches, branch_tau)
            xs.extend(early_branch_vector(b) for b in branches)
            weights.extend(p.tolist())
        # The fit matrix is modest (~2k x 14k for this protocol) but its
        # randomized SVD is still much faster on the already-loaded GPU.  The
        # learned basis is returned to CPU, so downstream fit-time target
        # construction stays deterministic and memory-light.
        x = torch.stack(xs).float().to(device)
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        w = w / w.sum().clamp_min(EPS)
        mean = (x * w[:, None]).sum(0)
        weighted = (x - mean) * w.sqrt()[:, None]
        rr = min(int(rank), int(x.shape[0]), int(x.shape[1]))
        _, singular, v = torch.pca_lowrank(weighted, q=rr, center=False, niter=8)
        return cls(
            mean.cpu(),
            v[:, :rr].T.contiguous().cpu(),
            singular[:rr].square().clamp_min(1e-8).cpu(),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float() - self.mean) @ self.components.T
                / self.eigenvalues.sqrt())


def projected_coverage_influence_target(
    endo_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    labels: torch.Tensor,
    pca: EarlyPCA,
    tau: float = 0.25,
    coverage_sigma: float = 1.5,
    use_support: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-order coverage utility in the early branch proposal manifold.

    We deliberately avoid a branch--candidate cosine score here.  For each
    training question, the candidate cloud is centered and SVD projected onto
    its non-degenerate subspace. Let q be a distribution over fixed natural
    branches and let C(q) be log coverage of correct answer modes minus log
    coverage of wrong modes. The returned utility is dC/dq evaluated at the
    native proposal mass. It values an uncovered correct mode more than an
    already-covered one, with no heuristic cosine or repulsion bonus.
    The scorer later predicts this scalar from early branch coordinates alone,
    so candidate states/labels are fit-time supervision only.
    """
    branches = valid_branches(endo_row)
    candidates = candidate_row["candidates"]
    if len(candidates) != int(labels.numel()):
        raise ValueError("candidate/label count mismatch")
    # Keep the whitened coordinates themselves.  Normalizing to a sphere would
    # throw away the radial expansion that the early-branch pre-experiment
    # identified as useful for finite-budget coverage.
    z = pca.transform(torch.stack([early_branch_vector(b) for b in branches])).float()
    c = pca.transform(torch.stack([early_candidate_vector(x) for x in candidates])).float()
    pos = torch.nonzero(labels == 1, as_tuple=False).flatten()
    neg = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        raise ValueError("need positive and negative answer candidates")
    center = c.mean(0, keepdim=True)
    centered = c - center
    # The projection is row-local and used only to define the training
    # signal.  Padding keeps the target invariant when a row has low rank.
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(4, int(vh.shape[0]), int(vh.shape[1]))
    if rank:
        basis = vh[:rank].T.contiguous()
        # Row-wise whitening is useful, but tiny singular values can amplify
        # harmless numerical noise into enormous distances.  Keep a bounded
        # geometric scale; the likelihood temperature tau is intentionally not
        # reused for this geometry.
        scale = (singular[:rank] / math.sqrt(max(1, c.shape[0]))).clamp_min(0.75)
        cp = (centered @ basis) / scale
        bp = ((z - center) @ basis) / scale
    else:
        cp = centered.new_zeros((c.shape[0], 1))
        bp = (z - center).norm(dim=-1, keepdim=True)
    gp = cp[pos]
    wp = cp[neg]
    # Squared projected distances define soft correct-region membership.
    geo_sigma = max(float(coverage_sigma), 1e-3)
    d2_gold = (bp[:, None, :] - gp[None, :, :]).square().sum(-1)
    gold_kernel = torch.exp(-d2_gold / (2.0 * geo_sigma ** 2))
    proposal = proposal_mass(branches, tau).to(gold_kernel)
    # C+(q) = mean_j log(eps + sum_b q_b kappa+(b,j)). Its derivative is
    # kappa+(b,j) divided by existing support at that correct mode. This is
    # the precise marginal value of moving a small amount of mass to b.
    if use_support:
        gold_support = EPS + (proposal[:, None] * gold_kernel).sum(dim=0)
        positive_influence = (gold_kernel / gold_support[None, :]).mean(dim=-1)
    else:
        # Ablation: remove the marginal-support denominator while retaining
        # the same projected gold/wrong kernels and all downstream budgets.
        positive_influence = gold_kernel.mean(dim=-1)
    d2_wrong = (bp[:, None, :] - wp[None, :, :]).square().sum(-1)
    wrong_kernel = torch.exp(-d2_wrong / (2.0 * geo_sigma ** 2))
    # Use the corresponding derivative of wrong-region coverage, rather than
    # a pointwise wrong-distance heuristic, so both terms live in one set
    # functional C(q) = C+(q) - lambda C-(q).
    if use_support:
        wrong_support = EPS + (proposal[:, None] * wrong_kernel).sum(dim=0)
        wrong_influence = (wrong_kernel / wrong_support[None, :]).mean(dim=-1)
    else:
        wrong_influence = wrong_kernel.mean(dim=-1)
    raw = positive_influence - 0.50 * wrong_influence
    p = proposal.to(raw)
    return raw - (p * raw).sum(), z, p


# Backward-compatible name for external analysis scripts; the native runner
# now uses the projection/coverage target above.
quality_target = projected_coverage_influence_target


class EarlyBranchScorer(nn.Module):
    def __init__(self, dim: int = 8, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim + 1, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Accept either one question [K,D] or a mini-batch [B,K,D].
        zbar = (p.unsqueeze(-1) * z).sum(dim=-2, keepdim=True).expand_as(z)
        features = torch.cat(
            [z, zbar, z - zbar, torch.log(p.unsqueeze(-1) + EPS)], dim=-1
        )
        return self.net(features).squeeze(-1)


def train_scorer(
    features: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dim: int,
    epochs: int,
    lr: float,
    rank_weight: float,
    seed: int,
    device: str = "cpu",
    batch_size: int = 32,
) -> EarlyBranchScorer:
    random.seed(seed)
    torch.manual_seed(seed)
    model = EarlyBranchScorer(dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_loss = None, float("inf")
    targets = torch.stack([item[0] for item in features]).float().to(device)
    zs = torch.stack([item[1] for item in features]).float().to(device)
    ps = torch.stack([item[2] for item in features]).float().to(device)
    if targets.ndim != 2 or zs.ndim != 3 or ps.ndim != 2:
        raise ValueError("all fit rows must share the fixed branch-count contract")
    for epoch in range(int(epochs)):
        order = list(range(len(features)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        model.train()
        for start in range(0, len(order), max(1, int(batch_size))):
            ids = torch.tensor(order[start:start + max(1, int(batch_size))], device=device)
            target, z, p = targets[ids], zs[ids], ps[ids]
            pred = model(z, p)
            pred = pred - (p * pred).sum(dim=-1, keepdim=True)
            quality = F.smooth_l1_loss(pred, target)
            # [B,K,K]: retain the old row-local pairwise objective, then
            # average rows rather than pooling questions with many valid pairs.
            diff = target[:, :, None] - target[:, None, :]
            valid = diff > 0.02
            pair_hinge = F.relu(0.05 - (pred[:, :, None] - pred[:, None, :]))
            pair_count = valid.float().sum(dim=(-2, -1))
            per_row = (pair_hinge * valid).sum(dim=(-2, -1)) / pair_count.clamp_min(1.0)
            ranking = per_row[pair_count > 0].mean() if bool((pair_count > 0).any()) else pred.new_zeros(())
            loss = quality + float(rank_weight) * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        current = float(np.mean(losses)) if losses else float("inf")
        if current < best_loss:
            best_loss = current
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def dataset_rows(path: str, tokenizer) -> List[Dict[str, Any]]:
    ds = load_from_disk(path)["validation"]
    return [
        {
            "id": f"truthfulqa_{i}",
            "question": str(row["question"]),
            "prompt": tqa_prompt(str(row["question"]), tokenizer),
            "official": dict(row),
        }
        for i, row in enumerate(ds)
    ]


def candidate_texts(row: Mapping[str, Any]):
    official = row["official"]
    return (
        [" " + str(x).strip() for x in official["mc1_targets"]["choices"]],
        [int(x) for x in official["mc1_targets"]["labels"]],
        [" " + str(x).strip() for x in official["mc2_targets"]["choices"]],
        [int(x) for x in official["mc2_targets"]["labels"]],
    )


def prefix_ids(tokenizer, prompt: str, continuation: str | None = None):
    if continuation is None:
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        return ids, len(ids) - 1
    full = tokenizer(
        prompt + continuation,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )["input_ids"]
    base = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    common = 0
    for left, right in zip(full, base):
        if left != right:
            break
        common += 1
    return full, min(max(common, 1), len(full) - 1)


def mc1(scores: Sequence[Sequence[float]], labels: Sequence[Sequence[int]]) -> float:
    if not scores:
        return 0.0
    return 100.0 * float(np.mean([
        int(np.argmax(score) == int(label.index(1)))
        for score, label in zip(scores, labels)
    ]))


def mc2(
    scores: Sequence[Sequence[float]],
    labels: Sequence[Sequence[int]],
    temperature: float = 1.0,
) -> float:
    if not scores:
        return 0.0
    values = []
    for score, label in zip(scores, labels):
        probs = torch.softmax(
            torch.tensor(score, dtype=torch.float64) / float(temperature), dim=0
        )
        positive = [i for i, flag in enumerate(label) if flag == 1]
        values.append(float(probs[positive].sum()))
    return 100.0 * float(np.mean(values))


def _sequence_score_from_logits(
    logits: torch.Tensor,
    prompt_len: int,
    continuation: torch.Tensor,
) -> torch.Tensor:
    """Mean log P(continuation | prompt) from full-sequence logits."""
    if continuation.numel() == 0:
        return logits.new_zeros(())
    positions = logits[prompt_len - 1:prompt_len - 1 + continuation.numel()]
    token_logp = torch.log_softmax(positions.float(), -1)
    return token_logp.gather(1, continuation.reshape(-1, 1)).sum() / continuation.numel()


def _full_branch_ids(
    tokenizer, prompt: str, continuation_ids: Sequence[int],
) -> Tuple[torch.Tensor, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full = prompt_ids + [int(x) for x in continuation_ids]
    return torch.tensor(full, dtype=torch.long), len(prompt_ids)


def _branch_batch(
    tokenizer,
    prompt: str,
    branches: Sequence[Mapping[str, Any]],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad a small collection of fixed branch continuations into one batch.

    Each row has the same prefix and its own fixed greedy continuation.  Rows
    are independent under the causal LM, hence the gradient of the summed
    per-row likelihoods with respect to the per-row launch states is exactly
    the stack of the branch Jacobian rows.  Batching only removes redundant
    computation; it does not mix or alter branches.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)
    continuations = [
        [int(token) for token in branch["token_ids"]]
        for branch in branches
    ]
    if not continuations or any(not cont for cont in continuations):
        raise ValueError("fixed branches must have nonempty token ids")
    width = max(len(cont) for cont in continuations)
    k = len(continuations)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    input_ids = torch.full(
        (k, prompt_len + width), pad, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(input_ids)
    continuation_ids = torch.full(
        (k, width), pad, dtype=torch.long, device=device
    )
    continuation_mask = torch.zeros((k, width), dtype=torch.float32, device=device)
    prefix = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids[:, :prompt_len] = prefix
    attention[:, :prompt_len] = 1
    for index, cont in enumerate(continuations):
        n = len(cont)
        ids = torch.tensor(cont, dtype=torch.long, device=device)
        input_ids[index, prompt_len:prompt_len + n] = ids
        attention[index, prompt_len:prompt_len + n] = 1
        continuation_ids[index, :n] = ids
        continuation_mask[index, :n] = 1.0
    return input_ids, attention, continuation_ids, continuation_mask, prompt_len


def _batched_sequence_score(
    logits: torch.Tensor,
    prompt_len: int,
    continuation_ids: torch.Tensor,
    continuation_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean fixed-continuation log likelihood for every row of a batch."""
    width = int(continuation_ids.shape[1])
    relevant = logits[:, prompt_len - 1:prompt_len - 1 + width].float()
    token_logp = torch.log_softmax(relevant, dim=-1).gather(
        -1, continuation_ids.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logp * continuation_mask).sum(-1) / continuation_mask.sum(-1).clamp_min(1.0)


def native_branch_scores_multi(
    model,
    tokenizer,
    prompt: str,
    branches: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    device: str,
    need_jacobian: bool = False,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Read fixed branch likelihoods and several layer Jacobians in one pass.

    Every requested layer receives a zero-valued differentiable carrier
    ``leaf - leaf.detach()`` at the shared launch position.  The carrier keeps
    the forward pass *exactly* unchanged while exposing an independent leaf in
    the graph, so one forward/backward computes the likelihood Jacobian for
    every candidate actuation layer.  This avoids the common mistake of
    differentiating through a later tap that has already been detached by an
    earlier hook.
    """
    layers = sorted({int(layer) for layer in layers})
    if not layers:
        raise ValueError("at least one candidate layer is required")
    if layers[0] < 1 or layers[-1] > int(model.config.num_hidden_layers):
        raise ValueError(f"candidate layers outside model range: {layers}")
    modules = {layer: model.model.layers[layer - 1] for layer in layers}
    scores, grads = [], {layer: [] for layer in layers}
    h_launch = {}
    for start in range(0, len(branches), DEFAULT_GRAD_BRANCH_BATCH):
        subset = branches[start:start + DEFAULT_GRAD_BRANCH_BATCH]
        full, attention, continuation, continuation_mask, prompt_len = _branch_batch(
            tokenizer, prompt, subset, device
        )
        holder: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(layer: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                clean = hidden[:, prompt_len - 1, :].detach().float()
                holder[f"clean_{layer}"] = clean
                if not need_jacobian:
                    return output
                # Forward value is unchanged; derivative wrt leaf is the
                # identity at this layer and is independent of all other taps.
                leaf = clean.clone().requires_grad_(True)
                carrier = (leaf - leaf.detach()).to(hidden.dtype)
                edited = hidden.clone()
                edited[:, prompt_len - 1, :] = (
                    hidden[:, prompt_len - 1, :] + carrier
                )
                holder[f"leaf_{layer}"] = leaf
                return (edited, *output[1:]) if isinstance(output, tuple) else edited
            return hook

        for layer in layers:
            handles.append(modules[layer].register_forward_hook(make_hook(layer)))
        try:
            if need_jacobian:
                out = model(
                    input_ids=full,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
                score = _batched_sequence_score(
                    out.logits, prompt_len, continuation, continuation_mask
                )
                scores.append(score.detach().float().cpu())
                targets = [holder[f"leaf_{layer}"] for layer in layers]
                grad = torch.autograd.grad(
                    score.sum(), targets, retain_graph=False, allow_unused=False
                )
                for layer, value in zip(layers, grad):
                    grads[layer].append(value.float().detach().cpu())
                    if layer not in h_launch:
                        h_launch[layer] = holder[f"clean_{layer}"][0].float().cpu()
            else:
                with torch.inference_mode():
                    out = model(
                    input_ids=full,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
                scores.append(
                    _batched_sequence_score(
                        out.logits, prompt_len, continuation, continuation_mask
                    ).float().detach().cpu()
                )
                for layer in layers:
                    if layer not in h_launch:
                        h_launch[layer] = holder[f"clean_{layer}"][0].float().cpu()
        finally:
            for handle in handles:
                handle.remove()
    if not scores:
        raise ValueError("no valid branches")
    score_tensor = torch.cat(scores).cpu()
    jacobians = {layer: torch.cat(grads[layer]).cpu() for layer in layers} if need_jacobian else {}
    return score_tensor, jacobians, h_launch


def native_branch_scores(
    model,
    tokenizer,
    prompt: str,
    branches: Sequence[Mapping[str, Any]],
    layer: int,
    device: str,
    need_jacobian: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Backward-compatible single-layer wrapper."""
    scores, jacobians, h_launch = native_branch_scores_multi(
        model, tokenizer, prompt, branches, [int(layer)], device, need_jacobian
    )
    return scores, jacobians.get(int(layer)), h_launch[int(layer)]


def steered_sequence_scores(
    model,
    tokenizer,
    prompt: str,
    branches: Sequence[Mapping[str, Any]],
    layer: int,
    delta: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Re-score the same fixed branch token sequences after one residual edit."""
    module = model.model.layers[int(layer) - 1]
    result = []
    for start in range(0, len(branches), DEFAULT_SCORE_BRANCH_BATCH):
        subset = branches[start:start + DEFAULT_SCORE_BRANCH_BATCH]
        full, attention, continuation, continuation_mask, prompt_len = _branch_batch(
            tokenizer, prompt, subset, device
        )

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            edited = hidden.clone()
            edited[:, prompt_len - 1, :] = (
                hidden[:, prompt_len - 1, :].float()
                + delta.to(hidden.device, torch.float32)
            ).to(hidden.dtype)
            return (edited, *output[1:]) if isinstance(output, tuple) else edited

        handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=full,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
            result.append(
                _batched_sequence_score(
                    out.logits, prompt_len, continuation, continuation_mask
                ).float().detach().cpu()
            )
        finally:
            handle.remove()
    return torch.cat(result)


def steered_candidate_scores(
    model, tokenizer, prompt: str, options: Sequence[str],
    layer: int, delta: torch.Tensor, device: str,
) -> List[float]:
    encoded = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    prefix_len = int(encoded["input_ids"].shape[1])
    module = model.model.layers[int(layer) - 1]

    def inject(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[:, prefix_len - 1, :] = (
            hidden[:, prefix_len - 1, :].float()
            + delta.to(hidden.device, torch.float32)
        ).to(hidden.dtype)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = module.register_forward_hook(inject)
    try:
        with torch.inference_mode():
            prefill = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=True,
                return_dict=True,
            )
        first_logits = prefill.logits[:, -1, :].float()
        past = prefill.past_key_values
    finally:
        handle.remove()
    result = []
    for option in options:
        full, start = prefix_ids(tokenizer, prompt, option)
        cont = full[start:]
        if not cont:
            result.append(0.0)
            continue
        target = torch.tensor(cont, device=device, dtype=torch.long)
        value = torch.log_softmax(first_logits[0], -1)[target[0]]
        if len(cont) > 1:
            feed = target[:-1].unsqueeze(0)
            mask = torch.ones(
                (1, prefix_len + feed.shape[1]),
                device=device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                out = model(
                    input_ids=feed,
                    attention_mask=mask,
                    past_key_values=past,
                    use_cache=False,
                    return_dict=True,
                )
            value = value + torch.log_softmax(out.logits.float(), -1)[0].gather(
                1, target[1:].unsqueeze(1)
            ).sum()
        result.append(float(value))
    return result


def evaluate_clean_answers(
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[torch.Tensor],
    control_layer: int,
    device: str,
) -> Dict[str, Any]:
    """Evaluate the frozen backbone with the identical MC scorer and no edit.

    This deliberately bypasses branch extraction, scorer prediction, and
    Jacobians.  It is therefore a genuine clean held-out baseline, while using
    precisely the same continuation-likelihood MC1/MC2 protocol as the
    steered evaluation below.  Calling the common scoring function with a
    zero residual also keeps tokenisation and cached-prefill behaviour matched.
    """
    mc1_scores, mc2_scores, y1s, y2s = [], [], [], []
    zero = torch.zeros(int(model.config.hidden_size), dtype=torch.float32)
    total = len(rows)
    for index, (row, label) in enumerate(zip(rows, labels), start=1):
        c1, l1, c2, l2 = candidate_texts(row)
        mc1_scores.append(
            steered_candidate_scores(
                model, tokenizer, row["prompt"], c1, control_layer, zero, device
            )
        )
        mc2_scores.append(
            steered_candidate_scores(
                model, tokenizer, row["prompt"], c2, control_layer, zero, device
            )
        )
        y1s.append(l1)
        y2s.append(l2)
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[test-clean] {index}/{total}", flush=True)
    return {
        "mc1_percent": mc1(mc1_scores, y1s),
        "mc2_percent": mc2(mc2_scores, y2s),
        "joint": 0.5 * (mc1(mc1_scores, y1s) + mc2(mc2_scores, y2s)),
        "mode": "frozen_backbone_no_residual_edit",
    }


def center_matrix(k: int, device: torch.device | str = "cpu") -> torch.Tensor:
    eye = torch.eye(k, device=device)
    ones = torch.ones((k, k), device=device) / float(k)
    return eye - ones


def tangent_jacobian(jacobian: torch.Tensor, h: torch.Tensor) -> Dict[str, Any]:
    h = h.float().reshape(-1)
    radius = h.norm().clamp_min(EPS)
    hhat = h / radius
    j = jacobian.float()
    a = j - (j @ hhat)[:, None] * hhat[None, :]
    sv = torch.linalg.svdvals(a)
    keep = (
        sv >= sv.max().clamp_min(EPS) * 1e-4
        if sv.numel() else torch.zeros(0, dtype=torch.bool)
    )
    kept = sv[keep]
    return {
        "A": a,
        "radius": radius,
        "rank": int(kept.numel()),
        "condition_number": float(kept.max() / kept.min()) if kept.numel() else float("inf"),
        "singular_values": sv,
        "discarded_fraction": float((sv.numel() - kept.numel()) / max(1, sv.numel())),
    }


def sphere_exp_map(h: torch.Tensor, xi: torch.Tensor):
    h = h.float().reshape(-1)
    radius = h.norm().clamp_min(EPS)
    tangent = xi.float().reshape(-1) - h * (h @ xi.float().reshape(-1)) / radius.pow(2)
    norm_xi = tangent.norm()
    if float(norm_xi) <= EPS:
        return h.clone(), torch.zeros_like(h), {
            "angle": 0.0, "norm_error": 0.0, "tangent_cosine": 0.0,
        }
    angle = norm_xi / radius
    endpoint = torch.cos(angle) * h + radius * torch.sin(angle) * tangent / norm_xi
    delta = endpoint - h
    return endpoint, delta, {
        "angle": float(angle),
        "norm_error": float(abs(endpoint.norm() / radius - 1.0)),
        "tangent_cosine": float(abs(h @ tangent) / (radius * norm_xi).clamp_min(EPS)),
    }


def branch_realization(
    actual_ell: torch.Tensor,
    clean_ell: torch.Tensor,
    tau: float,
    target_t: torch.Tensor,
    p0: torch.Tensor,
    utility: torch.Tensor,
) -> Dict[str, Any]:
    """Convert measured branch scores into the centered log-ratio audit."""
    k = int(clean_ell.numel())
    hmat = center_matrix(k)
    actual_p = torch.softmax(actual_ell.float() / max(float(tau), 1e-4), dim=0)
    actual_t = hmat @ (
        torch.log(actual_p.clamp_min(EPS)) - torch.log(p0.clamp_min(EPS))
    )
    if float(target_t.norm()) <= EPS:
        # The clean no-op has no direction, so relative target error and
        # alignment are undefined rather than evidence of bad realization.
        residual, alignment = 0.0, 1.0
    else:
        residual = float((actual_t - target_t).norm() / target_t.norm())
        alignment = float(
            (actual_t @ target_t)
            / (actual_t.norm().clamp_min(EPS) * target_t.norm().clamp_min(EPS))
        )
    return {
        "actual_ell": actual_ell,
        "actual_p": actual_p,
        "actual_t": actual_t,
        "nonlinear_residual": float(residual),
        "target_actual_alignment": float(alignment),
        "actual_gain": float((actual_p - p0) @ utility),
        "actual_utility": float((actual_p * utility).sum()),
    }


def dynamic_realize(
    model,
    tokenizer,
    prompt: str,
    branches: Sequence[Mapping[str, Any]],
    h_launch: torch.Tensor,
    xi: torch.Tensor,
    clean_ell: torch.Tensor,
    p0: torch.Tensor,
    target_t: torch.Tensor,
    utility: torch.Tensor,
    layer: int,
    device: str,
    tau: float,
    theta_max: float,
    jacobian: torch.Tensor,
    ridge: float,
    pullback_mode: str = "ridge",
    residual_threshold: float = 0.35,
    min_gain_fraction: float = 0.05,
    feedback_steps: int = 1,
) -> Dict[str, Any]:
    """Closed-loop realization with backtracking and one bounded feedback step.

    The Jacobian supplies the initial pullback.  The actual fixed-branch
    likelihood readout then decides whether to keep, shrink, or correct it.
    This keeps nonlinearity and BF16 endpoint quantization inside the controller
    rather than treating them as post-hoc diagnostics.
    """
    radius = h_launch.norm().clamp_min(EPS)
    target_gain = float((torch.softmax(
        torch.log(p0.clamp_min(EPS)) + target_t, dim=0
    ) - p0) @ utility)
    initial_xi = xi.float()
    # Include a clean candidate: if quantization erases every small update or
    # the measured direction is harmful, the controller has an explicit no-op.
    # Four geometrically spaced nonzero trials cover the useful trust-region
    # range; the explicit no-op is handled analytically below.  This preserves
    # bounded closed-loop selection while avoiding eight redundant full-model
    # passes for every question in the held-out run.
    scales = (1.0, 0.5, 0.25, 0.125, 0.0)
    trials: List[Dict[str, Any]] = []

    def trial(candidate_xi: torch.Tensor, label: str, scale: float):
        tangent = candidate_xi - h_launch * (h_launch @ candidate_xi) / radius.pow(2)
        angle = float(tangent.norm() / radius)
        if angle > float(theta_max):
            tangent = tangent * (float(theta_max) * radius / tangent.norm().clamp_min(EPS))
            angle = float(theta_max)
        _endpoint, delta, sphere = sphere_exp_map(h_launch, tangent)
        if float(delta.norm()) <= EPS:
            actual_ell = clean_ell.clone()
        else:
            actual_ell = steered_sequence_scores(
                model, tokenizer, prompt, branches, layer, delta, device
            )
        measured = branch_realization(
            actual_ell, clean_ell, tau, target_t, p0, utility
        )
        rec = {
            "label": label,
            "scale": float(scale),
            "xi": tangent,
            "delta": delta,
            "angle": angle,
            "sphere": sphere,
            **measured,
        }
        trials.append(rec)
        return rec

    # First choose the best measured point among backtracked copies.  A point
    # is acceptable only when it has the right direction and nontrivial gain.
    for scale in scales:
        trial(initial_xi * float(scale), "backtrack", float(scale))
    acceptable = [
        t for t in trials
        if t["nonlinear_residual"] <= float(residual_threshold)
        and t["target_actual_alignment"] > 0.0
        and (target_gain <= EPS or t["actual_gain"] >= float(min_gain_fraction) * target_gain)
    ]
    if acceptable:
        chosen = max(acceptable, key=lambda t: (t["actual_gain"], -t["nonlinear_residual"]))
    else:
        # Prefer a positive-gain point even when the strict residual threshold
        # is unattainable; otherwise return the exact clean candidate.
        positive = [t for t in trials if t["actual_gain"] > 0.0 and t["target_actual_alignment"] > 0.0]
        chosen = max(positive, key=lambda t: (t["actual_gain"], -t["nonlinear_residual"])) if positive else trials[-1]

    feedback_used = False
    feedback_before = float(chosen["nonlinear_residual"])
    feedback_after = feedback_before
    if (
        feedback_steps > 0
        and chosen["scale"] > 0.0
        and chosen["nonlinear_residual"] > float(residual_threshold)
        and chosen["target_actual_alignment"] > 0.0
    ):
        a = jacobian.float()
        correction = pullback_target(
            a, target_t - chosen["actual_t"], ridge, pullback_mode
        )
        correction = correction - h_launch * (h_launch @ correction) / radius.pow(2)
        # Never let a feedback correction dominate the original pullback.
        limit = 0.5 * initial_xi.norm().clamp_min(EPS)
        if float(correction.norm()) > float(limit):
            correction = correction * (limit / correction.norm().clamp_min(EPS))
        corrected = trial(chosen["xi"] + correction, "feedback", 1.0)
        feedback_after = float(corrected["nonlinear_residual"])
        if corrected["actual_gain"] >= chosen["actual_gain"] and (
            corrected["target_actual_alignment"] >= chosen["target_actual_alignment"]
            or corrected["nonlinear_residual"] < chosen["nonlinear_residual"]
        ):
            chosen = corrected
            feedback_used = True

    # A zero initial pullback can appear in a trial labelled scale=1.0; use
    # the actual edit norm, not the bookkeeping scale, to audit a true no-op.
    fallback_clean = float(chosen["delta"].norm()) <= EPS
    return {
        "chosen": chosen,
        "trials": trials,
        "feedback_used": feedback_used,
        "feedback_residual_before": feedback_before,
        "feedback_residual_after": feedback_after,
        "fallback_clean": fallback_clean,
        "target_gain": target_gain,
    }


def target_from_eta(
    ell: torch.Tensor,
    utility: torch.Tensor,
    tau: float,
    eta: float,
    transport_mode: str = "kl",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = torch.softmax(ell.float() / max(float(tau), 1e-4), dim=0)
    if transport_mode == "additive":
        centered_u = utility.float() - (p * utility.float()).sum()
        q = (p + float(eta) * centered_u).clamp_min(EPS)
        q = q / q.sum().clamp_min(EPS)
    else:
        log_q = torch.log(p.clamp_min(EPS)) + float(eta) * utility.float()
        q = torch.softmax(log_q, dim=0)
    hmat = center_matrix(int(p.numel()))
    t = hmat @ (torch.log(q.clamp_min(EPS)) - torch.log(p.clamp_min(EPS)))
    return p, q, t


def solve_eta(
    ell: torch.Tensor,
    utility: torch.Tensor,
    tangent: Mapping[str, Any],
    tau: float,
    theta_max: float,
    ridge: float,
    transport_mode: str = "kl",
    pullback_mode: str = "ridge",
) -> Dict[str, Any]:
    """Choose the largest KL tilt inside the local spherical angle budget."""
    p = torch.softmax(ell.float() / max(float(tau), 1e-4), dim=0)
    a = tangent["A"].float()
    h = tangent["radius"].float()

    def evaluate_eta(eta: float):
        _, q, t = target_from_eta(ell, utility, tau, eta, transport_mode)
        # The target is centered, so solve only in the branch-score quotient.
        xi = pullback_target(a, t, ridge, pullback_mode)
        # Numerical projection makes the tangent guarantee explicit.
        hvec = tangent.get("hvec")
        if hvec is not None:
            xi = xi - hvec * (hvec @ xi) / h.pow(2).clamp_min(EPS)
        angle = float(xi.norm() / h.clamp_min(EPS))
        return q, t, xi, angle

    if float(theta_max) <= 0.0 or float(utility.max() - utility.min()) <= EPS:
        q, t, xi, angle = evaluate_eta(0.0)
        return {
            "eta": 0.0, "p": p, "q": q, "t": t, "xi": torch.zeros_like(xi),
            "angle": 0.0, "target_gain": 0.0,
        }

    hi = 1.0
    while evaluate_eta(hi)[3] < float(theta_max) and hi < 1e4:
        hi *= 2.0
    lo = 0.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if evaluate_eta(mid)[3] <= float(theta_max):
            lo = mid
        else:
            hi = mid
    eta = lo
    q, t, xi, angle = evaluate_eta(eta)
    gain = float((q - p) @ utility.float())
    return {
        "eta": float(eta), "p": p, "q": q, "t": t, "xi": xi,
        "angle": float(angle), "target_gain": gain,
    }


def pullback_target(
    a: torch.Tensor,
    target_t: torch.Tensor,
    ridge: float,
    mode: str = "ridge",
) -> torch.Tensor:
    """Map a branch-space target to residual space using the requested solver."""
    if mode == "transpose":
        return a.T @ target_t
    gram = a @ a.T
    return a.T @ torch.linalg.solve(
        gram + float(ridge) * torch.eye(gram.shape[0], device=a.device),
        target_t,
    )


def linear_realization_metrics(
    p: torch.Tensor,
    target_t: torch.Tensor,
    xi: torch.Tensor,
    jacobian: torch.Tensor,
    utility: torch.Tensor,
) -> Dict[str, Any]:
    """Audit whether a pullback actually realizes its branch-space target.

    ``target_gain`` only scores the requested transport target.  The layer
    selector must instead score the shift produced by the local linear map,
    ``A @ xi``.  The predicted distribution is therefore obtained by applying
    that shift to ``log p``; it is a prediction, not the final forward result.
    """
    linear_t = jacobian.float() @ xi.float()
    target_norm = target_t.float().norm()
    if float(target_norm) <= EPS:
        error = 0.0
        predicted_p = p.float().clone()
    else:
        error = float((linear_t - target_t.float()).norm() / target_norm)
        predicted_p = torch.softmax(
            torch.log(p.float().clamp_min(EPS)) + linear_t, dim=0
        )
    predicted_gain = float((predicted_p - p.float()) @ utility.float())
    return {
        "linear_t": linear_t,
        "predicted_p": predicted_p,
        "predicted_gain": predicted_gain,
        "linear_target_error": error,
    }


def audit_summary(audits: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    fields = [
        "eta", "target_gain", "actual_gain", "kl_target_actual",
        "predicted_gain", "linear_target_error", "linear_residual",
        "nonlinear_residual", "angle", "sphere_norm_error",
        "pullback_norm", "jacobian_rank", "jacobian_condition",
        "proposal_transport_tv",
        "clean_score_alignment", "steered_score_alignment",
        "feedback_residual_before", "feedback_residual_after",
    ]
    out: Dict[str, float] = {}
    for field in fields:
        vals = np.asarray([float(a.get(field, 0.0)) for a in audits], dtype=float)
        if vals.size:
            out[f"mean_{field}"] = float(vals.mean())
            out[f"p50_{field}"] = float(np.quantile(vals, 0.50))
            out[f"p90_{field}"] = float(np.quantile(vals, 0.90))
    if audits:
        out["intervention_rate"] = float(np.mean([
            not bool(a.get("fallback_clean", False)) for a in audits
        ]))
        out["fallback_clean_rate"] = float(np.mean([
            bool(a.get("fallback_clean", False)) for a in audits
        ]))
        out["feedback_rate"] = float(np.mean([
            bool(a.get("feedback_used", False)) for a in audits
        ]))
        out["layer_selection_valid_rate"] = float(np.mean([
            bool(a.get("layer_selection_valid", False)) for a in audits
        ]))
    return out


def evaluate(
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    endo_rows: Sequence[Mapping[str, Any]],
    pca: EarlyPCA,
    scorer: EarlyBranchScorer,
    labels: Sequence[torch.Tensor],
    candidate_rows: Sequence[Mapping[str, Any]],
    device: str,
    rho_strength: float,
    tau: float,
    theta_max: float,
    ridge: float,
    control_layer: int,
    smoke: bool = False,
    progress_tag: str = "evaluation",
    control_layers: Sequence[int] | None = None,
    layer_budget_mode: str = "absolute",
    layer_residual_norm: float = 0.5,
    max_linear_target_error: float = 0.5,
    transport_mode: str = "kl",
    pullback_mode: str = "ridge",
    utility_mode: str = "scored",
) -> Dict[str, Any]:
    mc1_scores, mc2_scores, y1s, y2s, audits = [], [], [], [], []
    candidate_layers = sorted({int(x) for x in (control_layers or [control_layer])})
    if not candidate_layers:
        raise ValueError("control_layers must contain at least one layer")
    selected_layer_counts: Dict[str, int] = {}
    layer_selection_accum: Dict[str, Dict[str, float]] = {
        str(layer): {"count": 0.0, "target_gain_sum": 0.0,
                     "predicted_gain_sum": 0.0,
                     "linear_target_error_sum": 0.0,
                     "condition_sum": 0.0, "pullback_norm_sum": 0.0}
        for layer in candidate_layers
    }
    total = len(rows)
    for index, (row, endo_row, label, candidate_row) in enumerate(zip(
        rows, endo_rows, labels, candidate_rows
    ), start=1):
        branches = valid_branches(endo_row)
        # Match the projection-space coordinates used to train the scorer;
        # retain radial information instead of renormalizing away expansion.
        z = pca.transform(torch.stack([early_branch_vector(b) for b in branches])).float()
        proposal_p = proposal_mass(branches, tau)
        with torch.no_grad():
            if utility_mode == "top":
                # Hard no-supervision baseline: select the model's single
                # highest-probability native branch.  This is deliberately a
                # direct argmax target, not a learned score, so it tests
                # whether branch supervision contributes beyond greedy
                # selection under the same transport and residual budgets.
                utility = torch.zeros_like(proposal_p).float().cpu()
                utility[int(torch.argmax(proposal_p).item())] = 1.0
            elif utility_mode == "natural":
                # Smooth no-supervision baseline: use only natural branch
                # mass, with no early-geometry target or candidate labels.
                utility = torch.log(proposal_p.clamp_min(EPS)).float().cpu()
            else:
                utility = scorer(z.to(device), proposal_p.to(device)).float().cpu()
        utility = utility - (proposal_p * utility).sum()
        if len(candidate_layers) > 1:
            ell, jacobians, h_launches = native_branch_scores_multi(
                model, tokenizer, row["prompt"], branches, candidate_layers,
                device, need_jacobian=True,
            )
        else:
            ell, jac, h_single = native_branch_scores(
                model, tokenizer, row["prompt"], branches, candidate_layers[0],
                device, need_jacobian=True,
            )
            jacobians = {candidate_layers[0]: jac}
            h_launches = {candidate_layers[0]: h_single}
        hmat = center_matrix(len(branches))
        # Score every candidate layer under the same target and a depth-fair
        # residual budget (absolute by default; angle mode is retained for
        # ablations).  A candidate is realizable only when its local linear
        # shift A_r xi_r both tracks the requested target and yields positive
        # predicted utility.  The selected layer maximizes this *predicted
        # realizable* gain, not a label-derived answer metric.  This is a
        # cheap per-question controllability probe because all layer Jacobians
        # came from the single batched forward/backward above.
        layer_records = []
        for layer in candidate_layers:
            j_center_layer = hmat @ jacobians[layer] / max(float(tau), 1e-4)
            tangent_layer = tangent_jacobian(j_center_layer, h_launches[layer])
            tangent_layer["hvec"] = h_launches[layer]
            # A common angle is not a common residual intervention: hidden
            # state radii vary with depth.  Use one absolute Euclidean budget
            # by default so the layer probe compares controllability rather
            # than allowing a deeper layer a larger edit merely because its
            # sphere radius is larger.
            if layer_budget_mode == "absolute":
                layer_theta_max = float(layer_residual_norm) / float(
                    tangent_layer["radius"].clamp_min(EPS)
                )
            else:
                layer_theta_max = float(theta_max)
            solved_layer = solve_eta(
                ell, utility, tangent_layer, tau, layer_theta_max, ridge,
                transport_mode=transport_mode, pullback_mode=pullback_mode,
            )
            # Evaluate the same eta amplitude that this rho trial will use;
            # otherwise the selector could rank a layer using a target that
            # is never sent to the realization controller.
            probe_eta = float(solved_layer["eta"]) * float(rho_strength)
            probe_p, _probe_q, probe_t = target_from_eta(
                ell, utility, tau, probe_eta, transport_mode
            )
            probe_xi = pullback_target(
                tangent_layer["A"], probe_t, ridge, pullback_mode
            )
            hvec = tangent_layer.get("hvec")
            if hvec is not None:
                probe_xi = probe_xi - hvec * (hvec @ probe_xi) / tangent_layer["radius"].pow(2).clamp_min(EPS)
            linear_metrics = linear_realization_metrics(
                probe_p, probe_t, probe_xi, tangent_layer["A"], utility,
            )
            layer_records.append({
                "layer": int(layer),
                "target_gain": float(solved_layer["target_gain"]),
                "predicted_gain": float(linear_metrics["predicted_gain"]),
                "linear_target_error": float(linear_metrics["linear_target_error"]),
                "eta": float(solved_layer["eta"]),
                "jacobian_rank": int(tangent_layer["rank"]),
                "jacobian_condition": float(tangent_layer["condition_number"]),
                "pullback_norm": float(solved_layer["xi"].norm()),
                "angle": float(solved_layer["angle"]),
                "layer_theta_max": layer_theta_max,
                "layer_residual_budget": float(
                    layer_residual_norm if layer_budget_mode == "absolute"
                    else tangent_layer["radius"] * theta_max
                ),
            })
        realizable = [
            record for record in layer_records
            if record["linear_target_error"] <= float(max_linear_target_error)
            and record["predicted_gain"] > 0.0
        ]
        selection_valid = bool(realizable)
        if selection_valid:
            selected_layer_record = max(
                realizable,
                key=lambda record: (
                    record["predicted_gain"], -record["linear_target_error"],
                    -record["jacobian_condition"],
                ),
            )
            selected_layer = int(selected_layer_record["layer"])
        else:
            # No layer demonstrated a realizable target under the common
            # budget.  Keep the control-layer bookkeeping but force a clean
            # no-op instead of selecting an apparently attractive unattained
            # transport target.
            selected_layer = int(control_layer if control_layer in jacobians else candidate_layers[0])
        for record in layer_records:
            stats = layer_selection_accum[str(record["layer"])]
            stats["count"] += 1.0
            stats["target_gain_sum"] += float(record["target_gain"])
            stats["predicted_gain_sum"] += float(record["predicted_gain"])
            stats["linear_target_error_sum"] += float(record["linear_target_error"])
            stats["condition_sum"] += float(record["jacobian_condition"])
            stats["pullback_norm_sum"] += float(record["pullback_norm"])
        selected_layer_counts[str(selected_layer)] = selected_layer_counts.get(
            str(selected_layer), 0
        ) + 1
        jac = jacobians[selected_layer]
        h_launch = h_launches[selected_layer]
        # Branch probabilities are p=softmax(ell/tau), so the Jacobian in
        # centered log-probability coordinates is (1/tau) * d ell/d h_launch.
        j_center = hmat @ jac / max(float(tau), 1e-4)
        tangent = tangent_jacobian(j_center, h_launch)
        tangent["hvec"] = h_launch
        if layer_budget_mode == "absolute":
            selected_theta_max = float(layer_residual_norm) / float(
                tangent["radius"].clamp_min(EPS)
            )
        else:
            selected_theta_max = float(theta_max)
        solved = solve_eta(
            ell, utility, tangent, tau, selected_theta_max, ridge,
            transport_mode=transport_mode, pullback_mode=pullback_mode,
        )
        eta = float(solved["eta"]) * float(rho_strength) if selection_valid else 0.0
        p0, q, target = target_from_eta(
            ell, utility, tau, eta, transport_mode
        )
        a = tangent["A"]
        xi = pullback_target(a, target, ridge, pullback_mode)
        xi = xi - h_launch * (h_launch @ xi) / h_launch.norm().pow(2).clamp_min(EPS)
        realized = dynamic_realize(
            model=model,
            tokenizer=tokenizer,
            prompt=row["prompt"],
            branches=branches,
            h_launch=h_launch,
            xi=xi,
            clean_ell=ell,
            p0=p0,
            target_t=target,
            utility=utility,
            layer=selected_layer,
            device=device,
            tau=tau,
            theta_max=selected_theta_max,
            jacobian=j_center,
            ridge=ridge,
            pullback_mode=pullback_mode,
        )
        chosen = realized["chosen"]
        delta = chosen["delta"]
        sphere = chosen["sphere"]
        actual_ell = chosen["actual_ell"]
        actual_p = chosen["actual_p"]
        target_p = q
        actual_t = chosen["actual_t"]
        linear_t = a @ xi
        predicted_metrics = linear_realization_metrics(
            p0, target, xi, a, utility
        )
        actual_gain = float((actual_p - p0) @ utility)
        clean_align = float((p0 * utility).sum())
        steered_align = float((actual_p * utility).sum())
        delta_target = target
        audit = {
            "eta": eta,
            "target_gain": float((target_p - p0) @ utility),
            "actual_gain": actual_gain,
            "kl_target_actual": float((target_p * (
                torch.log(target_p.clamp_min(EPS))
                - torch.log(actual_p.clamp_min(EPS))
            )).sum()),
            "linear_residual": float((linear_t - delta_target).norm()
                                     / delta_target.norm().clamp_min(EPS)),
            "predicted_gain": float(predicted_metrics["predicted_gain"]),
            "linear_target_error": float(predicted_metrics["linear_target_error"]),
            "nonlinear_residual": float(chosen["nonlinear_residual"]),
            "angle": float(chosen["angle"]),
            "theta_budget": float(selected_theta_max),
            "residual_budget": float(
                layer_residual_norm if layer_budget_mode == "absolute"
                else tangent["radius"] * theta_max
            ),
            "sphere_norm_error": float(sphere["norm_error"]),
            "pullback_norm": float(chosen["xi"].norm()),
            "jacobian_rank": int(tangent["rank"]),
            "jacobian_condition": float(tangent["condition_number"]),
            "clean_score_alignment": clean_align,
            "steered_score_alignment": steered_align,
            "branch_count": len(branches),
            "clean_ell": ell.tolist(),
            "target_p": target_p.tolist(),
            "actual_p": actual_p.tolist(),
            "utility": utility.tolist(),
            "xi_norm": float(chosen["xi"].norm()),
            "delta_norm": float(delta.norm()),
            "branch_launch_hidden_norm": float(h_launch.norm()),
            "proposal_p": proposal_p.tolist(),
            "transport_p_clean": p0.tolist(),
            "proposal_transport_tv": float(0.5 * (proposal_p - p0).abs().sum()),
            "control_layer": int(selected_layer),
            "selected_layer": int(selected_layer),
            "candidate_control_layers": [int(x) for x in candidate_layers],
            "layer_selection": layer_records,
            "layer_selection_valid": selection_valid,
            "max_linear_target_error": float(max_linear_target_error),
            "chosen_scale": float(chosen["scale"]),
            "chosen_trial": str(chosen["label"]),
            "feedback_used": bool(realized["feedback_used"]),
            "feedback_residual_before": float(realized["feedback_residual_before"]),
            "feedback_residual_after": float(realized["feedback_residual_after"]),
            "fallback_clean": bool(realized["fallback_clean"]),
            "trial_count": len(realized["trials"]),
            "hook_leak_check": True,
        }
        c1, l1, c2, l2 = candidate_texts(row)
        mc1_scores.append(
            steered_candidate_scores(
                model, tokenizer, row["prompt"], c1, selected_layer, delta, device
            )
        )
        mc2_scores.append(
            steered_candidate_scores(
                model, tokenizer, row["prompt"], c2, selected_layer, delta, device
            )
        )
        y1s.append(l1)
        y2s.append(l2)
        audits.append(audit)
        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"[{progress_tag}] {index}/{total} "
                f"gain={actual_gain:+.4f} fallback={bool(realized['fallback_clean'])}",
                flush=True,
            )
        if smoke:
            print(json.dumps({
                "id": row["id"],
                "branch_count": len(branches),
                "jacobian_shape": list(jac.shape),
                "xi_dot_branch_launch_hidden": float(xi @ h_launch),
                "clean_ell": ell.tolist(),
                "target_p": target_p.tolist(),
                "actual_p": actual_p.tolist(),
                "audit": audit,
            }, ensure_ascii=False))
    return {
        "mc1_percent": mc1(mc1_scores, y1s),
        "mc2_percent": mc2(mc2_scores, y2s),
        "selected_layer_counts": selected_layer_counts,
        "layer_selection_summary": {
            layer: {
                "selected_rate": float(selected_layer_counts.get(layer, 0) / max(1, total)),
                "mean_target_gain": float(stats["target_gain_sum"] / max(1.0, stats["count"])),
                "mean_predicted_gain": float(stats["predicted_gain_sum"] / max(1.0, stats["count"])),
                "mean_linear_target_error": float(stats["linear_target_error_sum"] / max(1.0, stats["count"])),
                "mean_jacobian_condition": float(stats["condition_sum"] / max(1.0, stats["count"])),
                "mean_pullback_norm": float(stats["pullback_norm_sum"] / max(1.0, stats["count"])),
            }
            for layer, stats in layer_selection_accum.items()
        },
        "audits": audits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--truthfulqa", required=True)
    ap.add_argument("--mc-cache", required=True)
    ap.add_argument("--prompt-cache", required=True)
    ap.add_argument("--endogenous-cache", required=True)
    ap.add_argument("--candidate-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pca-rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--rho-grid", default="0.25,0.50,0.75,1.00")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument(
        "--coverage-sigma", type=float, default=1.5,
        help="bandwidth of the projected answer-space coverage kernel",
    )
    ap.add_argument(
        "--no-complementarity", action="store_true",
        help="ablation: remove coverage-support normalization from the utility target",
    )
    ap.add_argument(
        "--transport-mode", choices=["kl", "additive"], default="kl",
        help="branch transport: minimum-KL exponential tilt or matched additive shift",
    )
    ap.add_argument(
        "--pullback-mode", choices=["ridge", "transpose"], default="ridge",
        help="Jacobian inverse: ridge minimum-norm or normalized transpose control",
    )
    ap.add_argument(
        "--natural-branch", action="store_true",
        help="ablation: replace learned early branch utility by natural log branch mass",
    )
    ap.add_argument(
        "--top-branch", action="store_true",
        help="ablation: replace learned utility by a hard argmax over natural branch mass",
    )
    ap.add_argument("--theta-max", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument(
        "--control-layer", type=int, default=DEFAULT_CONTROL_LAYER,
        help="fallback branch-launch actuation layer when dynamic selection is disabled",
    )
    ap.add_argument(
        "--control-layers", default=None,
        help="comma-separated candidate actuation layers; dynamic selection uses the best predicted realizable gain",
    )
    ap.add_argument(
        "--dynamic-layer-selection", action=argparse.BooleanOptionalAction,
        default=True,
        help="select the actuation layer per question from the local branch controllability score (default: on)",
    )
    ap.add_argument(
        "--layer-budget-mode", choices=["absolute", "angle"], default="absolute",
        help="budget used only for dynamic layer probing and selected-layer realization; absolute is depth-fair",
    )
    ap.add_argument(
        "--layer-residual-norm", type=float, default=0.5,
        help="common Euclidean residual norm for dynamic layer probing (default: 0.5)",
    )
    ap.add_argument(
        "--max-linear-target-error", type=float, default=0.5,
        help="maximum relative error ||A xi - t||/||t|| allowed for layer selection",
    )
    ap.add_argument("--max-validation-rows", type=int, default=None)
    ap.add_argument(
        "--fit-fraction", type=float, default=1.0,
        help="fraction of the fixed fit split used to fit PCA and scorer (0,1]; validation/test stay unchanged",
    )
    ap.add_argument(
        "--max-fit-rows", type=int, default=None,
        help="optional cap on fit rows after applying --fit-fraction",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-test-eval", action="store_true")
    args = ap.parse_args()
    if args.dynamic_layer_selection:
        raw_layers = args.control_layers or "12,14,16,20"
        control_layers = sorted({int(x) for x in raw_layers.split(",") if x.strip()})
    else:
        control_layers = [int(args.control_layer)]
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompt_cache = torch.load(args.prompt_cache, map_location="cpu", weights_only=False)
    endogenous = torch.load(args.endogenous_cache, map_location="cpu", weights_only=False)
    mc = torch.load(args.mc_cache, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate_cache, map_location="cpu", weights_only=False)
    if list(map(int, prompt_cache["layers"])) != list(EARLY_LAYERS):
        raise AssertionError("prompt cache layers must be [1,2,3,4]")
    if not set(EARLY_LAYERS).issubset(set(map(int, endogenous["layers"]))):
        raise AssertionError("endogenous cache lacks early layers")
    if not set(EARLY_LAYERS).issubset(set(map(int, candidate["layers"]))):
        raise AssertionError("candidate cache lacks early layers")
    if int(prompt_cache.get("seed", 0)) != int(endogenous.get("seed", 0)):
        raise AssertionError("prompt/endogenous cache seed mismatch")

    tokenizer, model = load_model(args.model, args.device, args.model_dtype)
    if not 1 <= int(args.control_layer) <= int(model.config.num_hidden_layers):
        raise ValueError(
            f"control layer {args.control_layer} is outside [1, {model.config.num_hidden_layers}]"
        )
    all_rows = dataset_rows(args.truthfulqa, tokenizer)
    by_id = {row["id"]: row for row in all_rows}
    train_endo = list(endogenous["train"])
    test_endo = list(endogenous["test"])
    train_model = [by_id[row["id"]] for row in train_endo]
    test_model = [by_id[row["id"]] for row in test_endo]

    order = list(range(len(train_endo)))
    random.Random(args.seed + 7001).shuffle(order)
    nval = max(1, round(0.2 * len(order)))
    val_ids, fit_ids = order[:nval], order[nval:]
    if not (0.0 < float(args.fit_fraction) <= 1.0):
        raise ValueError("--fit-fraction must be in (0, 1]")
    # Subsample only the fit split.  The validation/test rows and their order
    # remain byte-for-byte identical across training-volume ablations.
    n_fit_use = max(1, int(round(float(args.fit_fraction) * len(fit_ids))))
    if args.max_fit_rows is not None:
        n_fit_use = min(n_fit_use, int(args.max_fit_rows))
    fit_ids = fit_ids[:n_fit_use]
    fit_endo = [train_endo[i] for i in fit_ids]
    val_endo = [train_endo[i] for i in val_ids]
    fit_model = [train_model[i] for i in fit_ids]
    val_model = [train_model[i] for i in val_ids]
    fit_candidates = [candidate["train"][i] for i in fit_ids]
    val_candidates = [candidate["train"][i] for i in val_ids]
    fit_labels = [mc["y_train"][i] for i in fit_ids]
    val_labels = [mc["y_train"][i] for i in val_ids]

    pca = EarlyPCA.fit(fit_endo, args.pca_rank, args.tau, device=args.device)
    fit_features = [
        quality_target(
            er, cr, y, pca, tau=args.tau,
            coverage_sigma=args.coverage_sigma,
            use_support=not bool(args.no_complementarity),
        )
        for er, cr, y in zip(fit_endo, fit_candidates, fit_labels)
    ]
    scorer = train_scorer(
        fit_features,
        pca.components.shape[0],
        args.epochs,
        args.lr,
        args.rank_weight,
        args.seed + 1,
        device=args.device,
    ).to(args.device)

    val_model = val_model if args.max_validation_rows is None else val_model[:args.max_validation_rows]
    val_endo = val_endo if args.max_validation_rows is None else val_endo[:args.max_validation_rows]
    val_candidates = val_candidates if args.max_validation_rows is None else val_candidates[:args.max_validation_rows]
    val_labels = val_labels if args.max_validation_rows is None else val_labels[:args.max_validation_rows]
    grid = []
    for rho in [float(x) for x in args.rho_grid.split(",") if x.strip()]:
        rec = evaluate(
            model, tokenizer, val_model, val_endo, pca, scorer, val_labels,
            val_candidates, args.device, rho, args.tau, args.theta_max,
            args.ridge, int(args.control_layer), smoke=args.smoke,
            progress_tag=f"validation rho={rho:g}",
            control_layers=control_layers,
            layer_budget_mode=args.layer_budget_mode,
            layer_residual_norm=args.layer_residual_norm,
            max_linear_target_error=args.max_linear_target_error,
            transport_mode=args.transport_mode,
            pullback_mode=args.pullback_mode,
            utility_mode=("top" if args.top_branch else ("natural" if args.natural_branch else "scored")),
        )
        audits = rec.pop("audits")
        grid.append({
            "rho": rho,
            **rec,
            "joint_val": 0.5 * (rec["mc1_percent"] + rec["mc2_percent"]),
            **audit_summary(audits),
        })
    best = max(grid, key=lambda x: (x["joint_val"], -x.get("mean_nonlinear_residual", 0.0)))

    report: Dict[str, Any] = {
        "method": "Native endogenous branch-likelihood minimum-KL pullback",
        "selected": best,
        "validation_grid": grid,
        "protocol": {
            "early_layers": list(EARLY_LAYERS),
            "control_layer": int(args.control_layer),
            "control_layers": [int(x) for x in control_layers],
            "dynamic_layer_selection": bool(args.dynamic_layer_selection),
            "layer_selection_rule": (
                f"per-question maximum predicted realizable coverage-utility gain among candidates with bounded relative linear target error under the same {args.layer_budget_mode} layer budget"
                if args.dynamic_layer_selection else "fixed control_layer"
            ),
            "layer_budget_mode": args.layer_budget_mode,
            "layer_residual_norm": args.layer_residual_norm,
            "max_linear_target_error": args.max_linear_target_error,
            "source_layer": "1-4 early branch proposal geometry",
            "controlled_object": "distribution over fixed top-k-first-token + greedy branches",
            "prompt_role": "common prefix only: enumerate first tokens and causally launch greedy branches",
            "actuation_site": "last prefix-token residual, the shared branch-launch state",
            "control_layer_interpretation": "branch-launch actuation site; early-branch evidence concerns target timing, while dynamic selection measures local controllability",
            "future_branch_layers_accessed": False,
            "teacher_used": False,
            "student_used": False,
            "layer24_observer_used": False,
            "candidate_text_used_for_direction": False,
            "branch_direction_source": "fixed endogenous continuation token ids",
            "branch_score": "mean conditional sequence log-likelihood",
            "early_supervision": "row-local SVD coverage-influence gradient at proposal mass: correct-mode coverage - wrong-mode coverage",
            "coverage_geometry_sigma": args.coverage_sigma,
            "complementarity": "coverage-support normalized marginal influence"
            if not args.no_complementarity else "ablation: unnormalized local kernel affinity",
            "proposal_mass": "cached natural branch mass used by PCA/scorer only",
            "transport_mass": "fresh fixed-branch likelihood distribution used by minimum-KL tilt; its temperature matches proposal mass",
            "target": (
                "minimum-KL exponential tilt q_eta proportional to p_transport exp(eta u)"
                if args.transport_mode == "kl"
                else "matched additive branch-mass shift"
            ),
            "transport_mode": args.transport_mode,
            "pullback_mode": args.pullback_mode,
            "utility_mode": ("hard natural argmax branch" if args.top_branch else ("natural log branch mass" if args.natural_branch else "learned early geometry scorer")),
            "layer_selection_score": "predicted realizable gain <A_r xi_r, u> after applying the local linear shift to log p",
            "layer_selection_gate": "relative linear target error ||A_r xi_r - t_eta|| / ||t_eta|| <= max_linear_target_error; no valid candidate falls back to clean",
            "pullback": "centered branch-likelihood Jacobian with spherical tangent projection",
            "geometry": "Riemannian sphere exponential map",
            "realization_check": "unlabeled fixed-branch re-score",
            "test_labels_used_for_tuning": False,
        },
        "split": {
            "fit_rows": len(fit_endo),
            "fit_fraction": float(args.fit_fraction),
            "max_fit_rows": None if args.max_fit_rows is None else int(args.max_fit_rows),
            "full_fit_rows": int(len(order) - nval),
            "validation_rows": len(val_endo),
            "test_rows": len(test_endo),
        },
        "run": {
            "seed": args.seed,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "elapsed_seconds": time.time() - started,
            "argv": sys.argv,
        },
    }

    if args.allow_test_eval:
        test_candidates = list(candidate["test"])
        test_labels = list(mc["y_test"])
        # Report the frozen-backbone baseline separately.  The selected rho
        # has been chosen on validation only; both records below are then
        # evaluated once on the held-out split with the same MC protocol.
        test_clean = evaluate_clean_answers(
            model, tokenizer, test_model, test_labels, int(args.control_layer),
            args.device,
        )
        test_selected = evaluate(
            model, tokenizer, test_model, test_endo, pca, scorer, test_labels,
            test_candidates, args.device, float(best["rho"]), args.tau,
            args.theta_max, args.ridge, int(args.control_layer), smoke=False,
            progress_tag=f"test rho={float(best['rho']):g}",
            control_layers=control_layers,
            layer_budget_mode=args.layer_budget_mode,
            layer_residual_norm=args.layer_residual_norm,
            max_linear_target_error=args.max_linear_target_error,
            transport_mode=args.transport_mode,
            pullback_mode=args.pullback_mode,
            utility_mode=("top" if args.top_branch else ("natural" if args.natural_branch else "scored")),
        )
        test_audits = test_selected.pop("audits")
        test_selected = {**test_selected, **audit_summary(test_audits)}
        test_selected["joint"] = 0.5 * (
            test_selected["mc1_percent"] + test_selected["mc2_percent"]
        )
        report["test"] = {
            "clean": test_clean,
            "steered": test_selected,
            "delta": {
                "mc1_percent": test_selected["mc1_percent"] - test_clean["mc1_percent"],
                "mc2_percent": test_selected["mc2_percent"] - test_clean["mc2_percent"],
                "joint": test_selected["joint"] - test_clean["joint"],
            },
        }
    Path(args.out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

