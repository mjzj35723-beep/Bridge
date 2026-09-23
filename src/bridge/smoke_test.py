"""CPU-only contract test; does not download a model or dataset."""
from __future__ import annotations

import json
import torch

from .scorer import EarlyBranchScorer


def main() -> None:
    torch.manual_seed(0)
    scorer = EarlyBranchScorer(dim=8, width=64)
    z = torch.randn(2, 16, 8)
    p = torch.softmax(torch.randn(2, 16), dim=-1)
    y = scorer(z, p)
    assert y.shape == (2, 16) and torch.isfinite(y).all()
    print(json.dumps({"status": "ok", "scorer": "Linear(25,64)-GELU-LayerNorm-Linear(64,1)", "output_shape": list(y.shape)}))


if __name__ == "__main__":
    main()
