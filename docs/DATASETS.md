# Dataset provenance

| Family | Source | Split used by BRIDGE | Notes |
|---|---|---|---|
| TruthfulQA | `truthful_qa/multiple_choice` | validation | MC1/MC2 labels are retained for evaluation. |
| MMLU | `cais/mmlu` | test | The runner accepts the `test` save-to-disk directory or a parquet file. |
| ARC-Easy | `allenai/ai2_arc`, `ARC-Easy` | test | Separate from ARC-Challenge. |
| ARC-Challenge | `allenai/ai2_arc`, `ARC-Challenge` | test | Separate from ARC-Easy. |
| BBQ | NYU BBQ official JSONL or `heegyu/bbq` | test/disambiguated subset as configured | Official JSONL preserves category and context condition metadata. |
| SorryBench | `sorry-bench/sorry-bench-202503` | question release | Gated; official scoring needs the SorryBench judge. |

The downloader records split sizes and local paths in `manifest.json`. For a paper release, also record the dataset revision and SHA256 of the exact files used. Do not redistribute a gated dataset or a judge checkpoint without the upstream permission.
