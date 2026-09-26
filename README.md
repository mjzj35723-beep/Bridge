# BRIDGE

BRIDGE is an early branch supervision and native likelihood pullback method for activation steering. A frozen causal LM proposes a fixed set of top-k first-token branches, each completed by a greedy rollout. Layer-4 branch representations train a small scorer; a centered branch-likelihood Jacobian then pulls the desired branch distribution change back to a bounded residual edit at a later layer. The edit is checked by re-scoring the same fixed branches and can fall back to a clean no-op.

The canonical implementation is `src/bridge/bridge_core.py`. `src/bridge/six_dataset_runner.py` is the maintained public entry point for all six benchmark families.

## Datasets

The release supports:

- TruthfulQA multiple-choice (`truthful_qa/multiple_choice`, validation);
- MMLU (`cais/mmlu`, test);
- ARC-Easy and ARC-Challenge (`allenai/ai2_arc`, test);
- BBQ (official NYU JSONL files or the `heegyu/bbq` mirror);
- SorryBench 2025 (`sorry-bench/sorry-bench-202503`, gated).

ARC-Easy and ARC-Challenge are separate dataset configurations, not two labels for the same split. Download them with `scripts/download_datasets.py`. SorryBench requires Hugging Face access approval and a token. This source release contains no benchmark result tables or generated experiment outputs.

## Models

The experiments used these checkpoint families:

| Local name | Hugging Face repository | 
|---|---|---|
| Qwen3-8B-Base | `Qwen/Qwen3-8B-Base` | 
| Qwen3-14B-Base | `Qwen/Qwen3-14B-Base` | 
| Qwen3-32B | `Qwen/Qwen3-32B` | 
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | 
| Llama-2-13B | `NousResearch/Llama-2-13b-hf` | 

Model weights are not included. Use `scripts/download_models.py` or your institution's approved mirror. Some Llama repositories are gated.

## Scorer and protocol

For PCA rank `d`, the scorer receives `[z, z_bar, z-z_bar, log(p)]`, so its input width is `3d+1`. The default six-dataset pilot uses `d=8`, 16 branches, and an 8-token greedy rollout. The scorer is:

```text
Linear(25, 64) -> GELU -> LayerNorm(64) -> Linear(64, 1)
```

That is one hidden layer and two Linear layers. The backbone remains frozen. Candidate answer text and labels are used only to build fit-split coverage supervision; test-time steering uses the early branch states, the native branch likelihoods, and the unlabeled fixed-branch realization check.

## Install

```bash
conda env create -f environment.yml
conda activate bridge
pip install -e .
python -m bridge.smoke_test
```

The smoke test is CPU-only and does not download a model. A GPU run needs a CUDA-enabled PyTorch build, enough memory for the selected checkpoint, and `bfloat16` support (use `--dtype float32` when needed).

## Download inputs

```bash
python scripts/download_datasets.py --out ./datasets --skip-sorrybench
HF_TOKEN=... python scripts/download_datasets.py --out ./datasets
python scripts/download_models.py --out ./models --models Qwen3-14B-Base
```

The dataset downloader writes a manifest and Hugging Face `save_to_disk` directories. For BBQ you may instead point the runner at the official directory containing category JSONL files.

## Run the six-dataset pilot

```bash
PYTHONPATH=src python -m bridge.six_dataset_runner \
  --model ./models/Qwen3-14B-Base \
  --truthfulqa ./datasets/truthfulqa_multiple_choice \
  --mmlu ./datasets/mmlu/test \
  --arc-easy ./datasets/arc_easy/test \
  --arc-challenge ./datasets/arc_challenge/test \
  --bbq ./datasets/bbq \
  --sorrybench ./datasets/sorrybench/question.jsonl \
  --out ./results/six_dataset_run.json \
  --device cuda:0 --max-rows 6 --smoke
```

Remove `--max-rows 6` for a larger run. The default protocol is a short mechanism pilot (`topk=16`, `rollout=8`, early layer 4, candidate control layers 12/14/16/20). The JSON is written incrementally after each dataset.

The six-dataset runner is the maintained public entry point. `bridge_three_datasets.py` remains as an internal compatibility module because the six-dataset runner reuses its branch handling helpers.

## Repository map

- `src/bridge/bridge_core.py`: PCA, scorer, branch likelihood Jacobian, KL transport, pullback, realization controller, and metrics.
- `src/bridge/native_branch_utils.py`: model loading, prompt template, and split helpers.
Generated caches and runtime artifacts are intentionally excluded from this source release.
- `src/bridge/six_dataset_runner.py`: six-dataset normalized runner.
- `scripts/download_datasets.py` and `scripts/download_models.py`: public input/checkpoint acquisition.
- Runtime outputs: the runners write JSON to a user-selected path; generated results are excluded from this package.

No model weights, generated tensor caches, private server credentials, or private absolute paths are part of this package. Dataset and model licenses remain with their original authors; see `docs/DATASETS.md` and `docs/MODELS.md`.
The release keeps the native BRIDGE path and its reproducibility helpers; legacy anchor-field, dual-gate comparison, and pairwise representation-training scripts are omitted.

## Anonymous source boundary

This folder is source-only and intentionally contains no author identity, email address, host name, IP address, SSH material, access token, private checkpoint, local machine path, experiment result, runtime log, or generated tensor file. Runtime inputs and outputs are supplied through command-line arguments or environment variables; examples use relative paths such as `./models` and `./datasets`.

The following items are excluded from any public upload and must remain outside this folder:

- model weights and checkpoints (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.ckpt`, `*.onnx`);
- downloaded datasets, Hugging Face caches, tensor caches, and generated manifests containing local paths;
- experiment results, reports, logs, temporary files, and output directories;
- private credentials, SSH configuration, access tokens, and gated judge checkpoints;
- repository metadata or editor files that may contain local usernames or absolute paths.

