# Environment

The reference environment is Python 3.11, PyTorch 2.1 or newer, CUDA 12.1, Transformers 4.45 or newer, Datasets 2.18 or newer, NumPy, pandas, pyarrow, and tqdm. Exact lower bounds are in `requirements.txt`; `environment.yml` provides a CUDA Conda environment.

Set `CUDA_VISIBLE_DEVICES` before launching a run when selecting a physical GPU. Inside the process, use the corresponding logical device (usually `cuda:0`). All paths are command-line arguments or environment variables.
