# Model provenance

The code was exercised with Qwen3-14B-Base for the six-dataset pilot and Qwen3-8B-Base for the original three-dataset runner. Qwen3-32B, Llama-3.1-8B, and Llama-2-13B appear in the comparison/download inventory. The repository stores only model identifiers and loader settings; checkpoint weights are downloaded separately.

Use `bfloat16` on recent NVIDIA GPUs. Use `float32` for CPU or GPUs without stable BF16 support. Qwen3-32B generally needs multiple GPUs or quantization; the release does not silently quantize it.
