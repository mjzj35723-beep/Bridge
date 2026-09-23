# Reproduction checklist

1. Create the environment and run `python -m bridge.smoke_test`.
2. Download the exact dataset revisions and record the generated manifest.
3. Download a checkpoint and verify its `config.json` and tokenizer load locally.
4. Run the six-dataset command with `--max-rows 6 --smoke` first.
5. Remove `--max-rows` for the desired evaluation size and keep the JSON output, seed, dtype, GPU assignment, and command line.
6. Treat SorryBench proxy values in the pilot as diagnostics only; use the official gated judge pipeline for fulfillment/compliance results.
