# Data And Artifact Policy

This repository stores experiment logic and compact conclusions. It does not
store large or sensitive artifacts.

## Allowed In Git

- Source code
- Tests
- Configs
- Small JSON summaries
- Figures
- HTML reports
- Markdown result notes
- Dataset manifests and checksums

## Not Allowed In Git

- Raw datasets
- Processed datasets
- Full prediction JSONL files
- Checkpoints
- Model weights
- Training logs
- TensorBoard or W&B caches
- SSH keys, access tokens, API keys, `.env` files

## Data Manifests

Each dataset used for a serious experiment should have a manifest recording:

- Source dataset names
- Build script
- Number of examples
- Prompt template
- Seed
- File hash
- Creation time

The manifest is safe to commit if it contains no private examples.
