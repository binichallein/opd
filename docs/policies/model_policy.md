# Model Policy

The current clean-room OPD experiments should use official public checkpoints
unless the user explicitly asks for a different model in the current task.

## Current Defaults

- Student: official `Qwen/Qwen3-0.6B`
- Teacher: official `Qwen/Qwen3-4B`

## Do Not Use By Default

- User-trained checkpoints
- Private checkpoints
- Local checkpoints whose provenance is unclear
- Prior project checkpoints unrelated to the current OPD experiment

If a non-default model is used, record why it was used and where it came from
in the run note.
