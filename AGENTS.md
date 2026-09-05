# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/`. Data acquisition and preprocessing are under `src/download/` and `src/preprocessing/`; model definitions, training, evaluation, losses, and diagnostics are in `src/models/`; shared helpers are in `src/utils/`. Experiment-specific Python settings live in `src/experiment_configs/<experiment>/`.

Shell pipelines are grouped in `experiments/exp1_inputs/`, `experiments/exp2_data_volume/`, and `experiments/exp3_obs/`. Top-level `*_figures.ipynb` notebooks reproduce paper figures. Generated figures, logs, and Weights & Biases runs belong in `figures/`, `logs/`, and `wandb/`; these directories are ignored and should not be committed. There is currently no dedicated `tests/` directory.

## Environment, Build, and Development Commands

- `conda env create -f environment.yml`: create the pinned Conda environment.
- `conda activate sicpred_env_conda`: activate it before running Python commands.
- `python -m src.download.download_cesm_data`: download and regrid CESM2-LE inputs after configuring paths.
- `bash experiments/exp1_inputs/run_all.sh`: preprocess, train, evaluate, and diagnose experiment 1. Equivalent scripts exist for experiments 2 and 3; run experiment 2 before experiment 3.
- `bash experiments/exp1_inputs/run_all_notrain.sh`: reproduce results from downloaded checkpoints without retraining.
- `python -m compileall src`: perform a quick syntax check.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for configuration constants. Keep imports grouped by standard library, third-party packages, then `src` modules. Add concise docstrings to reusable functions and comments only where the scientific or data-flow intent is not obvious. No formatter or linter is configured, so keep edits consistent with nearby code and avoid unrelated notebook churn.

## Testing Guidelines

No automated test framework or coverage threshold is configured. For model or preprocessing changes, run `python -m compileall src` plus the smallest relevant experiment stage (`preprocess.sh`, `evaluate.sh`, or `diagnostics.sh`). If adding tests, use `pytest`, place them in `tests/`, and name files `test_<module>.py`. Document any required datasets, checkpoints, GPU, or long runtime in the pull request.

## Sherlock Patch Fallback

On the Sherlock cluster, the sandboxed patch helper can occasionally fail with a namespace-exhaustion error. Confirm that the session is on Sherlock with:

```bash
[[ "${SLURM_CLUSTER_NAME:-}" == "sherlock" ]] || hostname -f | grep -q "\.sherlock\.stanford\.edu$"
```

If that command succeeds and `apply_patch` is unavailable, use this minimal fallback for a targeted edit: copy the current file to `/tmp`, edit only the copy, inspect the generated unified diff, run `git apply --check` on that diff, then apply it with `git apply`. Always diff against the current working-tree file (not `HEAD`) so existing user edits are preserved. Do not use this fallback outside Sherlock or for broad/mechanical rewrites.

## Configuration, Commits, and Pull Requests

Set machine-specific data, model, prediction, and analysis paths in `src/config_cesm.py`; never commit credentials or large generated artifacts. Follow the existing concise, imperative commit style, for example `Add script for bootstrapping confidence intervals`. Pull requests should state the experiment affected, commands run, configuration assumptions, and expected scientific impact. Link relevant issues and include regenerated plots when results or figure notebooks change.