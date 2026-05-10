# DVC Usage

This project uses DVC to track large experiment artifacts without modifying the core RL code.

## Quick Start

```bash
bash mlops/dvc/setup_dvc.sh
```

## What is tracked

- experiments/
- plots/

Policies are not tracked by default.

## Common Commands

```bash
dvc status
dvc add experiments plots
dvc push
dvc pull
```

## Local Remote

The default DVC remote is stored in:

```
./dvc-storage
```

This directory is gitignored and safe to remove if you want to reset the local cache.
