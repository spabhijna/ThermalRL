# DVC Pipeline Notes

This project uses DVC for lightweight artifact tracking, not for automated training pipelines.

## Tracked Outputs

- experiments/
- plots/

Policies are intentionally not tracked by default to keep existing research artifacts immutable.

## Suggested Workflow

1. Initialize and configure DVC once:

   ```bash
   bash mlops/dvc/setup_dvc.sh
   ```

2. After generating new experiments or plots, update tracking:

   ```bash
   dvc add experiments plots
   git add experiments.dvc plots.dvc
   git commit -m "Track new experiment artifacts"
   ```

3. Push artifacts to the local remote:

   ```bash
   dvc push
   ```

4. Restore artifacts on another machine:

   ```bash
   dvc pull
   ```

## Exclusions

Do not DVC-track:

- policies/
- mlruns/
- venv/.venv/
- .dvc/cache
