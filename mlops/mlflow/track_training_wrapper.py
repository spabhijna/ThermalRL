import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml
import mlflow

from mlflow_utils import configure_mlflow, default_tags, log_params_from_config

ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def summarize_csv(csv_path):
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    metrics = {}
    if "reward" in df.columns:
        metrics["train.avg_reward"] = float(df["reward"].mean())
    if "avg_pue" in df.columns:
        metrics["train.avg_pue"] = float(df["avg_pue"].mean())
    if "epsilon" in df.columns and len(df) > 0:
        metrics["train.final_epsilon"] = float(df["epsilon"].iloc[-1])
    if "episode" in df.columns and len(df) > 0:
        metrics["train.episodes"] = int(df["episode"].max())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run training with MLflow logging")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--experiment-name", default="dc-cooling-rl", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--run-id", default=None, help="Override run_id")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text())

    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.run_id:
        config["run_id"] = args.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        model_path = tmp_dir_path / "policy.pkl"
        csv_path = tmp_dir_path / "results.csv"

        config["model_save_path"] = str(model_path)
        config["csv_save_path"] = str(csv_path)

        temp_config_path = tmp_dir_path / "config.yaml"
        temp_config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR / "src")

        subprocess.run(
            [sys.executable, "-m", "rel.train", "--config", str(temp_config_path)],
            cwd=str(ROOT_DIR),
            env=env,
            check=True,
        )

        configure_mlflow(experiment_name=args.experiment_name, tracking_uri=args.tracking_uri)
        run_name = args.run_name or f"train-{Path(args.config).stem}"

        with mlflow.start_run(run_name=run_name):
            tags = default_tags()
            tags.update({"wrapped_training": "true"})
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            log_params_from_config(config)

            for metric, value in summarize_csv(csv_path).items():
                mlflow.log_metric(metric, value)

            mlflow.log_artifact(str(temp_config_path), artifact_path="config")
            mlflow.log_artifact(str(csv_path), artifact_path="metrics")
            mlflow.log_artifact(str(model_path), artifact_path="model")


if __name__ == "__main__":
    main()
