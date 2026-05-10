import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
import mlflow
import mlflow.pytorch

from mlflow_utils import configure_mlflow, default_tags, log_params_from_config

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rel import evaluate


def resolve_path(root_dir, path_value):
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else root_dir / path


def load_config_index(configs_dir):
    index = {}
    for config_path in sorted(configs_dir.glob("*.yaml")):
        try:
            config = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError:
            continue
        model_path = config.get("model_save_path")
        if model_path:
            index[Path(model_path).name] = (config, config_path)
    return index


def summarize_csv(csv_path):
    if not csv_path or not csv_path.exists():
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


def log_plot_artifacts(plots_dir):
    if not plots_dir.exists():
        return
    for plot_path in sorted(plots_dir.iterdir()):
        if plot_path.is_file():
            mlflow.log_artifact(str(plot_path), artifact_path="plots")


def main():
    parser = argparse.ArgumentParser(description="Register existing policies in MLflow")
    parser.add_argument("--policies-dir", default="policies", help="Directory with saved policies")
    parser.add_argument("--configs-dir", default="configs", help="Directory with YAML configs")
    parser.add_argument("--experiments-dir", default="experiments", help="Directory with CSV logs")
    parser.add_argument("--plots-dir", default="plots", help="Directory with plots")
    parser.add_argument("--experiment-name", default="dc-cooling-rl", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--model-name", default="dc-cooling-rl", help="Registered model name")
    parser.add_argument("--log-plots", action="store_true", help="Log plots directory as artifacts")
    args = parser.parse_args()

    policies_dir = resolve_path(ROOT_DIR, args.policies_dir)
    configs_dir = resolve_path(ROOT_DIR, args.configs_dir)
    experiments_dir = resolve_path(ROOT_DIR, args.experiments_dir)
    plots_dir = resolve_path(ROOT_DIR, args.plots_dir)

    if not policies_dir.exists():
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    configure_mlflow(experiment_name=args.experiment_name, tracking_uri=args.tracking_uri)

    config_index = load_config_index(configs_dir)
    policy_paths = sorted(policies_dir.glob("*.pkl"))
    if not policy_paths:
        print(f"No policies found in {policies_dir}")
        return

    for policy_path in policy_paths:
        config_entry = config_index.get(policy_path.name)
        config = config_entry[0] if config_entry else None
        config_path = config_entry[1] if config_entry else None

        csv_path = None
        if config:
            csv_path = resolve_path(ROOT_DIR, config.get("csv_save_path"))

        run_name = policy_path.stem
        with mlflow.start_run(run_name=run_name):
            tags = default_tags()
            tags.update({
                "policy_filename": policy_path.name,
                "config_path": str(config_path) if config_path else "",
            })
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            if config:
                log_params_from_config(config)

            for metric, value in summarize_csv(csv_path).items():
                mlflow.log_metric(metric, value)

            if config_path and config_path.exists():
                mlflow.log_artifact(str(config_path), artifact_path="config")
            if csv_path and csv_path.exists():
                mlflow.log_artifact(str(csv_path), artifact_path="metrics")

            mlflow.log_artifact(str(policy_path), artifact_path="original_policy")

            if args.log_plots:
                log_plot_artifacts(plots_dir)

            model = evaluate.DQN()
            model.load_state_dict(torch.load(policy_path, map_location="cpu"))
            model.eval()

            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=args.model_name,
            )

        print(f"Registered policy: {policy_path.name}")


if __name__ == "__main__":
    main()
