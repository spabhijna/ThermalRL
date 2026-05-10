import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import mlflow

from mlflow_utils import configure_mlflow, default_tags

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rel import evaluate
from rel.sim.datacenter_env import DataCenterEnv


def aggregate(metrics):
    return {
        "avg_reward": float(np.mean(metrics["rewards"])) if metrics["rewards"] else 0.0,
        "avg_pue": float(np.mean(metrics["pues"])) if metrics["pues"] else 0.0,
        "avg_temp": float(np.mean(metrics["temps"])) if metrics["temps"] else 0.0,
        "viol_pct": float((metrics["violations"] / metrics["total_steps"]) * 100.0)
        if metrics["total_steps"]
        else 0.0,
    }


def save_comparison_plot(base_metrics, dqn_metrics, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(base_metrics["temp_history"], label="Baseline Controller", color="crimson", linewidth=2.5, alpha=0.75)
    plt.plot(dqn_metrics["temp_history"], label="DQN Agent", color="royalblue", linewidth=2.5, alpha=0.75)

    plt.axhline(y=35.0, color="black", linestyle="--", linewidth=2, label="Violation Threshold (>35C)")
    plt.axhline(y=32.0, color="darkorange", linestyle=":", label="Baseline Max Trigger (32C)")
    plt.axhline(y=28.0, color="forestgreen", linestyle=":", label="Baseline Min Trigger (28C)")

    plt.title("Baseline vs DQN Server Temperature Control", fontsize=15, fontweight="bold")
    plt.xlabel("Simulation Timestep", fontsize=12)
    plt.ylabel("Server Temperature (C)", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy and log to MLflow")
    parser.add_argument("--model", required=True, help="Path to policy .pkl file")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--experiment-name", default="dc-cooling-rl", help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    config = yaml.safe_load(config_path.read_text())

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT_DIR / model_path

    env = DataCenterEnv(config)
    dqn_model = evaluate.DQN()
    dqn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    dqn_model.eval()

    configure_mlflow(experiment_name=args.experiment_name, tracking_uri=args.tracking_uri)

    run_name = args.run_name or f"eval-{model_path.stem}"
    with mlflow.start_run(run_name=run_name):
        tags = default_tags()
        tags.update({"policy_filename": model_path.name})
        for key, value in tags.items():
            mlflow.set_tag(key, value)

        mlflow.log_param("episodes", args.episodes)
        mlflow.log_param("config", str(config_path))
        mlflow.log_param("reward_type", config.get("reward_type", ""))

        np.random.seed(42)
        base_metrics = evaluate.evaluate_policy(env, is_baseline=True, num_episodes=args.episodes)
        np.random.seed(42)
        dqn_metrics = evaluate.evaluate_policy(env, model=dqn_model, is_baseline=False, num_episodes=args.episodes)

        base_agg = aggregate(base_metrics)
        dqn_agg = aggregate(dqn_metrics)

        for key, value in base_agg.items():
            mlflow.log_metric(f"baseline.{key}", value)
        for key, value in dqn_agg.items():
            mlflow.log_metric(f"dqn.{key}", value)

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = Path(tmp_dir) / "temp_comparison.png"
            save_comparison_plot(base_metrics, dqn_metrics, plot_path)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(model_path), artifact_path="model")


if __name__ == "__main__":
    main()
