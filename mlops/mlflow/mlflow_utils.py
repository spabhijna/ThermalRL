from pathlib import Path
import os

import mlflow

DEFAULT_EXPERIMENT_NAME = "dc-cooling-rl"


def resolve_tracking_uri(tracking_uri=None):
    if tracking_uri:
        return tracking_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return f"file:{Path('mlruns').resolve()}"


def configure_mlflow(experiment_name=DEFAULT_EXPERIMENT_NAME, tracking_uri=None):
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def default_tags():
    return {
        "algorithm": "DQN",
        "environment": "data_center_cooling",
        "framework": "pytorch",
        "device": "cpu",
    }


def flatten_dict(data, parent_key=""):
    items = {}
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items


def log_params_from_config(config):
    for key, value in flatten_dict(config).items():
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        mlflow.log_param(key, value)
