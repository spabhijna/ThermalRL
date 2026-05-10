import sys
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.datacenter_env import DataCenterEnv


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    yield


@pytest.fixture
def env():
    return DataCenterEnv({"reward_type": "v1"})


@pytest.fixture
def temp_config(tmp_path):
    def _make(overrides=None):
        config = {
            "run_id": "test_run",
            "episodes": 2,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 4,
            "target_update_freq": 5,
            "reward_type": "v1",
            "epsilon": {"start": 1.0, "min": 0.05, "decay": 0.99},
            "model_save_path": str(tmp_path / "policy.pkl"),
            "csv_save_path": str(tmp_path / "results.csv"),
        }
        if overrides:
            config.update(overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    return _make
