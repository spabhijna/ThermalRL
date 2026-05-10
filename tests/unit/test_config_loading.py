from pathlib import Path

import pytest
import yaml

REQUIRED_KEYS = {
    "run_id",
    "episodes",
    "learning_rate",
    "gamma",
    "batch_size",
    "target_update_freq",
    "reward_type",
    "epsilon",
    "model_save_path",
    "csv_save_path",
}
REQUIRED_EPSILON_KEYS = {"start", "min", "decay"}


def validate_config(config):
    missing = sorted(key for key in REQUIRED_KEYS if key not in config)
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    epsilon = config.get("epsilon", {})
    missing_eps = sorted(key for key in REQUIRED_EPSILON_KEYS if key not in epsilon)
    if missing_eps:
        raise ValueError(f"Missing epsilon keys: {missing_eps}")



def test_load_all_configs():
    root = Path(__file__).resolve().parents[2]
    configs_dir = root / "configs"

    config_paths = sorted(configs_dir.glob("*.yaml"))
    assert config_paths, "No config files found in configs/"

    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text())
        validate_config(config)



def test_invalid_config():
    bad_config = {"episodes": 1, "epsilon": {"start": 1.0}}
    with pytest.raises(ValueError):
        validate_config(bad_config)
