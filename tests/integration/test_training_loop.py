import sys

from sim.datacenter_env import DataCenterEnv
import train


def test_training_loop_small(monkeypatch, temp_config, tmp_path):
    model_path = tmp_path / "policy.pkl"
    csv_path = tmp_path / "results.csv"
    config_path = temp_config({
        "episodes": 2,
        "batch_size": 4,
        "target_update_freq": 5,
        "model_save_path": str(model_path),
        "csv_save_path": str(csv_path),
        "run_id": "test_run",
    })

    class FastEnv(DataCenterEnv):
        def __init__(self, config):
            super().__init__(config)
            self.max_steps = 10

    monkeypatch.setattr(train, "DataCenterEnv", FastEnv)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train.main()

    assert model_path.exists()
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 2
