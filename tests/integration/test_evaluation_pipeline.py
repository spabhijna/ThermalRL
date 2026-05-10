import yaml
import torch

from rel import evaluate
from rel.sim.datacenter_env import DataCenterEnv


def test_evaluation_pipeline(tmp_path, temp_config):
    model_path = tmp_path / "policy.pkl"
    model = evaluate.DQN()
    torch.save(model.state_dict(), model_path)

    config_path = temp_config({"reward_type": "v10"})
    config = yaml.safe_load(config_path.read_text())

    env = DataCenterEnv(config)
    env.max_steps = 10

    loaded = evaluate.DQN()
    loaded.load_state_dict(torch.load(model_path, map_location="cpu"))
    loaded.eval()

    metrics = evaluate.evaluate_policy(env, model=loaded, is_baseline=False, num_episodes=2)
    assert len(metrics["rewards"]) == 2
    assert metrics["total_steps"] == env.max_steps * 2
    assert len(metrics["pues"]) > 0
