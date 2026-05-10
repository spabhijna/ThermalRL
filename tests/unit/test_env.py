import numpy as np

from sim.datacenter_env import DataCenterEnv


def test_reset_state_shape(env):
    state = env.reset()
    assert state.shape == (3,)
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)


def test_step_valid_action(env):
    env.reset()
    next_state, reward, done = env.step(2)
    assert next_state.shape == (3,)
    assert isinstance(reward, float)
    assert done is False
    assert np.isfinite(env.server_temp)


def test_invalid_action_clipped(env):
    env.reset()
    next_state, reward, done = env.step(10)
    assert env._prev_action == 4
    assert next_state.shape == (3,)
    assert isinstance(reward, float)
    assert done is False


def test_episode_termination(env):
    env.reset()
    env.current_step = env.max_steps - 1
    _, _, done = env.step(2)
    assert done is True


def test_overcool_penalty_v10():
    env = DataCenterEnv({"reward_type": "v10"})
    env.reset()
    env.server_temp = 15.0
    env.cooling_load = 10.0
    env.external_temp = 10.0

    _, reward, _ = env.step(4)
    it_power = env.cooling_load
    pue = (it_power + (4 * 15.0)) / it_power
    assert reward < -pue
