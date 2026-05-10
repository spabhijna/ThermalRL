import numpy as np

from rel.train import ReplayBuffer


def test_replay_buffer_insertion_and_capacity():
    buffer = ReplayBuffer(maxlen=3)
    for i in range(4):
        buffer.push(np.zeros(3) + i, i, float(i), np.ones(3) * i, False)

    assert len(buffer) == 3
    oldest_state = buffer.buffer[0][0]
    assert np.allclose(oldest_state, np.zeros(3) + 1)


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(maxlen=10)
    for _ in range(5):
        buffer.push(np.zeros(3), 1, 1.0, np.ones(3), False)

    state, action, reward, next_state, done = buffer.sample(2)
    assert state.shape == (2, 3)
    assert next_state.shape == (2, 3)
    assert action.shape == (2,)
    assert reward.shape == (2,)
    assert done.shape == (2,)
