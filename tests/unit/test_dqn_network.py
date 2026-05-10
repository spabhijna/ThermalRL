import torch

from rel.train import DQN


def test_forward_shape_cpu():
    model = DQN()
    x = torch.zeros((4, 3))
    y = model(x)

    assert y.shape == (4, 5)
    assert next(model.parameters()).device.type == "cpu"


def test_deterministic_init_with_seed():
    torch.manual_seed(42)
    model_a = DQN()
    torch.manual_seed(42)
    model_b = DQN()

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
