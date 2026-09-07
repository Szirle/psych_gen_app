import torch

import utils


def test_positive_ridge_direction_is_attribute_score_ascent(monkeypatch):
    latent = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    attribute = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    control = torch.zeros(4)

    def fake_get_data(dim, **_kwargs):
        target = control if dim == "age" else attribute
        return latent, target, torch.ones(4), [f"{i}.jpg" for i in range(4)]

    monkeypatch.setattr(utils, "get_data", fake_get_data)
    monkeypatch.setattr(utils, "device", torch.device("cpu"))

    direction = utils.ridge_coefs("dominant", alpha=0.01)
    base = torch.tensor([0.25])
    positive_step = base + direction / direction.norm()

    assert direction.item() > 0
    assert torch.dot(positive_step, direction) > torch.dot(base, direction)
