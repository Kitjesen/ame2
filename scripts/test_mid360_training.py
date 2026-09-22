"""Checks for the supervised MID-360 experiment's loss and evaluation contracts."""
import torch

from ame2.lidar_mapping import GridSpec
from ame2.networks.ame2_model import MappingNet
from ame2.networks.ame2_model import MappingConfig
from ame2.networks.lidar_mapping_model import LidarContextMappingNet
from scripts.train_mid360_mapping import (edges, metrics, nearest_fill, sample_weights,
                                         edge_gradient_loss, training_inputs, load_initial_weights)


def test_flat_batch_and_flat_examples_keep_training_signal():
    truth = torch.full((3, 1, 6, 6), -.5)
    weights = sample_weights(truth)
    torch.testing.assert_close(weights, torch.ones(3) / 3)
    prediction = torch.zeros_like(truth, requires_grad=True)
    loss = MappingNet.beta_nll_loss(prediction, torch.zeros_like(truth), truth, tv_weights=weights)
    loss.backward()
    assert prediction.grad.abs().sum() > 0
    truth[1, :, :, 3:] += .2
    weights = sample_weights(truth)
    assert weights[0] > 0 and weights[2] > 0 and weights[1] > weights[0]
    torch.testing.assert_close(weights.sum(), torch.tensor(1.))


def test_unknown_cells_and_step_edges_are_evaluated_separately():
    grid = GridSpec(x_min=0., x_max=.2, y_min=0., y_max=.15)
    truth = torch.zeros(1, 1, 3, 4)
    truth[..., 2:] = .2
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., :2] = True
    prediction = torch.zeros_like(truth)
    result = metrics(prediction, truth, observed, grid)
    assert result["observed_mae_m"] == 0.
    assert abs(result["unknown_mae_m"] - .2) < 1e-6
    assert edges(truth)[0, 0].tolist() == [[False, True, True, False]] * 3
    assert result["edge_recall_within_one_cell"] == 0.


def test_nearest_baseline_only_uses_observed_values():
    grid = GridSpec(x_min=0., x_max=.2, y_min=0., y_max=.15)
    raw = torch.full((2, 1, 3, 4), grid.missing_height)
    observed = torch.zeros_like(raw, dtype=torch.bool)
    raw[0, 0, 1, 1], observed[0, 0, 1, 1] = -.4, True
    filled = nearest_fill(raw, observed, grid)
    torch.testing.assert_close(filled[0], torch.full_like(filled[0], -.4))
    torch.testing.assert_close(filled[1], raw[1])


def test_context_model_connects_blind_cell_to_distant_observations():
    model = LidarContextMappingNet(MappingConfig(map_h=40, map_w=48))
    # Average pooling gives the structural receptive-field support without
    # the incidental one-winner routing of a particular max-pool input.
    model.pool = torch.nn.AvgPool2d(3, 3)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.constant_(module.weight, .01)
            torch.nn.init.constant_(module.bias, .01)
    x = torch.ones(1, 1, 40, 48, requires_grad=True)
    mean, log_var = model(x)
    assert mean.shape == log_var.shape == x.shape
    mean[0, 0, 20, 32].backward()
    assert x.grad[0, 0, 20, 8].abs() > 0  # 1.2 m away at 5 cm resolution.
    explicit_mask = torch.ones_like(x, dtype=torch.bool)
    torch.testing.assert_close(model(x, explicit_mask)[0], mean)


def test_gradient_loss_distinguishes_steps_from_smoothed_or_false_edges():
    truth = torch.zeros(1, 1, 4, 6)
    truth[..., 3:] = .2
    assert edge_gradient_loss(truth, truth) == 0
    smooth = torch.linspace(0, .2, 6).expand_as(truth).clone().requires_grad_()
    loss = edge_gradient_loss(smooth, truth)
    assert loss > 0
    loss.backward()
    assert smooth.grad[..., 2].mean() > 0
    assert smooth.grad[..., 3].mean() < 0
    assert edge_gradient_loss(truth, torch.zeros_like(truth)) > 0


def test_clean_frame_mixture_keeps_each_mask_with_its_scan():
    noisy = torch.full((8, 1, 3, 4), -2.)
    clean = torch.ones_like(noisy)
    batch = {"raw": noisy, "observed": noisy > 0, "clean_raw": clean,
             "clean_observed": clean > 0, "truth": torch.full_like(clean, torch.nan)}
    generator = torch.Generator().manual_seed(923)
    raw, mask = training_inputs(batch, torch.arange(8), .5, generator)
    assert mask.any() and (~mask).any()
    assert torch.isfinite(raw).all()
    torch.testing.assert_close(raw == 1, mask)
    assert all(frame.all() or not frame.any() for frame in mask)


def test_initial_weights_reject_a_different_physical_mount(tmp_path):
    import json
    from pathlib import Path
    import pytest
    config = json.loads((Path(__file__).parents[1] / "configs/thunder_v4_mid360_side_view.json").read_text())
    model = LidarContextMappingNet(MappingConfig())
    path = tmp_path / "init.pt"
    torch.save({"model": model.state_dict(), "config": config, "model_kind": "lidar-context", "steps": 5500}, path)
    assert load_initial_weights(model, path, config, "lidar-context", "cpu") == 5500
    config["sensor_rpy_rad"] = [0., 0., 0.]
    with pytest.raises(ValueError, match="sensor_rpy_rad"):
        load_initial_weights(model, path, config, "lidar-context", "cpu")
