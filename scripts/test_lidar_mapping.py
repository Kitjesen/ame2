"""Coordinate, observation and history contracts; no simulator required."""
import math

import pytest
import torch

from ame2.lidar_mapping import (GridSpec, LidarElevationMap, omni_points_metres,
                               policy_map_from_height, project_scan, transform_points)
from ame2.networks.ame2_model import AME2Policy, MappingConfig, MappingNet, PolicyConfig

torch.set_num_threads(2)
G = GridSpec(-.5, .5, -.5, .5, .1)


def pose(b=1):
    return torch.zeros(b, 4)


def dense_scan(z=0., b=1):
    xy = G.xy("cpu").reshape(-1, 2)
    return torch.cat((xy, torch.full((len(xy), 1), z)), -1)[None].expand(b, -1, -1).clone()


def projection(points, poses=None, transform=None, valid=None, exclude=None):
    b = points.shape[0]
    return project_scan(points, torch.ones(points.shape[:2], dtype=torch.bool) if valid is None else valid,
                        torch.eye(4)[None].expand(b, -1, -1) if transform is None else transform,
                        pose(b) if poses is None else poses, G, exclude=exclude)


def fusion(b=1):
    return LidarElevationMap(b, G, map_cells=30, variance_rate=.01, max_age=1.)


def integrate(m, z, p, t, observed=None, lv=-6.):
    h = torch.full((len(p), 1, G.height, G.width), float(z))
    obs = torch.ones_like(h, dtype=torch.bool) if observed is None else observed
    m.update(h, torch.full_like(h, lv), obs, p, torch.full((len(p),), float(t)))


def test_omni_scale_and_invalid_returns():
    obs = torch.tensor([[[.1, .2, -.3, 1.], [0., 0., 0., 0.], [float("nan"), 0., 0., 1.]]])
    pts, valid = omni_points_metres(obs, 10.)
    torch.testing.assert_close(pts[0, 0], torch.tensor([1., 2., -3.]))
    assert valid.tolist() == [[True, False, False]]


def test_flat_plane_and_empty_grid_are_distinct():
    p = dense_scan(-.5)
    grid = projection(p)
    assert grid.observed.all() and (grid.count == 1).all()
    torch.testing.assert_close(grid.height, torch.full_like(grid.height, -.5))
    empty = projection(p, valid=torch.zeros(p.shape[:2], dtype=torch.bool))
    assert not empty.observed.any()
    assert (empty.height == G.missing_height).all()


def test_max_height_keeps_step_and_excludes_self_hits():
    p = torch.tensor([[[.05, .05, 0.], [.05, .05, .18], [.05, .05, .6],
                       [float("nan"), 0., 1.], [5., 5., 2.]]])
    g = projection(p, exclude=torch.tensor([[False, False, True, False, False]]))
    assert g.observed.sum() == 1 and g.count.sum() == 2
    assert g.height[0, 0, 5, 5].item() == pytest.approx(.18)


def test_half_open_bounds_do_not_clip_outside_points_to_border():
    g = projection(torch.tensor([[[-.5, -.5, .1], [.5, .1, 2.], [-.501, -.1, 3.]]]))
    assert g.count.sum() == 1
    assert g.height[0, 0, 0, 0].item() == pytest.approx(.1)


def test_full_sensor_rotation_and_translation_recover_ground_plane():
    points_world = dense_scan(-.4)
    a = .55
    t = torch.eye(4)[None]
    t[0, :3, :3] = torch.tensor([[math.cos(a), 0., math.sin(a)], [0., 1., 0.], [-math.sin(a), 0., math.cos(a)]])
    t[0, :3, 3] = torch.tensor([.1, -.1, .2])
    points_sensor = (points_world - t[:, None, :3, 3]) @ t[:, :3, :3]
    g = projection(points_sensor, transform=t)
    assert g.observed.all()
    torch.testing.assert_close(g.height, torch.full_like(g.height, -.4))


def test_per_point_transform_supports_motion_deskew():
    p = torch.tensor([[[0., 0., -1.], [0., 0., -1.]]])
    tf = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    tf[0, :, 0, 3] = torch.tensor([-.15, .15])
    tf[0, 1, 2, 3] = .18
    out = projection(p, transform=tf)
    assert out.observed.sum() == 2
    assert out.height[0, 0, 5, 3].item() == pytest.approx(-1.)
    assert out.height[0, 0, 5, 6].item() == pytest.approx(-.82)


def test_yaw_translation_and_environment_origins_do_not_change_local_grid():
    p = dense_scan(.18, b=2)
    poses = pose(2)
    poses[1] = torch.tensor([20., -10., .7, math.pi/2])
    tf = torch.eye(4)[None].repeat(2, 1, 1)
    tf[1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    tf[1, :3, 3] = poses[1, :3]
    g = projection(p, poses, tf)
    assert g.observed.all()
    torch.testing.assert_close(g.height[0], g.height[1], atol=2e-6, rtol=0)


def test_policy_channels_are_xy_z_variance():
    h = torch.full((1, 1, 10, 10), -.32)
    out = policy_map_from_height(h, torch.full_like(h, .04), G)
    torch.testing.assert_close(out[0, :2].permute(1, 2, 0), G.xy("cpu"))
    assert torch.all(out[:, 2] == -.32) and torch.all(out[:, 3] == .04)


def test_existing_depth_fusion_uses_same_channel_contract_and_partial_reset():
    from ame2.networks.ame2_model import WTAMapFusion
    from ame2.networks.rslrl_wrapper import WTAMapManager, AME2MapEnvWrapper
    from types import SimpleNamespace
    wta = WTAMapFusion(2)
    wta.global_elev.fill_(.18)
    wta.global_var.fill_(.04)
    out = wta.crop(torch.zeros(2, 3))
    xy = wta._policy_pts.T.reshape(2, wta.policy_h, wta.policy_w)
    torch.testing.assert_close(out[0, :2], xy)
    torch.testing.assert_close(out[:, 2], torch.full_like(out[:, 2], .18))
    wta.reset(torch.tensor([0]))
    assert (wta.global_elev[0] == 0).all() and (wta.global_elev[1] == .18).all()
    wrapper = object.__new__(AME2MapEnvWrapper)
    wrapper.wta_manager = SimpleNamespace(wta=wta)
    height = torch.full((2, 1, 31, 51), .18)
    local = wrapper._build_local_map(height, torch.full_like(height, math.log(.04)))
    torch.testing.assert_close(local[0, :2], xy)
    assert (local[:, 3, :, 0] == wta.INF_VAR).all(), "Area behind local scan must remain unknown"
    torch.testing.assert_close(local[:, 2, :, -2], torch.full_like(local[:, 2, :, -2], .18))
    manager = WTAMapManager(2, device="cpu")
    manager.wta.global_elev.fill_(.18)
    manager.reset(torch.tensor([], dtype=torch.long))
    assert (manager.wta.global_elev == .18).all()


def test_fusion_preserves_height_when_robot_moves_vertically():
    m, p = fusion(), pose()
    p[:, 2] = .5
    integrate(m, -.32, p, 0.)
    p[:, 2] = .7
    out, seen, _ = m.crop(p, torch.tensor([.1]))
    assert seen.all()
    torch.testing.assert_close(out[:, 2], torch.full_like(out[:, 2], -.52))


def test_map_augmentation_keeps_metric_coordinates_and_changes_terrain():
    from ame2.networks.rslrl_wrapper import _corrupt_map_batch, _shift_map_batch
    h = (G.xy("cpu")[..., 0] > 0).float()[None, None] * .18
    maps = policy_map_from_height(h, torch.full_like(h, .04), G)
    corrupted = _corrupt_map_batch(maps, 1., 1.)
    torch.testing.assert_close(corrupted[:, :2], maps[:, :2])
    assert not torch.equal(corrupted[:, 2], maps[:, 2])
    assert (corrupted[:, 3] >= 1.).all()
    torch.testing.assert_close(_corrupt_map_batch(maps, 0., 1.), maps)
    shifted = _shift_map_batch(maps, torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(shifted[:, :2], maps[:, :2])
    torch.testing.assert_close(shifted[:, 2, :, :-1], maps[:, 2, :, 1:])


def test_turning_crop_preserves_step_world_location():
    m, p = fusion(), pose()
    heights = (G.xy("cpu")[..., 0] > 0).float()[None, None] * .18
    m.update(heights, torch.full_like(heights, -6.), torch.ones_like(heights, dtype=torch.bool), p, torch.tensor([0.]))
    p[:, 3] = math.pi/2
    out, seen, _ = m.crop(p, torch.tensor([.1]))
    assert seen.all()
    expected = (G.xy("cpu")[..., 1] < 0).float() * .18
    torch.testing.assert_close(out[0, 2], expected)


def test_rolling_map_keeps_world_height_without_wraparound():
    m, p = fusion(), pose()
    integrate(m, .18, p, 0.)
    p[:, 0] = .2
    empty = torch.zeros(1, 1, 10, 10, dtype=torch.bool)
    integrate(m, 9., p, .1, observed=empty)
    out, seen, _ = m.crop(p, torch.tensor([.1]))
    assert seen[0, 0, :, :8].all() and not seen[0, 0, :, 8:].any()
    torch.testing.assert_close(out[0, 2, :, :8], torch.full((10, 8), .18))
    p[:, 0] = 20.
    integrate(m, 9., p, .2, observed=empty)
    assert not m.crop(p, torch.tensor([.2]))[1].any()


def test_duplicate_and_older_scans_do_not_reinforce_or_move_map():
    m, p = fusion(), pose()
    integrate(m, .18, p, 1.)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    p[:, 0] = 5.
    integrate(m, 4., p, 1.)
    integrate(m, 4., p, .9)
    for key, old in before.items():
        torch.testing.assert_close(m.state_dict()[key], old)


def test_partial_reset_and_empty_reset_leave_other_env_intact():
    m, p = fusion(2), pose(2)
    integrate(m, .18, p, 0.)
    before = m.height_world[1].clone()
    m.reset(torch.tensor([], dtype=torch.long))
    assert torch.isfinite(m.updated_at).any()
    m.reset(torch.tensor([0]))
    assert not torch.isfinite(m.updated_at[0]).any()
    torch.testing.assert_close(m.height_world[1], before)
    assert torch.isfinite(m.updated_at[1]).any()


def test_unseen_predictions_do_not_gain_measurement_provenance():
    m, p = fusion(), pose()
    observed = torch.zeros(1, 1, 10, 10, dtype=torch.bool)
    observed[0, 0, 4, 4] = True
    for step in range(5):
        integrate(m, .18, p, step*.1, observed=observed, lv=-12.)
    out, seen, age = m.crop(p, torch.tensor([.4]))
    assert seen.sum() == 1
    assert torch.all(out[:, 3:4][~seen] == m.unknown_variance)
    assert torch.isinf(age[~seen]).all()


def test_age_increases_uncertainty_and_expires_history():
    m, p = fusion(), pose()
    integrate(m, .18, p, 0.)
    first = m.crop(p, torch.tensor([0.]))
    later = m.crop(p, torch.tensor([.5]))
    assert torch.all(later[0][:, 3] > first[0][:, 3])
    assert torch.all(later[2] == .5)
    expired, seen, _ = m.crop(p, torch.tensor([1.1]))
    assert not seen.any()
    assert (expired[:, 2] == G.missing_height).all()
    assert (expired[:, 3] == m.unknown_variance).all()


def test_mapping_training_and_ame_student_accept_xyzu_without_ground_truth_input():
    torch.manual_seed(4)
    grid = GridSpec()
    model = MappingNet(MappingConfig(map_h=grid.height, map_w=grid.width))
    raw = torch.full((2, 1, grid.height, grid.width), -.5)
    mean, logvar = model(raw)
    loss = model.beta_nll_loss(mean, logvar, raw)
    loss.backward()
    assert torch.isfinite(loss) and model.head_unc.weight.grad.abs().sum() > 0
    cfg = PolicyConfig(map_h=grid.height, map_w=grid.width)
    student = AME2Policy(cfg, is_student=True)
    maps = policy_map_from_height(mean.detach(), logvar.detach().exp(), grid)
    actions, _, _ = student(maps, prop_hist=torch.zeros(2, cfg.prop_history, cfg.d_hist), commands=torch.zeros(2, 3))
    assert actions.shape == (2, 12) and torch.isfinite(actions).all()
    actions.square().mean().backward()
    assert student.map_encoder.local_cnn[0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)


def test_mapping_zero_residual_cannot_collapse_variance_to_nonfinite_loss():
    target = torch.zeros(1, 1, 2, 2)
    predicted = target.clone().requires_grad_()
    logvar = torch.tensor([[[[-1000., -50.], [-15., 0.]]]], requires_grad=True)
    loss = MappingNet.beta_nll_loss(predicted, logvar, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(predicted.grad).all() and torch.isfinite(logvar.grad).all()
    model = MappingNet()
    with torch.no_grad():
        model.head_unc.weight.zero_()
        model.head_unc.bias.fill_(-1000.)
        _, lv = model(torch.zeros(1, 1, 40, 48))
    torch.testing.assert_close(lv.exp(), torch.full_like(lv, model.MIN_VARIANCE))
