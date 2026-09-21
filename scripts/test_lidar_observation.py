"""Training input boundaries: frame reuse, self returns, resets and privileged data."""
import torch
from torch import nn

from ame2.lidar_mapping import GridSpec
from ame2.lidar_observation import LidarPolicyInput, ProprioceptiveHistory
from ame2.networks.thunder_actor_critic import ThunderLidarActorCritic

torch.set_num_threads(2)


class IdentityMapper(nn.Module):
    def forward(self, x):
        return x, torch.full_like(x, -6.)


def test_counts_distinguish_returns_self_hits_and_grid_occupancy():
    grid = GridSpec(-.5, .5, -.5, .5, .1)
    pipeline = LidarPolicyInput(2, IdentityMapper(), grid, max_distance=10)
    # Two returns in one cell, one outside the map, one self hit, one invalid.
    xyz = torch.tensor([[-.24, .04, -.5], [-.23, .04, -.4], [3., 0., 0.],
                        [.24, .04, 0.], [0., 0., 0.]])
    obs = torch.cat((xyz/10, torch.tensor([[1.], [1.], [1.], [1.], [0.]])), -1)[None].repeat(2, 1, 1)
    excluded = torch.tensor([[False, False, False, True, False]]).repeat(2, 1)
    poses, tf, time = torch.zeros(2, 4), torch.eye(4)[None].repeat(2, 1, 1), torch.ones(2)
    pipeline.ingest(obs, tf, poses, time, exclude=excluded)
    assert pipeline.counts.tolist() == [[5, 3, 1, 2, 1]] * 2
    saved = pipeline.history.height_world.clone()
    pipeline.ingest(obs * 0, tf, poses, time)
    torch.testing.assert_close(pipeline.history.height_world, saved)
    assert pipeline.frames.tolist() == [1, 1]
    pipeline.reset(torch.tensor([0]))
    assert pipeline.frames.tolist() == [0, 1]
    maps, measured, age = pipeline.crop(poses, time)
    assert not measured[0].any() and measured[1].sum() == 1
    assert torch.isinf(age[0]).all() and torch.isfinite(maps).all()


def test_control_history_clears_only_reset_environments():
    history = ProprioceptiveHistory(2)
    args = [torch.ones(2, n) for n in [3, 3, 16, 16, 16]]
    history.append(*args)
    history.reset(torch.tensor([0]))
    assert history.values[0].count_nonzero() == 0
    assert history.values[1, -1].sum() == 54


def test_thunder_actor_cannot_read_critic_truth_and_map_encoder_learns():
    obs = {"map": torch.randn(2, 4, 10, 10), "history": torch.randn(2, 20, 54),
           "commands": torch.randn(2, 3), "map_teacher": torch.randn(2, 3, 10, 10),
           "critic_prop": torch.randn(2, 60), "contact": torch.zeros(2, 17)}
    model = ThunderLidarActorCritic(obs, {"policy": ["map", "history", "commands"],
                                        "critic": ["map_teacher", "critic_prop", "contact"]}, 16)
    action = model.act_inference(obs)
    assert action.shape == (2, 16)
    for key in ("map_teacher", "critic_prop", "contact"):
        obs[key] = torch.full_like(obs[key], float("nan"))
    torch.testing.assert_close(model.act_inference(obs), action)
    changed = dict(obs, map=obs["map"] + .5)
    assert not torch.allclose(model.act_inference(changed), action)
    action.square().mean().backward()
    grad = model.actor.map_encoder.local_cnn[0].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Contact width follows Thunder's loaded bodies, not ANYmal's fixed 13 links.
    obs.update(map_teacher=torch.zeros(2, 3, 10, 10), critic_prop=torch.zeros(2, 60), contact=torch.zeros(2, 17))
    assert torch.isfinite(model.evaluate(obs)).all()
