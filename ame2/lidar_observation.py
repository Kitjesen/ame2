"""Sensor-only mapping input, shared by replay and the Thunder training adapter."""
import torch
from torch import nn

from .lidar_mapping import GridSpec, LidarElevationMap, omni_points_metres, project_scan


class LidarPolicyInput(nn.Module):
    """Freeze the supervised mapper and retain all accepted returns before binning."""

    def __init__(self, num_envs, mapper, grid=GridSpec(), *, max_distance=10.0):
        super().__init__()
        self.mapper = mapper.eval().requires_grad_(False)
        self.grid = grid
        self.max_distance = max_distance
        self.history = LidarElevationMap(num_envs, grid)
        # Counts are per scan, not accumulated across policy steps.
        self.register_buffer("counts", torch.zeros(num_envs, 5, dtype=torch.long))
        self.register_buffer("frames", torch.zeros(num_envs, dtype=torch.long))

    @torch.no_grad()
    def ingest(self, observation, world_from_sensor, poses, scan_time, *, exclude=None):
        points, valid = omni_points_metres(observation, self.max_distance)
        fresh = scan_time > self.history.last_scan
        scan = project_scan(points, valid, world_from_sensor, poses, self.grid, exclude=exclude)
        height, log_variance = self.mapper(scan.height)
        if not torch.isfinite(height).all() or not torch.isfinite(log_variance).all():
            raise ValueError("Mapping checkpoint produced non-finite predictions")
        rejected = torch.zeros_like(valid) if exclude is None else exclude.bool()
        counts = torch.stack((
            torch.full_like(valid.sum(1), observation.shape[1]), (valid & ~rejected).sum(1),
            rejected.sum(1), scan.count.flatten(1).sum(1), scan.observed.flatten(1).sum(1),
        ), 1)
        self.counts.copy_(torch.where(fresh[:, None], counts, self.counts))
        self.frames.add_(fresh.long())
        self.history.update(height, log_variance, scan.observed, poses, scan_time)
        return scan

    def crop(self, poses, now):
        return self.history.crop(poses, now)

    def reset(self, env_ids=None):
        self.history.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.counts[ids] = 0
        self.frames[ids] = 0


class ProprioceptiveHistory(nn.Module):
    """Control-rate history in explicit action joint order; no terrain inputs."""

    def __init__(self, num_envs, length=20, num_joints=16):
        super().__init__()
        self.register_buffer("values", torch.zeros(num_envs, length, 6 + 3 * num_joints))

    @torch.no_grad()
    def append(self, angular_velocity, gravity, joint_position, joint_velocity, actions):
        frame = torch.cat((angular_velocity, gravity, joint_position, joint_velocity, actions), -1)
        self.values.copy_(torch.cat((self.values[:, 1:], frame[:, None]), 1))

    def reset(self, env_ids=None):
        self.values[slice(None) if env_ids is None else env_ids] = 0
