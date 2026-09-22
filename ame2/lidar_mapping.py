"""Batched LiDAR input for AME-2. Coordinates and heights are always in metres.

The policy map is [x, y, z, variance] in a gravity-aligned, yaw-following frame.
Observation provenance stays outside these four policy channels. This module
requires no ROS, Isaac Lab, CuPy, or new dependencies beyond existing PyTorch.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GridSpec:
    x_min: float = -0.4
    x_max: float = 2.0
    y_min: float = -1.0
    y_max: float = 1.0
    resolution: float = 0.05
    missing_height: float = -2.0

    def __post_init__(self):
        for length in (self.x_max - self.x_min, self.y_max - self.y_min):
            if self.resolution <= 0 or length <= 0:
                raise ValueError("Grid extents and resolution must be positive")
            if not math.isclose(length / self.resolution, round(length / self.resolution)):
                raise ValueError("Grid extents must be integer multiples of resolution")

    @property
    def width(self):
        return round((self.x_max - self.x_min) / self.resolution)

    @property
    def height(self):
        return round((self.y_max - self.y_min) / self.resolution)

    def xy(self, device, dtype=torch.float32):
        x = self.x_min + (torch.arange(self.width, device=device, dtype=dtype) + .5) * self.resolution
        y = self.y_min + (torch.arange(self.height, device=device, dtype=dtype) + .5) * self.resolution
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), -1)


@dataclass
class ProjectedScan:
    height: torch.Tensor
    observed: torch.Tensor
    count: torch.Tensor


def omni_points_metres(observation, max_distance):
    """Convert our OmniPerception [XYZ/max_distance, valid] output exactly once."""
    if observation.shape[-1] != 4 or max_distance <= 0:
        raise ValueError("Expected (..., 4) Omni observations and positive max_distance")
    xyz = observation[..., :3] * max_distance
    valid = (observation[..., 3] > .5) & torch.isfinite(xyz).all(-1)
    return xyz, valid


def transform_points(points, world_from_sensor):
    """Apply [B,4,4] or timestamp-matched per-point [B,N,4,4] transforms.

    Per-point transforms come from the synchronized odometry/deskew frontend.
    Passing one transform per scan explicitly assumes a rigid, deskewed scan.
    """
    if world_from_sensor.ndim == 3:
        return torch.bmm(points, world_from_sensor[:, :3, :3].transpose(1, 2)) + world_from_sensor[:, None, :3, 3]
    if world_from_sensor.ndim == 4:
        return (world_from_sensor[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + world_from_sensor[..., :3, 3]
    raise ValueError("Expected scan or per-point homogeneous transforms")


def world_to_local(points_world, poses):
    """poses: [B,4] = world x,y,z,yaw, with gravity-aligned world z."""
    delta = points_world - poses[:, None, :3]
    c, s = poses[:, 3:4].cos(), poses[:, 3:4].sin()
    x, y, z = delta.unbind(-1)
    return torch.stack((c * x + s * y, -s * x + c * y, z), -1)


def local_xy_to_world(xy, poses):
    c, s = poses[:, 3:4].cos(), poses[:, 3:4].sin()
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack((c*x - s*y + poses[:, 0:1], s*x + c*y + poses[:, 1:2]), -1)


def project_scan(points_sensor, valid, world_from_sensor, poses, grid=GridSpec(), *, exclude=None):
    """Project all accepted returns; empty cells retain an explicit mask.

    exclude is a per-return robot/self-hit or sensor-quality mask supplied by
    the sensor frontend. It must not be a ground-removal mask.
    """
    if poses.shape != (points_sensor.shape[0], 4):
        raise ValueError("poses must contain world x,y,z,yaw for each environment")
    local = world_to_local(transform_points(points_sensor, world_from_sensor), poses)
    good = valid.bool() & torch.isfinite(local).all(-1)
    if exclude is not None:
        good &= ~exclude.bool()
    good &= (local[..., 0] >= grid.x_min) & (local[..., 0] < grid.x_max)
    good &= (local[..., 1] >= grid.y_min) & (local[..., 1] < grid.y_max)
    safe = torch.where(good[..., None], local, torch.zeros_like(local))
    col = torch.floor((safe[..., 0] - grid.x_min) / grid.resolution).long()
    row = torch.floor((safe[..., 1] - grid.y_min) / grid.resolution).long()
    index = (row * grid.width + col).clamp(0, grid.height * grid.width - 1)
    b = local.shape[0]
    heights = local.new_full((b, grid.height * grid.width), -torch.inf)
    heights.scatter_reduce_(1, index, torch.where(good, safe[..., 2], -torch.inf), reduce="amax", include_self=True)
    counts = torch.zeros_like(heights, dtype=torch.long)
    counts.scatter_add_(1, index, good.long())
    observed = counts > 0
    heights = torch.where(observed, heights, grid.missing_height)
    shape = (b, 1, grid.height, grid.width)
    return ProjectedScan(heights.reshape(shape), observed.reshape(shape), counts.reshape(shape))


def policy_map_from_height(height, variance, grid=GridSpec()):
    """Make [x,y,z,variance], never [height,nx,ny,variance]."""
    xy = grid.xy(height.device, height.dtype).permute(2, 0, 1)[None].expand(height.shape[0], -1, -1, -1)
    return torch.cat((xy, height, variance), 1)


class LidarElevationMap(nn.Module):
    """Rolling odometry-frame history with per-environment scan/reset state.

    update consumes base-relative MappingNet height and log-variance. It also
    accepts projected height + an explicit measurement-variance model for a
    geometry-only baseline. Neither case has access to terrain ground truth.
    """
    def __init__(self, num_envs, grid=GridSpec(), *, map_cells=160,
                 unknown_variance=1.0, variance_rate=.01, max_age=2.0):
        super().__init__()
        self.grid, self.map_cells = grid, map_cells
        self.unknown_variance, self.variance_rate, self.max_age = unknown_variance, variance_rate, max_age
        n = map_cells * map_cells
        self.register_buffer("height_world", torch.zeros(num_envs, n))
        self.register_buffer("variance", torch.full((num_envs, n), unknown_variance))
        self.register_buffer("updated_at", torch.full((num_envs, n), -torch.inf))
        self.register_buffer("observed_at", torch.full((num_envs, n), -torch.inf))
        self.register_buffer("last_scan", torch.full((num_envs,), -torch.inf))
        self.register_buffer("origin_cells", torch.zeros(num_envs, 2, dtype=torch.long))
        self.register_buffer("origin_ready", torch.zeros(num_envs, dtype=torch.bool))
        self.register_buffer("local_xy", grid.xy("cpu").reshape(-1, 2))

    @torch.no_grad()
    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # Assignment, not advanced-index .zero_(), is required for partial resets.
        self.height_world[ids] = 0
        self.variance[ids] = self.unknown_variance
        self.updated_at[ids] = -torch.inf
        self.observed_at[ids] = -torch.inf
        self.last_scan[ids] = -torch.inf
        self.origin_ready[ids] = False

    @torch.no_grad()
    def _recenter(self, poses, fresh):
        new_origin = torch.floor(poses[:, :2] / self.grid.resolution).long() - self.map_cells // 2
        new_origin = torch.where(fresh[:, None], new_origin, self.origin_cells)
        shift = new_origin - self.origin_cells
        r, c = torch.meshgrid(torch.arange(self.map_cells, device=poses.device),
                             torch.arange(self.map_cells, device=poses.device), indexing="ij")
        old_c = c.flatten()[None] + shift[:, 0:1]
        old_r = r.flatten()[None] + shift[:, 1:2]
        inside = (old_c >= 0) & (old_c < self.map_cells) & (old_r >= 0) & (old_r < self.map_cells)
        inside &= self.origin_ready[:, None]
        index = (old_r * self.map_cells + old_c).clamp(0, self.map_cells**2 - 1)
        for values, fill in ((self.height_world, 0.), (self.variance, self.unknown_variance),
                             (self.updated_at, -torch.inf), (self.observed_at, -torch.inf)):
            values.copy_(torch.where(inside, values.gather(1, index), fill))
        self.origin_cells.copy_(new_origin)
        self.origin_ready |= fresh

    def _indices(self, world_xy):
        cells = torch.floor(world_xy / self.grid.resolution).long() - self.origin_cells[:, None]
        inside = (cells >= 0).all(-1) & (cells < self.map_cells).all(-1)
        index = (cells[..., 1] * self.map_cells + cells[..., 0]).clamp(0, self.map_cells**2 - 1)
        return index, inside

    @torch.no_grad()
    def update(self, height, log_variance, observed, poses, scan_time, *, generator=None):
        """Fuse each strictly newer capture timestamp once (seconds, per env).

        Within-frame collisions are reduced before stochastic WTA, avoiding
        undefined duplicate-index scatter writes on CUDA. Predictions cannot
        turn an unseen cell into a measured cell or refresh its observed age.
        """
        b, n = height.shape[0], self.local_xy.shape[0]
        fresh = scan_time > self.last_scan
        self._recenter(poses, fresh)
        xy = local_xy_to_world(self.local_xy[None].expand(b, -1, -1), poses)
        index, inside = self._indices(xy)
        h = height.reshape(b, n) + poses[:, 2:3]
        v = log_variance.reshape(b, n).exp()
        obs = observed.reshape(b, n).bool()
        # A numerical placeholder must never become a high-confidence flat cell.
        v = torch.where(obs, v, v.clamp_min(self.unknown_variance))
        accepted = inside & fresh[:, None] & torch.isfinite(h) & torch.isfinite(v) & (v > 0)
        accepted &= obs.any(1, keepdim=True)
        total = self.map_cells**2
        # Select one minimum-variance candidate per world cell, deterministic ties.
        best_v = v.new_full((b, total), torch.inf)
        best_v.scatter_reduce_(1, index, torch.where(accepted, v, torch.inf), reduce="amin")
        ranks = torch.arange(n, device=h.device)[None].expand(b, -1)
        candidate = accepted & (v == best_v.gather(1, index))
        selected = torch.full((b, total), n, device=h.device, dtype=torch.long)
        selected.scatter_reduce_(1, index, torch.where(candidate, ranks, n), reduce="amin")
        has = selected < n
        pick = selected.clamp_max(n - 1)
        h_new, v_new = h.gather(1, pick), v.gather(1, pick)
        age = (scan_time[:, None] - self.updated_at).clamp_min(0)
        prior = (self.variance + self.variance_rate * age).clamp_max(self.unknown_variance)
        empty = ~torch.isfinite(self.updated_at) | (age > self.max_age)
        effective = torch.maximum(v_new, .5 * prior)
        valid = (effective < 1.5 * prior) | (effective < .04)
        p_win = prior / (prior + effective)
        draw = torch.rand(p_win.shape, device=h.device, generator=generator)
        wins = has & (empty | (valid & (draw < p_win)))
        fused_v = torch.where(empty, v_new, effective)
        self.height_world.copy_(torch.where(wins, h_new, self.height_world))
        self.variance.copy_(torch.where(wins, fused_v, self.variance))
        self.updated_at.copy_(torch.where(wins, scan_time[:, None], self.updated_at))
        # Every accepted real return refreshes measurement provenance, even if WTA
        # keeps the old height. Inferred values never do so.
        measured = torch.zeros((b, total), device=h.device, dtype=torch.long)
        measured.scatter_reduce_(1, index, (accepted & obs).long(), reduce="amax")
        self.observed_at.copy_(torch.where(measured.bool(), scan_time[:, None], self.observed_at))
        self.last_scan.copy_(torch.where(fresh, scan_time, self.last_scan))

    @torch.no_grad()
    def crop(self, poses, now):
        """Return policy [B,4,H,W], measured mask and measurement age.

        Nearest-cell sampling preserves discontinuities and does not interpolate
        unknown heights across step edges. Outside/expired cells stay unknown.
        """
        b = poses.shape[0]
        xy = local_xy_to_world(self.local_xy[None].expand(b, -1, -1), poses)
        index, inside = self._indices(xy)
        update_age = (now[:, None] - self.updated_at.gather(1, index)).clamp_min(0)
        known = inside & self.origin_ready[:, None] & (update_age <= self.max_age)
        z = self.height_world.gather(1, index) - poses[:, 2:3]
        var = self.variance.gather(1, index) + self.variance_rate * update_age
        z = torch.where(known, z, self.grid.missing_height)
        var = torch.where(known, var.clamp_max(self.unknown_variance), self.unknown_variance)
        age = (now[:, None] - self.observed_at.gather(1, index)).clamp_min(0)
        measured = known & (age <= self.max_age)
        age = torch.where(measured, age, torch.inf)
        shape = (b, 1, self.grid.height, self.grid.width)
        policy = policy_map_from_height(z.reshape(shape), var.reshape(shape), self.grid)
        return policy, measured.reshape(shape), age.reshape(shape)
