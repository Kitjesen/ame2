"""Isaac Lab adapter: Thunder physics/actions, MID-360 actor, privileged critic.

Import after AppLauncher. The command is (vx, vy, yaw_rate), as in the existing
Thunder velocity task, rather than the goal command in the AME-2 paper.
"""
import math

import torch
from tensordict import TensorDict
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul, quat_apply, euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from LidarSensor.example.isaaclab.isaaclab.sensors import LidarSensorCfg, LivoxPatternCfg

from .lidar_mapping import GridSpec, policy_map_from_height
from .lidar_observation import LidarPolicyInput, ProprioceptiveHistory


def teacher_grid_pattern(cfg, device):
    grid = GridSpec()
    starts = torch.zeros(grid.height * grid.width, 3, device=device)
    starts[:, :2] = grid.xy(device).reshape(-1, 2)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1
    return starts, directions


def configure_lidar_scene(env_cfg, sensor_cfg, usd_dir):
    """Add explicit sensor geometry; leave robot motor and action configuration intact."""
    if GridSpec(**sensor_cfg["grid"]) != GridSpec():
        raise ValueError("Teacher pattern and mapper currently use the default 40x48 grid")
    rpy = [torch.tensor([v]) for v in sensor_cfg["sensor_rpy_rad"]]
    rotation = tuple(quat_from_euler_xyz(*rpy)[0].tolist())
    env_cfg.scene.lazy_sensor_update = True
    # Keep task conversion artifacts separate from the shared training asset.
    env_cfg.scene.robot.spawn.usd_dir = str(usd_dir)
    common = dict(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=LidarSensorCfg.OffsetCfg(pos=tuple(sensor_cfg["sensor_translation_m"]), rot=rotation),
        max_distance=sensor_cfg["max_distance_m"], update_period=1/sensor_cfg["scan_hz"],
        pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=sensor_cfg["samples_per_scan"], downsample=1),
    )
    env_cfg.scene.mid360 = LidarSensorCfg(mesh_prim_paths=[env_cfg.scene.terrain.prim_path], **common)
    env_cfg.scene.mid360_self = LidarSensorCfg(
        # Physics collision boxes can enclose the emitter; optical occlusion uses
        # the actual surface meshes, not those deliberately coarse physics hulls.
        mesh_prim_paths=[], dynamic_env_mesh_prim_paths=["{ENV_REGEX_NS}/Robot/.*/visuals"],
        mesh_exclude_paths=["{ENV_REGEX_NS}/Robot/base_link/visuals/lidar1_Link"], min_range=0., **common)
    env_cfg.scene.ame2_truth = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link", ray_alignment="yaw",
        offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 10.)),
        pattern_cfg=patterns.PatternBaseCfg(func=teacher_grid_pattern),
        mesh_prim_paths=[env_cfg.scene.terrain.prim_path], update_period=0., debug_vis=False,
    )
    # Avoid a remote sky texture download during headless training.
    env_cfg.scene.sky_light.spawn.texture_file = None
    return env_cfg


class ThunderLidarEnv(RslRlVecEnvWrapper):
    """Expose only LiDAR maps to the student; retain the existing task rewards."""

    def __init__(self, env, mapper, sensor_cfg):
        super().__init__(env)
        if self.num_actions != 16:
            raise ValueError("Expected Thunder's 12 position + 4 velocity actions")
        self.sensor_cfg = sensor_cfg
        self.grid = GridSpec(**sensor_cfg["grid"])
        self.robot = self.unwrapped.scene["robot"]
        self.sensor = self.unwrapped.scene["mid360"]
        self.self_sensor = self.unwrapped.scene["mid360_self"]
        self.pipeline = LidarPolicyInput(self.num_envs, mapper, self.grid,
                                         max_distance=sensor_cfg["max_distance_m"]).to(self.device)
        self.prop_history = ProprioceptiveHistory(self.num_envs).to(self.device)
        self.joint_names = list(self.cfg.leg_joint_names) + list(self.cfg.wheel_joint_names)
        self.joint_ids, names = self.robot.find_joints(self.joint_names, preserve_order=True)
        if names != self.joint_names:
            raise ValueError("Thunder observation joints must follow action order")
        action_names = []
        for name in self.unwrapped.action_manager.active_terms:
            action_cfg = self.unwrapped.action_manager.get_term(name).cfg
            if not action_cfg.preserve_order:
                raise ValueError("Thunder action terms must preserve their explicit joint order")
            action_names.extend(action_cfg.joint_names)
        if action_names != self.joint_names:
            raise ValueError("Action manager order differs from configured Thunder joints")
        period = 1/sensor_cfg["scan_hz"] / self.unwrapped.step_dt
        self.scan_every = round(period)
        if self.scan_every < 1 or not math.isclose(period, self.scan_every, abs_tol=1e-6):
            raise ValueError("MID-360 period must be an integer number of control steps")
        self.tick = 0
        self.scan_updates = 0
        self.reset_events = 0
        self.cached = None
        self.last_measured = None
        self.count_sum = torch.zeros(5, device=self.device, dtype=torch.float64)
        self.count_min = torch.full((5,), torch.inf, device=self.device)
        self.count_max = torch.zeros(5, device=self.device)
        self.count_samples = 0

    @torch.no_grad()
    def _observe(self):
        b = self.num_envs
        # The sensor stores its parent's pose; apply the configured mount once.
        parent_pos = self.robot.data.root_pos_w
        parent_quat = self.robot.data.root_quat_w
        yaw = euler_xyz_from_quat(parent_quat)[2]
        poses = torch.cat((parent_pos, yaw[:, None]), -1)
        now = torch.full((b,), self.tick * self.unwrapped.step_dt, device=self.device)
        if self.tick % self.scan_every == 0:
            environment = self.sensor.get_observation()
            body = self.self_sensor.get_observation()
            # Both sensors advance the same Livox sequence on the same control tick.
            blocked = (body[..., 3] > .5) & ((environment[..., 3] < .5) |
                (body[..., :3].norm(dim=-1) <= environment[..., :3].norm(dim=-1) + 1e-5))
            data = self.sensor.data
            offset = torch.tensor(self.sensor.cfg.offset.pos, device=self.device).expand(b, -1)
            rotation = torch.tensor(self.sensor.cfg.offset.rot, device=self.device).expand(b, -1)
            tf = torch.eye(4, device=self.device)[None].repeat(b, 1, 1)
            tf[:, :3, :3] = matrix_from_quat(quat_mul(data.quat_w, rotation))
            tf[:, :3, 3] = data.pos_w + quat_apply(data.quat_w, offset)
            self.pipeline.ingest(environment, tf, poses, now, exclude=blocked)
            self.scan_updates += 1
            counts = self.pipeline.counts
            self.count_sum += counts.sum(0)
            self.count_min = torch.minimum(self.count_min, counts.min(0).values)
            self.count_max = torch.maximum(self.count_max, counts.max(0).values)
            self.count_samples += b
        maps, measured, _ = self.pipeline.crop(poses, now)
        self.last_measured = measured
        q = (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self.joint_ids].clone()
        q[:, 12:] = 0.  # Wheel rotation phase is not a bounded posture coordinate.
        dq = self.robot.data.joint_vel[:, self.joint_ids] * .05
        self.prop_history.append(self.robot.data.root_ang_vel_b, self.robot.data.projected_gravity_b,
                                 q, dq, self.unwrapped.action_manager.action)
        commands = self.unwrapped.command_manager.get_command("base_velocity")
        truth = self.unwrapped.scene["ame2_truth"].data.ray_hits_w[..., 2] - parent_pos[:, 2:3]
        if not torch.isfinite(truth).all():
            raise ValueError("Critic ground-truth rays missed terrain")
        truth = truth.reshape(b, 1, self.grid.height, self.grid.width)
        teacher_map = policy_map_from_height(truth, torch.zeros_like(truth), self.grid)[:, :3]
        contacts = self.unwrapped.scene["contact_forces"].data.net_forces_w.norm(dim=-1) > 1.
        # No original observation dictionary enters the actor adapter.
        return TensorDict({
            "map": maps, "history": self.prop_history.values.clone(), "commands": commands.clone(),
            "map_teacher": teacher_map,
            "critic_prop": torch.cat((self.robot.data.root_lin_vel_b,
                                       self.prop_history.values[:, -1], commands), -1),
            "contact": contacts.float(),
        }, batch_size=[b])

    def get_observations(self):
        if self.cached is None:
            self.cached = self._observe()
        return self.cached

    def step(self, actions):
        _, rewards, dones, extras = super().step(actions)
        self.tick += 1
        ids = dones.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.pipeline.reset(ids)
            self.prop_history.reset(ids)
            self.reset_events += len(ids)
        self.cached = self._observe()
        return self.cached, rewards, dones, extras

    def reset(self):
        _, extras = super().reset()
        self.pipeline.reset()
        self.prop_history.reset()
        self.tick = 0
        self.cached = self._observe()
        return self.cached, extras
