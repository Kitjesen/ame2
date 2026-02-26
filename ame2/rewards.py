# Copyright (c) 2024-2026 Inovxio
# SPDX-License-Identifier: Apache-2.0

"""AME-2 Reward Functions — paper-exact implementation (Table I, Eq. 1-5).

Reference: arXiv:2601.08485, Sec. V-A and Table I.
All 14 reward terms are implemented with [stated] paper formulas.
dτ = 0.02s policy interval; weights in RewTerm already incorporate ×dτ.

Isaac Lab MDP convention:
    def reward_fn(env: ManagerBasedRLEnv, <params>, asset_cfg=...) -> Tensor
    Return shape: (num_envs,). Weights are NOT baked in — they go in RewTerm.weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# Helpers
# ============================================================================


def _t_mask(env: ManagerBasedRLEnv, T: float) -> torch.Tensor:
    """Time-based mask function, Eq. (2).

    t_mask(T) = (1/T) * 1(t_left < T)
    where t_left = (max_episode_length - current_step) * step_dt  [seconds].
    """
    t_left = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    return (1.0 / T) * (t_left < T).float()


def _goal_xy_dist(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    """Return (d_xy, goal_xy_b) from goal command.

    UniformPose2dCommandCfg outputs (N, 4) = [x_b, y_b, z_b, heading_b]
    in the robot body frame — d_xy is just the norm of (x_b, y_b).
    goal_xy_b is the body-frame direction vector to the goal.
    """
    cmd = env.command_manager.get_command(command_name)  # (N, 4)
    goal_xy_b = cmd[:, :2]       # body-frame displacement (x_b, y_b)
    d_xy = torch.norm(goal_xy_b, dim=1)
    return d_xy, goal_xy_b


# ============================================================================
# Task Rewards (Eq. 1-5)
# ============================================================================


def position_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position tracking reward — Eq. (1).

    r = 1 / (1 + 0.25 * d_xy^2) * t_mask(4)
    Weight: 2.0 (= 100 * 0.02)
    """
    d_xy, _ = _goal_xy_dist(env, asset_cfg, command_name)
    return (1.0 / (1.0 + 0.25 * d_xy ** 2)) * _t_mask(env, 4.0)


def heading_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Heading tracking reward — Eq. (3).

    r = 1 / (1 + d_yaw^2) * t_mask(2) * 1(d_xy < 0.5)
    Weight: 1.0 (= 50 * 0.02)

    d_yaw = |desired_yaw - actual_yaw|, wrapped to [0, pi].
    Goal command: [x_b, y_b, z_b, heading_b] — heading_b is scalar radians.
    """
    d_xy, _ = _goal_xy_dist(env, asset_cfg, command_name)
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # Desired yaw: cmd[:, 3] = heading_b (scalar radians, body-frame)
    desired_yaw = cmd[:, 3]
    # Actual yaw from root quaternion
    quat = asset.data.root_quat_w  # (N, 4) wxyz
    actual_yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    # Wrap to [0, pi]
    d_yaw = torch.abs(torch.atan2(torch.sin(desired_yaw - actual_yaw), torch.cos(desired_yaw - actual_yaw)))

    return (1.0 / (1.0 + d_yaw ** 2)) * _t_mask(env, 2.0) * (d_xy < 0.5).float()


def moving_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_min: float = 0.3,
    v_max: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Moving-to-goal reward — Eq. (4).

    r = 1 if (d_xy < 0.5) OR (cos(theta_v_goal) > 0.5 AND v_min <= ||v_b_xy|| <= v_max)
    else r = 0
    Weight: 0.1  (= 5 * 0.02)
    """
    d_xy, goal_xy_b = _goal_xy_dist(env, asset_cfg, command_name)
    asset = env.scene[asset_cfg.name]

    # Robot base velocity in body frame (xy), same frame as goal_xy_b
    vel_xy = asset.data.root_lin_vel_b[:, :2]  # (N, 2)
    vel_norm = torch.norm(vel_xy, dim=1)

    # Direction to goal (body frame)
    to_goal = goal_xy_b  # (N, 2) — already body-frame displacement

    # cos(theta) = dot(v, to_goal) / (||v|| * ||to_goal||)
    dot_product = torch.sum(vel_xy * to_goal, dim=1)
    denom = vel_norm * torch.norm(to_goal, dim=1) + 1e-8
    cos_theta = dot_product / denom

    at_goal = d_xy < 0.5
    moving_toward = (cos_theta > 0.5) & (vel_norm >= v_min) & (vel_norm <= v_max)

    return (at_goal | moving_toward).float()


def standing_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Standing-at-goal reward — Eq. (5).

    Active when d_xy < 0.5 AND d_yaw < 0.5.
    r = exp(-(d_foot + d_g + d_q + d_xy) / 4.0)
    Weight: 0.1  (= 5 * 0.02)

    d_foot = feet_not_in_contact / total_feet
    d_g = 1 - [g_b]_z^2
    d_q = mean(|q - q_default|)
    """
    d_xy, _ = _goal_xy_dist(env, asset_cfg, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # d_yaw: cmd[:, 3] = heading_b (scalar radians, body-frame)
    desired_yaw = cmd[:, 3]
    quat = asset.data.root_quat_w
    actual_yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    d_yaw = torch.abs(torch.atan2(torch.sin(desired_yaw - actual_yaw), torch.cos(desired_yaw - actual_yaw)))

    # Gate: only active when close and aligned
    gate = ((d_xy < 0.5) & (d_yaw < 0.5)).float()

    # d_foot: fraction of feet not in contact
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, num_feet, 3)
    feet_in_contact = (torch.norm(foot_forces, dim=-1) > 1.0).float()  # (N, num_feet)
    num_feet = feet_in_contact.shape[1]
    d_foot = 1.0 - feet_in_contact.sum(dim=1) / num_feet

    # d_g: base tilt relative to gravity
    g_b_z = asset.data.projected_gravity_b[:, 2]  # negative when upright
    d_g = 1.0 - g_b_z ** 2

    # d_q: mean deviation from default joint positions
    d_q = torch.mean(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    reward = torch.exp(-(d_foot + d_g + d_q + d_xy) / 4.0)
    return reward * gate


# ============================================================================
# Regularization and Penalties (Table I)
# ============================================================================


def early_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Early termination penalty.

    r = 1.0 if episode terminated early (not timeout).
    Weight: -500.0 (= -10 / dτ = -10 / 0.02)
    """
    return env.termination_manager.terminated.float()


def undesired_events(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elev_diff_threshold: float = 0.30,
) -> torch.Tensor:
    """Undesired events penalty — 1.0 for each event triggered (Sec. IV-D.1).

    Events (paper lists 6 types, all implemented):
      1. spinning: |omega_z| > 2.0 rad/s
      2. leaping: all feet off ground AND terrain elevation diff < 0.3 m  [stated]
      3. non_foot_contact: any non-foot link has contact force > threshold
      4. non_foot_contact_switch: non-foot link transitions from no-contact to contact
      5. stumbling: any link has horizontal contact force > vertical force
      6. slippage: any foot moving while in contact
    Weight: -1.0
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    N = asset.data.root_pos_w.shape[0]
    penalty = torch.zeros(N, device=asset.data.root_pos_w.device)

    # 1. Spinning: |omega_z| > 2.0 rad/s
    omega_z = asset.data.root_ang_vel_b[:, 2]
    penalty += (torch.abs(omega_z) > 2.0).float()

    # 2. Leaping: all feet off ground on flat terrain (elev diff < threshold)
    #    Paper: "all feet are off the ground when the elevation difference is < 30 cm"
    foot_forces = contact_sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    feet_in_contact = torch.norm(foot_forces, dim=-1) > 1.0  # (N, num_feet)
    all_feet_airborne = ~feet_in_contact.any(dim=1)  # (N,) True if NO foot is in contact
    # Estimate terrain flatness from height scanner if available, else use base height variance
    # Use a simple heuristic: check if height_scanner range is small
    if hasattr(env.scene, 'sensors') and 'height_scanner' in env.scene.sensors:
        scanner = env.scene.sensors['height_scanner']
        heights = scanner.data.ray_hits_w[:, :, 2]  # (N, num_rays, z)
        elev_range = heights.max(dim=1).values - heights.min(dim=1).values  # (N,)
        on_flat = elev_range < elev_diff_threshold
    else:
        # Fallback: assume elev diff check passes (conservative)
        on_flat = torch.ones(N, dtype=torch.bool, device=penalty.device)
    penalty += (all_feet_airborne & on_flat).float()

    # 3. Non-foot contact: any non-foot link has contact force > threshold
    non_foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    non_foot_contact_mag = torch.norm(non_foot_forces, dim=-1)  # (N, num_non_foot)
    non_foot_in_contact = non_foot_contact_mag > 1.0
    non_foot_contact = torch.any(non_foot_in_contact, dim=1)
    penalty += non_foot_contact.float()

    # 4. Non-foot contact switch: transition from no-contact to contact
    #    Paper separates "non-foot contacts" and "non-foot contact switches"
    #    to allow stable interactions while discouraging new unnecessary contacts.
    if hasattr(contact_sensor.data, 'net_forces_w_history') and contact_sensor.data.net_forces_w_history.shape[1] > 1:
        prev_non_foot_forces = contact_sensor.data.net_forces_w_history[:, 1, sensor_cfg.body_ids, :]
        prev_non_foot_contact = torch.norm(prev_non_foot_forces, dim=-1) > 1.0
        contact_switch = torch.any(non_foot_in_contact & ~prev_non_foot_contact, dim=1)
        penalty += contact_switch.float()

    # 5. Stumbling: horizontal force > vertical force for any link in contact
    all_forces = contact_sensor.data.net_forces_w  # (N, num_bodies, 3)
    horiz_force = torch.norm(all_forces[:, :, :2], dim=-1)
    vert_force = torch.abs(all_forces[:, :, 2])
    in_contact = torch.norm(all_forces, dim=-1) > 1.0
    stumbling = torch.any((horiz_force > vert_force) & in_contact, dim=1)
    penalty += stumbling.float()

    # 6. Slippage: foot moving while in contact
    foot_vel = asset.data.body_lin_vel_w[:, foot_sensor_cfg.body_ids, :]
    foot_speed = torch.norm(foot_vel[:, :, :2], dim=-1)
    slipping = torch.any((foot_speed > 0.1) & feet_in_contact, dim=1)
    penalty += slipping.float()

    return penalty



def base_roll_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base roll rate penalty.

    r = [omega_b]_x^2  (roll rate = x-axis angular velocity)
    Weight: -0.1
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])


def joint_regularization(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint regularization penalty — Table I.

    r = ||q - q_default||^2 + 0.01*||tau||^2 + 0.001*||qdot||^2
    Weight: -0.001
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    tau_sq = torch.sum(torch.square(asset.data.applied_torque), dim=1)
    vel_sq = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    return q_err + 0.01 * tau_sq + 0.001 * vel_sq


def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action smoothness penalty — Table I.

    r = ||a_t - a_{t-1}||^2
    Weight: -0.01
    """
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def link_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Link contact forces penalty — Table I.

    F_con = contact force for each link
    G = robot total weight (mass * 9.81)
    r = ||max(F_con - G, 0)||^2  summed over all links
    Weight: -0.00001
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w  # (N, num_bodies, 3)
    force_mag = torch.norm(forces, dim=-1)  # (N, num_bodies)
    G = asset.data.default_mass.sum(dim=1, keepdim=True) * 9.81  # (N, 1)
    excess = torch.clamp(force_mag - G, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def link_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Link acceleration (velocity) penalty — Table I.

    r = sum_over_all_links(||v_l||)  (1-norm of velocity, not squared)
    Weight: -0.001
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w
    return torch.sum(torch.norm(body_vel, dim=-1), dim=1)


# ============================================================================
# Simulation Fidelity Rewards (Table I)
# ============================================================================


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint position limits penalty — Table I.

    r = sum_j max(0, q_j - 0.95*q_j_max, 0.95*q_j_min - q_j)
    Weight: -1000.0
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    limits = asset.data.joint_pos_limits  # (N, num_joints, 2)
    q_min = limits[:, :, 0]
    q_max = limits[:, :, 1]
    over_upper = torch.clamp(pos - 0.95 * q_max, min=0.0)
    over_lower = torch.clamp(0.95 * q_min - pos, min=0.0)
    return torch.sum(over_upper + over_lower, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocity limits penalty — Table I.

    r = sum_j max(0, |qdot_j| - 0.9 * qdot_max_j)
    Weight: -1.0
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel
    vel_limits = asset.data.joint_vel_limits
    excess = torch.clamp(torch.abs(vel) - 0.9 * vel_limits, min=0.0)
    return torch.sum(excess, dim=1)


def joint_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint torque limits penalty — Table I.

    r = sum_j max(0, |tau_j| - 0.8 * tau_max_j)
    Weight: -1.0
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque
    effort_limits = asset.data.joint_effort_limits
    excess = torch.clamp(torch.abs(torques) - 0.8 * effort_limits, min=0.0)
    return torch.sum(excess, dim=1)


# ============================================================================
# Reward Configuration Dict — 14 terms matching Table I exactly
# ============================================================================

AME2_ANYMAL_D_REWARDS_CFG = {
    # === Task Rewards (weight * dτ) ===
    "position_tracking": RewTerm(
        func=position_tracking,
        weight=2.0,  # 100 * 0.02
        params={"command_name": "goal_pos"},
    ),
    "heading_tracking": RewTerm(
        func=heading_tracking,
        weight=1.0,  # 50 * 0.02
        params={"command_name": "goal_pos"},
    ),
    "moving_to_goal": RewTerm(
        func=moving_to_goal,
        weight=0.1,  # 5 * 0.02
        params={"command_name": "goal_pos", "v_min": 0.3, "v_max": 2.0},
    ),
    "standing_at_goal": RewTerm(
        func=standing_at_goal,
        weight=0.1,  # 5 * 0.02
        params={
            "command_name": "goal_pos",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_ids=".*FOOT"),
        },
    ),
    # === Regularization and Penalties ===
    "early_termination": RewTerm(
        func=early_termination,
        weight=-500.0,  # -10 / dτ = -10 / 0.02
        params={},
    ),
    "undesired_events": RewTerm(
        func=undesired_events,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_ids=".*THIGH|.*SHANK"),
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_ids=".*FOOT"),
        },
    ),
    "base_roll_rate": RewTerm(func=base_roll_rate, weight=-0.1, params={}),
    "joint_regularization": RewTerm(func=joint_regularization, weight=-0.001, params={}),
    "action_smoothness": RewTerm(func=action_smoothness, weight=-0.01, params={}),
    "link_contact_forces": RewTerm(
        func=link_contact_forces,
        weight=-0.00001,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    ),
    "link_acceleration": RewTerm(func=link_acceleration, weight=-0.001, params={}),
    # === Simulation Fidelity ===
    "joint_pos_limits": RewTerm(func=joint_pos_limits, weight=-1000.0, params={}),
    "joint_vel_limits": RewTerm(func=joint_vel_limits, weight=-1.0, params={}),
    "joint_torque_limits": RewTerm(func=joint_torque_limits, weight=-1.0, params={}),
}
