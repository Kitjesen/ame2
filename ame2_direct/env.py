"""AME-2 Direct Workflow Environment.

Isaac Lab DirectRLEnv — bypasses all manager overhead.
Equivalent to legged_gym coding style: all reward/obs/termination logic is inline.

API based on isaaclab 0.46.x anymal_c_env.py reference:
  - find_bodies on contact_sensor for contact-indexed ops
  - self._robot.reset(env_ids) before super()._reset_idx()
  - self._terrain.env_origins[env_ids] for terrain positions
  - net_forces_w_history for contact data

Paper: Zhang et al., "AME-2: Agile and Generalized Legged Locomotion via
       Attention-Based Neural Map Encoding", arXiv:2601.08485.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .config import AME2DirectEnvCfg


class AME2DirectEnv(DirectRLEnv):
    """AME-2 direct environment for ANYmal-D.

    All reward / obs / termination logic is inline (no manager overhead).
    """

    cfg: AME2DirectEnvCfg

    # ─────────────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, cfg: AME2DirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device

        # ── Body indices (on contact sensor for contact ops, robot for kinematics) ──
        self._base_cs_id, _     = self._contact_sensor.find_bodies("base")
        self._thigh_cs_ids, _   = self._contact_sensor.find_bodies(".*THIGH")
        self._shank_cs_ids, _   = self._contact_sensor.find_bodies(".*SHANK")
        self._foot_cs_ids, _    = self._contact_sensor.find_bodies(".*FOOT")
        self._non_foot_cs_ids   = list(self._thigh_cs_ids) + list(self._shank_cs_ids)
        self._all_cs_ids        = (                                                    # [stated Sec.IV-B]
            list(self._base_cs_id) + list(self._thigh_cs_ids)
            + list(self._shank_cs_ids) + list(self._foot_cs_ids)
        )  # 13 links: base(1) + thigh(4) + shank(4) + foot(4)

        # ── Runtime verification: log contact body ordering for L-R symmetry check ──
        # The _LR_CONTACT_PERM in rslrl_wrapper.py assumes body order [LF, RF, LH, RH]
        # within each group (consistent with original 4D perm [1,0,3,2]).
        # These prints let you verify at startup; expected with ANYmal-D USD traversal:
        #   feet:  ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT']
        #   thighs:['LF_THIGH','RF_THIGH','LH_THIGH','RH_THIGH']
        #   shanks:['LF_SHANK','RF_SHANK','LH_SHANK','RH_SHANK']
        # If order is [LF, LH, RF, RH] instead, update _LR_CONTACT_PERM to
        # [0, 3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10] in rslrl_wrapper.py.
        _body_names  = self._contact_sensor.body_names  # list[str], USD traversal order
        _foot_names  = [_body_names[i] for i in self._foot_cs_ids]
        _thigh_names = [_body_names[i] for i in self._thigh_cs_ids]
        _shank_names = [_body_names[i] for i in self._shank_cs_ids]
        print(f"[ContactOrder] feet:   {_foot_names}")
        print(f"[ContactOrder] thighs: {_thigh_names}")
        print(f"[ContactOrder] shanks: {_shank_names}")
        print(f"[ContactOrder] all_cs_ids={self._all_cs_ids}  (base+thigh+shank+foot, 13 total)")

        # Robot body indices for kinematics (thigh acc, foot vel)
        self._thigh_rb_ids, _   = self._robot.find_bodies(".*THIGH")
        self._foot_rb_ids, _    = self._robot.find_bodies(".*FOOT")

        # ── Action buffers ──
        import gymnasium as gym
        n_act = gym.spaces.flatdim(self.single_action_space)
        self._actions           = torch.zeros(n, n_act, device=dev)
        self._prev_actions      = torch.zeros(n, n_act, device=dev)
        self._processed_actions = torch.zeros(n, n_act, device=dev)

        # ── Goal command buffers ──
        self._goal_pos_w   = torch.zeros(n, 2, device=dev)   # world XY
        self._goal_heading = torch.zeros(n, device=dev)       # desired yaw (world)
        self._goal_radius  = float(cfg.goal_pos_range_init)

        # ── Stagnation detection ──
        self._stag_pos  = torch.zeros(n, 2, device=dev)
        self._stag_step = torch.zeros(n, device=dev, dtype=torch.long)

        # ── Contact history for switch detection ──
        self._prev_nf_contact = torch.zeros(n, len(self._non_foot_cs_ids), device=dev, dtype=torch.bool)

        # ── Foot contact history for air-time reward (Isaac Lab anymal_c pattern) ──
        self._prev_foot_contact = torch.zeros(n, len(self._foot_cs_ids), device=dev, dtype=torch.bool)

        # ── Terminated cache (set in _get_dones, used in _get_rewards) ──
        self._terminated = torch.zeros(n, device=dev, dtype=torch.bool)

        # ── Curriculum state (modified externally) ──
        self.heading_curriculum_frac: float = 0.0
        self.scan_noise_scale:        float = 0.0

        # ── Navigation progress buffers (privileged critic signals) ──
        # Stores d_xy and heading error from previous step to compute rates.
        self._prev_d_xy    = torch.zeros(n, device=dev)
        self._prev_yaw_err = torch.zeros(n, device=dev)

        # ── Episode sums for logging ──
        self._ep_sums = {k: torch.zeros(n, device=dev) for k in [
            "position_tracking", "position_approach", "upright_bonus",
            "heading_tracking", "moving_to_goal",
            "vel_toward_goal", "lin_vel_tracking", "standing_at_goal", "early_termination",
            "undesired_events", "base_height", "feet_air_time", "base_roll_rate", "joint_reg",
            "action_smooth", "link_contact", "link_acc",
            "jpos_lim", "jvel_lim", "jtau_lim",
        ]}

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Height scanners
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self._height_scanner_policy = RayCaster(self.cfg.height_scanner_policy)
        self.scene.sensors["height_scanner_policy"] = self._height_scanner_policy

        # Terrain
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone + filter
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions  = self._actions.clone()
        self._actions       = actions.clamp(-100.0, 100.0)
        self._processed_actions = (
            self._robot.data.default_joint_pos + self._actions * self.cfg.action_scale
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    # ─────────────────────────────────────────────────────────────────────────
    # Observations
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        actor_cmd = self._get_actor_cmd()   # (N, 3)

        # Teacher actor obs: 48D  [stated]
        obs_policy = torch.cat([
            self._robot.data.root_lin_vel_b  * 2.0,                              # 3
            self._robot.data.root_ang_vel_b  * 0.25,                             # 3
            self._robot.data.projected_gravity_b,                                 # 3
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,     # 12
            self._robot.data.joint_vel * 0.05,                                   # 12
            self._prev_actions,                                                   # 12
            actor_cmd,                                                            # 3
        ], dim=-1)  # (N, 48)

        if self.scan_noise_scale > 0.0:
            obs_policy = self._add_obs_noise(obs_policy)

        # GT policy map: (N, 3, 14, 36) — [x_rel, y_rel, z_rel] per cell [stated]
        gt_map_4d = self._get_gt_policy_map()            # (N, 3, 14, 36)

        # All-link binary contact states: (N, 13) — for AsymmetricCritic [stated Sec.IV-B]
        # Paper: "contact state of each link" — base(1)+thigh(4)+shank(4)+foot(4) = 13 links
        all_f      = self._contact_sensor.data.net_forces_w_history[:, 0, self._all_cs_ids, :]
        contact_13 = (torch.norm(all_f, dim=-1) > 1.0).float()   # (N, 13) binary
        # Foot forces needed separately for privileged obs logging
        foot_f     = self._contact_sensor.data.net_forces_w_history[:, 0, self._foot_cs_ids, :]

        # Critic prop: base(45D) + critic_cmd(5D) + nav_extra(5D) = 55D
        # base(45D)      = prop_flat[:, :45]  (base_vel+ang_vel+grav+q+dq+act)
        # critic_cmd(5D) = [x_rel, y_rel, sin(yaw), cos(yaw), t_remaining]  [stated Sec.IV-E.3]
        # nav_extra(5D)  = privileged navigation signals unavailable to actor:
        #   [v_toward_goal, d_progress_rate, heading_align_rate, vel_w_x, vel_w_y]
        prop_base  = obs_policy[:, :45]
        goal_xy_b  = self._get_goal_xy_body()            # (N, 2)
        d_yaw      = self._get_d_yaw()                   # (N,)
        t_rem      = (
            (self.max_episode_length - self.episode_length_buf).float()
            * self.step_dt / max(self.max_episode_length_s, 1.0)
        )                                                # (N,)
        critic_cmd  = torch.stack([
            goal_xy_b[:, 0].clamp(-5.0, 5.0),
            goal_xy_b[:, 1].clamp(-5.0, 5.0),
            torch.sin(d_yaw),
            torch.cos(d_yaw),
            t_rem,
        ], dim=-1)                                       # (N, 5)

        # ── Privileged navigation signals (5D) ──────────────────────────────
        # These tell the critic whether the robot is making navigation progress,
        # which the actor cannot observe directly (no ground-truth world frame).
        vel_xy   = self._robot.data.root_lin_vel_b[:, :2]          # body frame
        d_xy_raw = torch.norm(goal_xy_b, dim=1)                    # true distance
        to_goal  = goal_xy_b / (d_xy_raw.unsqueeze(1) + 1e-8)     # unit vector

        # 1. Velocity toward goal: positive = approaching, in [-1, 1] after /2
        v_proj = (vel_xy * to_goal).sum(1)

        # 2. Distance reduction rate (m/s): positive = getting closer to goal
        d_progress = (self._prev_d_xy - d_xy_raw) / max(self.step_dt, 1e-6)

        # 3. Heading alignment rate (rad/s): positive = turning toward goal
        heading_align_rate = (self._prev_yaw_err - d_yaw) / max(self.step_dt, 1e-6)

        # 4-5. World-frame XY velocity: privileged global reference frame
        #      Actor only sees noisy body-frame vel; critic sees true world vel.
        yaw     = self._get_yaw()
        cy, sy  = torch.cos(yaw), torch.sin(yaw)
        vel_w_x = cy * vel_xy[:, 0] - sy * vel_xy[:, 1]
        vel_w_y = sy * vel_xy[:, 0] + cy * vel_xy[:, 1]

        nav_extra = torch.stack([
            (v_proj / 2.0).clamp(-1.0, 1.0),           # v_toward_goal    [-1, 1]
            d_progress.clamp(-2.0, 2.0) / 2.0,         # d_progress_rate  [-1, 1]
            heading_align_rate.clamp(-5.0, 5.0) / 5.0, # heading_rate     [-1, 1]
            (vel_w_x * 2.0).clamp(-4.0, 4.0),          # world_vel_x  (same scale as body vel obs)
            (vel_w_y * 2.0).clamp(-4.0, 4.0),          # world_vel_y
        ], dim=-1)                                       # (N, 5)

        # Update prev buffers for next step's progress computation
        self._prev_d_xy    = d_xy_raw.detach().clone()
        self._prev_yaw_err = d_yaw.detach().clone()

        critic_prop = torch.cat([prop_base, critic_cmd, nav_extra], dim=-1)  # (N, 55)

        # Privileged obs for logging: height scan (1581) + foot forces (12) = 1593
        heights     = self._get_height_scan_rel()        # (N, 1581)
        foot_f_flat = foot_f.reshape(self.num_envs, -1)  # (N, 12)
        obs_priv    = torch.cat([heights, foot_f_flat], dim=-1)  # (N, 1593)

        return {
            "policy":             obs_policy,   # (N, 48) flat — RSL-RL rollout storage
            "prop":               obs_policy,   # (N, 48) — AME2ActorCritic actor
            "map":                gt_map_4d,    # (N, 3, 14, 36) — actor map input
            "map_teacher":        gt_map_4d,    # (N, 3, 14, 36) — critic map input
            "critic_prop":        critic_prop,  # (N, 55) — AsymmetricCritic input (50+5 nav)
            "contact":            contact_13,   # (N, 13) — all-link binary contact [stated Sec.IV-B]
            "teacher_privileged": obs_priv,     # (N, 1593) — logging only
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Rewards (all inline)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        rew = torch.zeros(self.num_envs, device=self.device)

        goal_xy_b = self._get_goal_xy_body()
        d_xy      = torch.norm(goal_xy_b, dim=1)

        # 1. position_tracking Eq.(1) — terminal signal, last 4s only [paper]
        r_pos = (1.0 / (1.0 + 0.25 * d_xy**2)) * self._t_mask(4.0)
        rew += cfg.w_position_tracking * r_pos
        self._ep_sums["position_tracking"] += cfg.w_position_tracking * r_pos

        # 1b. position_approach — always-on dense gradient (not in paper, fills t_mask gap)
        #     r = exp(-sigma * d_xy): sigma=0.5 → gradient non-trivial at 4m (was 1.5 → ≈0)
        #     sigma=0.5: d=4m→0.135, d=2m→0.368, d=1m→0.607, d=0.5m→0.779, d=0→1.0
        r_approach = torch.exp(-0.5 * d_xy)
        rew += cfg.w_position_approach * r_approach
        self._ep_sums["position_approach"] += cfg.w_position_approach * r_approach

        # 1c. upright_bonus — survival incentive: stay upright throughout episode
        #     Upright: g_b_z ≈ -1 (gravity points down in body frame) → r = -(-1) = 1.
        #     Inverted: g_b_z ≈ +1 → r = -(+1) clamped to 0.
        #     Fix: g_b_z.square() was symmetric (same reward for upright AND inverted).
        g_b_z = self._robot.data.projected_gravity_b[:, 2]
        r_upright = (-g_b_z).clamp(0.0, 1.0)
        rew += cfg.w_upright_bonus * r_upright
        self._ep_sums["upright_bonus"] += cfg.w_upright_bonus * r_upright

        # 1d. base_height — prevent prone local optimum (lying still avoids -100 penalty)
        #     r = clamp(height/0.6, 0, 1): 1.0 at nominal 0.6m, 0.0 on ground
        base_z    = self._robot.data.root_pos_w[:, 2]
        terrain_z = self._terrain.env_origins[:, 2]
        r_height  = torch.clamp((base_z - terrain_z) / 0.6, 0.0, 1.0)
        rew += cfg.w_base_height * r_height
        self._ep_sums["base_height"] += cfg.w_base_height * r_height

        # 1e. feet_air_time — Isaac Lab anymal_c pattern:
        #     Reward foot on FIRST CONTACT after being airborne; target = 0.5s air per step.
        #     last_air_time holds the duration of the most recent completed air phase.
        #     (last_air - 0.5): positive if foot was in the air > 0.5s, negative if < 0.5s.
        #     Only fires when robot is upright (g_b_z < -0.5 ≈ base faces up).
        foot_contact = torch.norm(
            self._contact_sensor.data.net_forces_w_history[:, 0, self._foot_cs_ids, :], dim=-1
        ) > 1.0                                                                        # (N, 4) bool
        first_contact = foot_contact & ~self._prev_foot_contact                        # just landed
        last_air = self._contact_sensor.data.last_air_time[:, self._foot_cs_ids]      # (N, 4)
        r_feet = ((last_air - 0.5) * first_contact.float()).sum(dim=1)                # (N,)
        r_feet = r_feet * (g_b_z < -0.5).float()   # only when upright (gravity down in body frame)
        self._prev_foot_contact = foot_contact.clone()
        rew += cfg.w_feet_air_time * r_feet
        self._ep_sums["feet_air_time"] += cfg.w_feet_air_time * r_feet

        # 2. heading_tracking Eq.(3)
        d_yaw = self._get_d_yaw()
        r_head = (1.0/(1.0+d_yaw**2)) * self._t_mask(2.0) * (d_xy<0.5).float()
        rew += cfg.w_heading_tracking * r_head
        self._ep_sums["heading_tracking"] += cfg.w_heading_tracking * r_head

        # 3. moving_to_goal Eq.(4) — v_min=0.3 m/s [stated], v_max=2.0 m/s [stated]
        vel_xy  = self._robot.data.root_lin_vel_b[:, :2]
        vel_nrm = torch.norm(vel_xy, dim=1)
        to_goal = goal_xy_b / (d_xy.unsqueeze(1) + 1e-8)
        cos_t   = (vel_xy * to_goal).sum(1) / (vel_nrm + 1e-8)
        r_move  = ((d_xy<0.5) | ((cos_t>0.5) & (vel_nrm>=0.3) & (vel_nrm<=2.0))).float()
        rew += cfg.w_moving_to_goal * r_move
        self._ep_sums["moving_to_goal"] += cfg.w_moving_to_goal * r_move

        # 4. vel_toward_goal (dense, kept for logging compatibility)
        v_proj = (vel_xy * to_goal).sum(1)
        r_vel  = torch.clamp(v_proj/2.0, -1.0, 1.0) * (d_xy>=0.5).float()
        rew += cfg.w_vel_toward_goal * r_vel
        self._ep_sums["vel_toward_goal"] += cfg.w_vel_toward_goal * r_vel

        # 4b. lin_vel_tracking — Isaac Lab standard formula (anymal_c_env.py)
        # cmd_vel: approach goal at min(d_xy, 0.5 m/s) in body frame
        # exp(-||cmd - vel||^2 / 0.25): dense, smooth, gradient everywhere
        # Only active when d_xy >= 0.5m (let standing_at_goal handle close range)
        cmd_vel = to_goal * torch.clamp(d_xy, max=0.5).unsqueeze(1)  # (N, 2), body frame
        lin_err = (cmd_vel - vel_xy).square().sum(1)
        r_lin   = torch.exp(-lin_err / 0.25) * (d_xy >= 0.5).float()
        rew += cfg.w_lin_vel_tracking * r_lin
        self._ep_sums["lin_vel_tracking"] += cfg.w_lin_vel_tracking * r_lin

        # 5. standing_at_goal Eq.(5)
        r_stand = self._standing_at_goal_reward(d_xy, d_yaw)
        rew += cfg.w_standing_at_goal * r_stand
        self._ep_sums["standing_at_goal"] += cfg.w_standing_at_goal * r_stand

        # 6. early_termination
        r_term = self._terminated.float()
        rew += cfg.w_early_termination * r_term
        self._ep_sums["early_termination"] += cfg.w_early_termination * r_term

        # 7. undesired_events
        r_undes = self._undesired_events()
        rew += cfg.w_undesired_events * r_undes
        self._ep_sums["undesired_events"] += cfg.w_undesired_events * r_undes

        # 8. base_roll_rate
        r_roll = self._robot.data.root_ang_vel_b[:, 0].square()
        rew += cfg.w_base_roll_rate * r_roll
        self._ep_sums["base_roll_rate"] += cfg.w_base_roll_rate * r_roll

        # 9. joint_regularization
        q_err  = (self._robot.data.joint_pos - self._robot.data.default_joint_pos).square().sum(1)
        tau_sq = self._robot.data.applied_torque.square().sum(1)
        vel_sq = self._robot.data.joint_vel.square().sum(1)
        r_jreg = q_err + 0.01*tau_sq + 0.001*vel_sq
        rew += cfg.w_joint_regularization * r_jreg
        self._ep_sums["joint_reg"] += cfg.w_joint_regularization * r_jreg

        # 10. action_smoothness
        r_smooth = (self._actions - self._prev_actions).square().sum(1)
        rew += cfg.w_action_smoothness * r_smooth
        self._ep_sums["action_smooth"] += cfg.w_action_smoothness * r_smooth

        # 11. link_contact_forces (thigh+shank only; base excluded — joint forces inflate it)
        # Only penalize non-foot, non-base links to detect body crashes into terrain.
        nf_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._non_foot_cs_ids, :]  # (N, nf, 3)
        fmag      = torch.norm(nf_forces, dim=-1)                          # (N, nf)
        excess    = torch.clamp(fmag - 1.0, min=0.0)                      # >1N contact = undesired
        r_lfc     = excess.square().sum(1)
        rew += cfg.w_link_contact_forces * r_lfc
        self._ep_sums["link_contact"] += cfg.w_link_contact_forces * r_lfc

        # 12. link_acceleration (sum of body speeds as proxy for acc)
        bvmag = torch.norm(self._robot.data.body_lin_vel_w, dim=-1).sum(1)
        rew += cfg.w_link_acceleration * bvmag
        self._ep_sums["link_acc"] += cfg.w_link_acceleration * bvmag

        # 13-15. joint limits
        pos    = self._robot.data.joint_pos
        lims   = self._robot.data.joint_pos_limits    # (N, J, 2) or (J, 2)
        if lims.dim() == 2:
            lims = lims.unsqueeze(0).expand(self.num_envs, -1, -1)
        over_u = torch.clamp(pos - 0.95*lims[:,:,1], min=0.0)
        over_l = torch.clamp(0.95*lims[:,:,0] - pos, min=0.0)
        r_jpos = (over_u+over_l).sum(1)
        rew += cfg.w_joint_pos_limits * r_jpos
        self._ep_sums["jpos_lim"] += cfg.w_joint_pos_limits * r_jpos

        vlims  = self._robot.data.joint_vel_limits
        if vlims.dim() == 1:
            vlims = vlims.unsqueeze(0).expand(self.num_envs, -1)
        vex = torch.clamp(self._robot.data.joint_vel.abs() - 0.9*vlims, min=0.0)
        r_jvel = vex.sum(1)
        rew += cfg.w_joint_vel_limits * r_jvel
        self._ep_sums["jvel_lim"] += cfg.w_joint_vel_limits * r_jvel

        elims  = self._robot.data.joint_effort_limits
        if elims.dim() == 1:
            elims = elims.unsqueeze(0).expand(self.num_envs, -1)
        tex = torch.clamp(self._robot.data.applied_torque.abs() - 0.8*elims, min=0.0)
        r_jtau = tex.sum(1)
        rew += cfg.w_joint_torque_limits * r_jtau
        self._ep_sums["jtau_lim"] += cfg.w_joint_torque_limits * r_jtau

        return rew

    # ─────────────────────────────────────────────────────────────────────────
    # Terminations
    # ─────────────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        bad_o  = self._bad_orientation()
        base_c = self._base_collision()
        thigh_a = self._high_thigh_acceleration()
        stag   = self._stagnation()
        terminated = bad_o | base_c | thigh_a | stag
        truncated  = self.episode_length_buf >= self.max_episode_length - 1

        self._terminated = terminated
        # Per-condition extras for logging
        self._term_bad_o  = bad_o
        self._term_base_c = base_c
        self._term_thigh  = thigh_a
        self._term_stag   = stag
        return terminated, truncated

    def _bad_orientation(self) -> torch.Tensor:
        """[stated Sec.IV-D.2] — skip step 0 to avoid noisy physics right after reset."""
        g = self._robot.data.projected_gravity_b
        bad = (g[:,0].abs()>0.985) | (g[:,1].abs()>0.7) | (g[:,2]>0.0)
        return bad & (self.episode_length_buf > 1)

    def _base_collision(self) -> torch.Tensor:
        """Base collision: base height drops below terrain origin level.

        The ContactSensor reports constraint/joint forces for articulated body bases
        (not just external contacts), making force-based detection unreliable.
        Use geometry: if base Z drops below terrain_origin_z the robot has collapsed.
        Threshold of -0.1m is conservative to avoid false positives on rough terrain.
        """
        base_z    = self._robot.data.root_pos_w[:, 2]
        terrain_z = self._terrain.env_origins[:, 2]
        return (base_z - terrain_z) < -0.1

    def _high_thigh_acceleration(self) -> torch.Tensor:
        """Any thigh acceleration > 60 m/s² [stated]."""
        if not hasattr(self._robot.data, 'body_lin_acc_w'):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        acc = self._robot.data.body_lin_acc_w[:, self._thigh_rb_ids, :]
        return torch.any(torch.norm(acc, dim=-1) > 60.0, dim=-1)

    def _stagnation(self) -> torch.Tensor:
        """10s displacement < 0.5m AND goal dist > 0.5m.

        Paper uses 5s, but early training needs more exploration time.
        At 5s, 86% of episodes terminate before robot can attempt locomotion.
        10s doubles the exploration window, allowing gait learning to begin.
        """
        curr = self._robot.data.root_pos_w[:, :2]
        win  = max(1, int(10.0 / self.step_dt))

        steps_since  = self.episode_length_buf - self._stag_step
        disp         = torch.norm(curr - self._stag_pos, dim=-1)
        ep_reset     = self.episode_length_buf < self._stag_step
        win_elapsed  = steps_since >= win
        upd          = ep_reset | win_elapsed
        if upd.any():
            self._stag_pos[upd]  = curr[upd].clone()
            self._stag_step[upd] = self.episode_length_buf[upd].clone()

        d_xy = torch.norm(self._get_goal_xy_body(), dim=-1)
        # Only trigger AFTER the full 5s window has elapsed.
        # Without win_elapsed, disp≈0 at episode start → instant stagnation for any d_xy>1m.
        # Use 0.5m threshold: matches the "at goal" condition (d_xy < 0.5m).
        # Original paper uses 1m, but with goal_radius=0.8m all goals are <1m → never fires.
        return win_elapsed & ~ep_reset & (disp < 0.5) & (d_xy > 0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Must reset robot FIRST (resets internal state cache)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset sensors to clear stale history buffers from previous episodes
        self._contact_sensor.reset(env_ids)
        self._height_scanner.reset(env_ids)
        self._height_scanner_policy.reset(env_ids)

        n   = len(env_ids)
        dev = self.device

        # ── Terrain curriculum: record TERMINAL goal distance BEFORE resampling ──
        # Must use robot.data which still holds pre-reset terminal positions.
        terminal_d_xy = torch.norm(self._get_goal_xy_body()[env_ids], dim=-1)

        # ── Robot state reset ──
        default_root = self._robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self._terrain.env_origins[env_ids]

        # ── Write XYZ position first so _resample_goals has correct robot XY ──
        # Pass base_xy directly from default_root[:, :2] to avoid robot.data cache lag
        # (write_root_pose_to_sim writes to PhysX but robot.data may not update until sim step)
        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        base_xy_init = default_root[:, :2].clone()

        # ── Resample goals (uses base_xy_init; must precede yaw computation) ──
        self._resample_goals(env_ids, base_xy=base_xy_init)

        # ── Initial yaw: heading curriculum [stated Sec.IV-D.3] ──
        # frac=0: robot faces goal direction (easiest for early learning).
        # frac=1: full random yaw. Interpolated probability in between.
        goal_heading_w = self._goal_heading[env_ids]           # world-frame direction to goal
        rand_yaw       = torch.rand(n, device=dev) * 2 * math.pi - math.pi
        use_rand       = torch.rand(n, device=dev) < self.heading_curriculum_frac
        yaw_init       = torch.where(use_rand, rand_yaw, goal_heading_w)

        cy, sy = torch.cos(yaw_init / 2), torch.sin(yaw_init / 2)
        default_root[:, 3] = cy        # w
        default_root[:, 4] = 0.0       # x
        default_root[:, 5] = 0.0       # y
        default_root[:, 6] = sy        # z
        # Small velocity perturbation — ±0.1 m/s: large perturbation causes instant falls
        default_root[:, 7:10]  = (torch.rand(n, 3, device=dev) - 0.5) * 0.2
        default_root[:, 10:13] = (torch.rand(n, 3, device=dev) - 0.5) * 0.2

        # Re-write with correct quaternion (overrides the temporary pose above)
        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            + (torch.rand(n, 12, device=dev) - 0.5) * 0.1
        )
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Buffers reset ──
        self._actions[env_ids]      = 0.0
        self._prev_actions[env_ids] = 0.0
        self._stag_pos[env_ids]     = self._robot.data.root_pos_w[env_ids, :2].clone()
        self._stag_step[env_ids]    = self.episode_length_buf[env_ids].clone()
        self._prev_nf_contact[env_ids]   = False
        self._prev_foot_contact[env_ids] = False

        # ── Reset navigation progress buffers ──
        d_xy_all = torch.norm(self._get_goal_xy_body(), dim=-1)
        self._prev_d_xy[env_ids]    = d_xy_all[env_ids]
        self._prev_yaw_err[env_ids] = self._get_d_yaw()[env_ids]

        # ── Terrain curriculum (using pre-resample terminal distance) ──
        self._update_terrain_curriculum(env_ids, terminal_d_xy)

        # ── Episode logging ──
        self._log_episode_stats(env_ids)

    def _resample_goals(self, env_ids: torch.Tensor,
                        base_xy: torch.Tensor | None = None) -> None:
        n   = len(env_ids)
        dev = self.device
        r   = self._goal_radius

        # Annulus sampling: [r_min, r_max] uniform area — avoids trivial at-goal rewards.
        # With full-disk sampling (old), 39% of goals land within 0.5m (at-goal threshold).
        r_min = min(0.5, r * 0.5)   # at least half r if goal_radius < 1m
        r_sq  = r**2 - r_min**2
        dist  = torch.sqrt(torch.rand(n, device=dev) * r_sq + r_min**2)
        angle = torch.rand(n, device=dev) * 2 * math.pi
        dx    = dist * torch.cos(angle)
        dy    = dist * torch.sin(angle)

        # base_xy passed explicitly avoids robot.data cache lag after write_root_pose_to_sim
        if base_xy is None:
            base_xy = self._robot.data.root_pos_w[env_ids, :2]
        self._goal_pos_w[env_ids, 0] = base_xy[:, 0] + dx
        self._goal_pos_w[env_ids, 1] = base_xy[:, 1] + dy

        # Heading curriculum: face-goal → random over training [stated Sec.IV-D.3]
        to_goal_yaw = torch.atan2(dy, dx)
        rand_yaw    = torch.rand(n, device=dev) * 2 * math.pi - math.pi
        use_rand    = torch.rand(n, device=dev) < self.heading_curriculum_frac
        self._goal_heading[env_ids] = torch.where(use_rand, rand_yaw, to_goal_yaw)

    def _update_terrain_curriculum(self, env_ids: torch.Tensor,
                                    terminal_d_xy: torch.Tensor | None = None) -> None:
        if not hasattr(self._terrain, "terrain_levels"):
            return
        # Use pre-computed terminal d_xy (distance at episode end, BEFORE goal resample).
        # If not provided, fall back to current distance (legacy behaviour).
        if terminal_d_xy is None:
            terminal_d_xy = torch.norm(self._get_goal_xy_body()[env_ids], dim=-1)
        move_up   = terminal_d_xy < 0.5
        move_down = (terminal_d_xy > 1.0) & ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _log_episode_stats(self, env_ids: torch.Tensor) -> None:
        """Log per-episode reward sums to self.extras."""
        if not hasattr(self, "extras"):
            return
        if "log" not in self.extras:
            self.extras["log"] = {}

        for key, buf in self._ep_sums.items():
            avg = buf[env_ids].mean() / max(self.max_episode_length_s, 1.0)
            self.extras["log"][f"Episode_Reward/{key}"] = avg.item()
            self._ep_sums[key][env_ids] = 0.0

        terrain_lv = self.get_terrain_level()
        self.extras["log"]["Curriculum/terrain_level"] = terrain_lv
        n = max(len(env_ids), 1)
        self.extras["log"]["Episode_Termination/bad_orientation"] = (
            getattr(self, "_term_bad_o", self.reset_terminated)[env_ids].sum().item() / n
        )
        self.extras["log"]["Episode_Termination/base_collision"] = (
            getattr(self, "_term_base_c", self.reset_terminated.new_zeros(self.num_envs))[env_ids].sum().item() / n
        )
        self.extras["log"]["Episode_Termination/thigh_acc"] = (
            getattr(self, "_term_thigh", self.reset_terminated.new_zeros(self.num_envs))[env_ids].sum().item() / n
        )
        self.extras["log"]["Episode_Termination/stagnation"] = (
            getattr(self, "_term_stag", self.reset_terminated.new_zeros(self.num_envs))[env_ids].sum().item() / n
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_goal_xy_body(self) -> torch.Tensor:
        """Goal XY in robot body frame (N, 2)."""
        base_xy = self._robot.data.root_pos_w[:, :2]
        diff_w  = self._goal_pos_w - base_xy             # (N, 2) world
        yaw     = self._get_yaw()
        cy, sy  = torch.cos(yaw), torch.sin(yaw)
        x_b     =  cy * diff_w[:, 0] + sy * diff_w[:, 1]
        y_b     = -sy * diff_w[:, 0] + cy * diff_w[:, 1]
        return torch.stack([x_b, y_b], dim=1)

    def _get_yaw(self) -> torch.Tensor:
        q = self._robot.data.root_quat_w
        return torch.atan2(
            2.0*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
            1.0 - 2.0*(q[:,2]**2 + q[:,3]**2),
        )

    def _get_actor_cmd(self) -> torch.Tensor:
        """[d_xy, sin(bearing), cos(bearing)] — bearing-to-goal in body frame.

        bearing = atan2(gxy_b[1], gxy_b[0]): direction from robot forward to goal.
        This is independent of goal_heading so it always points toward the goal
        position regardless of heading curriculum state. The final-heading signal
        reaches the actor only through the value function (critic sees goal_heading
        explicitly via critic_cmd).
        """
        gxy     = self._get_goal_xy_body()          # (N, 2) in body frame
        d_xy    = torch.norm(gxy, dim=1)
        bearing = torch.atan2(gxy[:, 1], gxy[:, 0]) # angle from forward to goal
        return torch.stack([d_xy.clamp(max=2.0), bearing.sin(), bearing.cos()], dim=1)

    def _get_d_yaw(self) -> torch.Tensor:
        err = self._goal_heading - self._get_yaw()
        return torch.atan2(torch.sin(err), torch.cos(err)).abs()

    def _t_mask(self, T: float) -> torch.Tensor:
        t_left = (self.max_episode_length - self.episode_length_buf).float() * self.step_dt
        return (1.0/T) * (t_left < T).float()

    def _get_height_scan_rel(self) -> torch.Tensor:
        """31×51 height scan relative to scanner position (N, 1581), ±1m."""
        # Follow anymal_c_env.py convention: pos_w[z] - ray_hits_w[z]
        scanner_z = self._height_scanner.data.pos_w[:, 2].unsqueeze(1)  # (N, 1)
        hits_z    = self._height_scanner.data.ray_hits_w[:, :, 2]        # (N, 1581)
        rel_z     = (scanner_z - hits_z - 0.5).clamp(-1.0, 1.0)

        if self.scan_noise_scale > 0.0:
            noise = (torch.rand_like(rel_z) - 0.5) * 2.0 * 0.05 * self.scan_noise_scale
            rel_z = (rel_z + noise).clamp(-1.0, 1.0)
        return rel_z

    def _get_gt_policy_map(self) -> torch.Tensor:
        """14×36 GT policy map: (N, 3, 14, 36) — [x_rel, y_rel, z_rel] [stated]."""
        hits  = self._height_scanner_policy.data.ray_hits_w   # (N, 504, 3)
        base  = self._robot.data.root_pos_w                    # (N, 3)
        rel   = (hits - base.unsqueeze(1)).clamp(-5.0, 5.0)   # (N, 504, 3)
        return rel.permute(0, 2, 1).reshape(self.num_envs, 3, 14, 36)  # (N, 3, 14, 36)

    def _standing_at_goal_reward(self, d_xy, d_yaw) -> torch.Tensor:
        gate   = ((d_xy<0.5) & (d_yaw<0.5)).float()
        foot_f = self._contact_sensor.data.net_forces_w_history[:, 0, self._foot_cs_ids, :]
        feet_c = (torch.norm(foot_f, dim=-1) > 1.0).float()
        d_foot = 1.0 - feet_c.sum(1) / max(len(self._foot_cs_ids), 1)
        d_g    = 1.0 - self._robot.data.projected_gravity_b[:, 2]**2
        d_q    = (self._robot.data.joint_pos - self._robot.data.default_joint_pos).abs().mean(1)
        return torch.exp(-(d_foot + d_g + d_q + d_xy) / 4.0) * gate

    def _undesired_events(self) -> torch.Tensor:
        pen   = torch.zeros(self.num_envs, device=self.device)
        forces = self._contact_sensor.data.net_forces_w_history[:, 0]  # (N, B, 3)

        # 1. Spinning
        pen += (self._robot.data.root_ang_vel_b[:, 2].abs() > 2.0).float()

        # 2. Leaping (all feet air + flat terrain)
        foot_f  = forces[:, self._foot_cs_ids, :]
        all_air = ~(torch.norm(foot_f, dim=-1) > 1.0).any(dim=1)
        h       = self._height_scanner.data.ray_hits_w[:, :, 2]
        flat    = (h.max(1).values - h.min(1).values) < 0.3
        pen += (all_air & flat).float()

        # 3. Non-foot contact
        nf_f  = forces[:, self._non_foot_cs_ids, :]
        nf_in = torch.norm(nf_f, dim=-1) > 1.0
        pen += nf_in.any(1).float()

        # 4. Non-foot contact switch
        pen += (nf_in & ~self._prev_nf_contact).any(1).float()
        self._prev_nf_contact = nf_in.clone()

        # 5. Stumbling (horizontal > vertical)
        horiz  = torch.norm(forces[:, :, :2], dim=-1)
        vert   = forces[:, :, 2].abs()
        in_c   = torch.norm(forces, dim=-1) > 1.0
        pen += (in_c & (horiz > vert)).any(1).float()

        # 6. Slippage
        foot_v = self._robot.data.body_lin_vel_w[:, self._foot_rb_ids, :]
        sp     = torch.norm(foot_v[:, :, :2], dim=-1)
        in_c_f = torch.norm(foot_f, dim=-1) > 1.0
        pen += ((sp > 0.1) & in_c_f).any(1).float()

        return pen

    def _add_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Appendix B noise [stated]."""
        noise           = torch.zeros_like(obs)
        scale           = self.scan_noise_scale
        noise[:, 0:3]   = (torch.rand_like(obs[:, 0:3]) - 0.5) * 0.2 * scale
        noise[:, 3:6]   = (torch.rand_like(obs[:, 3:6]) - 0.5) * 0.4 * scale
        noise[:, 6:9]   = (torch.rand_like(obs[:, 6:9]) - 0.5) * 0.1 * scale
        noise[:, 9:21]  = (torch.rand_like(obs[:, 9:21]) - 0.5) * 0.02 * scale
        noise[:, 21:33] = (torch.rand_like(obs[:, 21:33]) - 0.5) * 3.0 * scale
        return obs + noise

    # ─────────────────────────────────────────────────────────────────────────
    # Curriculum API
    # ─────────────────────────────────────────────────────────────────────────

    def set_heading_curriculum(self, frac: float) -> None:
        """frac=0: face goal.  frac=1: full random yaw. [stated Sec.IV-D.3]"""
        self.heading_curriculum_frac = float(frac)

    def set_scan_noise_scale(self, scale: float) -> None:
        """Linear noise ramp 0→1 over first 20% iters. [stated Sec.IV-D.3]"""
        self.scan_noise_scale = float(scale)

    def set_goal_radius(self, radius: float) -> None:
        """Expand goal sampling radius (goal distance curriculum)."""
        self._goal_radius = float(max(self.cfg.goal_pos_range_init, min(radius, self.cfg.goal_pos_range_max)))

    def get_terrain_level(self) -> float:
        if hasattr(self._terrain, "terrain_levels"):
            return self._terrain.terrain_levels.float().mean().item()
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# RSL-RL / AME2ActorCritic wrapper (no MappingNet/WTA — GT map only)
# ─────────────────────────────────────────────────────────────────────────────

class AME2DirectWrapper:
    """Thin wrapper around AME2DirectEnv for rsl_rl OnPolicyRunner + AME2ActorCritic.

    Implements rsl_rl.env.VecEnv interface:
      - get_observations() → TensorDict
      - step(actions) → (obs, rewards, dones, extras)  [4-tuple]
      - episode_length_buf, num_envs, num_actions, max_episode_length, device, cfg

    No MappingNet/WTA — uses GT height map directly as teacher map.
    Use for Phase 1 teacher PPO with the Direct workflow env.
    """

    def __init__(self, env: AME2DirectEnv, device: str = "cuda"):
        self._env      = env
        self._device   = device
        self._last_obs = None   # cached TensorDict; populated on first get_observations()

    # ── rsl_rl VecEnv required attributes ──────────────────────────────────

    @property
    def num_envs(self) -> int:            return self._env.num_envs

    @property
    def num_actions(self) -> int:         return self._env.cfg.action_space  # 12

    @property
    def max_episode_length(self) -> int:  return self._env.max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, val: torch.Tensor):
        self._env.episode_length_buf = val

    @property
    def device(self) -> str:              return self._device

    @property
    def cfg(self):                        return self._env.cfg

    # ── Extras / unwrapped ──────────────────────────────────────────────────

    @property
    def unwrapped(self):                  return self._env

    @property
    def extras(self) -> dict:             return getattr(self._env, "extras", {})

    # ── rsl_rl VecEnv required methods ─────────────────────────────────────

    def get_observations(self):
        """Return current obs as TensorDict (rsl_rl interface)."""
        if self._last_obs is None:
            # First call: reset env to get initial obs
            obs_dict, _ = self._env.reset()
            self._last_obs = self._to_td(obs_dict)
        return self._last_obs

    def step(self, actions: torch.Tensor):
        """4-tuple: (obs_td, rewards, dones, extras) — rsl_rl VecEnv interface."""
        obs_dict, reward, terminated, truncated, info = self._env.step(actions)
        dones            = terminated | truncated
        self._last_obs   = self._to_td(obs_dict)
        return self._last_obs, reward, dones, info

    def reset(self):
        """Gymnasium-style reset; also updates cached obs."""
        obs_dict, info   = self._env.reset()
        self._last_obs   = self._to_td(obs_dict)
        return self._last_obs, info

    def close(self):
        self._env.close()

    # ── Curriculum passthrough ───────────────────────────────────────────────

    def set_scan_noise_scale(self, scale: float):
        self._env.set_scan_noise_scale(scale)

    def set_heading_curriculum(self, frac: float):
        self._env.set_heading_curriculum(frac)

    def set_goal_radius(self, radius: float):
        self._env.set_goal_radius(radius)

    def get_terrain_level(self) -> float:
        return self._env.get_terrain_level()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _to_td(self, obs_dict: dict):
        """Wrap obs dict in TensorDict (or plain dict if tensordict not installed)."""
        keys = ("prop", "map", "map_teacher", "critic_prop", "contact")
        sub  = {k: obs_dict[k] for k in keys}
        try:
            from tensordict import TensorDict
            return TensorDict(sub, batch_size=[self.num_envs], device=self._device)
        except ImportError:
            return sub
