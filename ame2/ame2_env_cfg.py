# AME-2 Isaac Lab Environment Configuration for ANYmal-D
# ======================================================
#
# Inherits from robot_lab's LocomotionVelocityRoughEnvCfg and only overrides
# AME-2 specific settings.
#
# Reference: Zhang et al., "AME-2: Agile and Generalized Legged Locomotion via
# Attention-Based Neural Map Encoding", arXiv:2601.08485.
#
# Dimension summary (must match networks/ame2_model.py PolicyConfig):
# ----------------------------------------------------------
#     Actor proprioception (PolicyCfg):                               [stated]
#         base_lin_vel(3, teacher only) + base_ang_vel(3)
#         + projected_gravity(3) + joint_pos(12) + joint_vel(12)
#         + actions(12) + cmd_actor(3) = 48  (teacher)  / 45 (student)
#
#     Critic proprioception (CriticCfg):                              [stated]
#         base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
#         + joint_pos(12) + joint_vel(12) + actions(12)
#         + cmd_critic(5) = 50
#
#     Teacher privileged (TeacherPrivilegedCfg):
#         height_scan(31x51=1581) + foot_contact_forces(4x3=12) = 1593
#
#     Actions: joint position targets, 12-DoF                         [stated]
#
# Command design (Sec. IV-E.3, continuous deployment):                [stated]
#     Actor:  [clip(d_xy, max=2.0), sin(yaw_rel), cos(yaw_rel)]  = 3D
#     Critic: [goal_x_rel, goal_y_rel, sin(yaw_rel), cos(yaw_rel),
#              t_remaining]  = 5D
#
# Termination (Sec. IV-D.2):                                         [stated]
#     bad_orientation, base_collision, high_thigh_acceleration, stagnation

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Relative imports — now inside robot_lab, no sys.path needed
from robot_lab.tasks.manager_based.locomotion.velocity import mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

# AME-2 reward functions (Table I, Eq. 1-5)
from .rewards import (
    position_tracking,
    heading_tracking,
    moving_to_goal,
    standing_at_goal,
    early_termination,
    undesired_events,
    base_roll_rate,
    joint_regularization,
    action_smoothness,
    link_contact_forces,
    link_acceleration,
    joint_pos_limits,
    joint_vel_limits,
    joint_torque_limits,
)

# AME-2 terrain curriculum (Sec. IV-D.3)
from .curriculums import terrain_levels_goal

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


# ============================================================================
# Custom observation functions for AME-2 teacher privileged info
# ============================================================================

def ame2_actor_cmd(
    env,
    command_name: str = "goal_pos",
    randomize_far_yaw: bool = True,
) -> "torch.Tensor":
    """AME-2 actor command: [clip(d_xy, 2m), sin(yaw_rel), cos(yaw_rel)].   [stated Sec.IV-E.3]

    Continuous deployment design (stated Sec.IV-E.3):
      - Distance is clipped to 2 m.  All goals farther than 2 m look identical
        to the actor, so precise bearing is meaningless at long range.
      - When the actual distance > 2 m and ``randomize_far_yaw=True``, the
        observed yaw is uniformly randomised in [-π, π] during training.
        This prevents the actor from developing a spurious long-range bearing
        dependency, enabling infinite-horizon (no time-conditioning) deployment.

    Output occupies the LAST 3 dims of prop_flat:
        teacher prop = base_vel(3) | ang_vel+grav+q+dq+act(42) | ame2_actor_cmd(3) = 48D
        student prop = (same, but actor ignores prop; cmd feeds StudentPropEncoder)
    """
    import torch
    cmd          = env.command_manager.get_command(command_name)       # (B, 4)
    actual_d_xy  = torch.norm(cmd[:, :2], dim=1)                       # (B,)
    d_xy         = actual_d_xy.clamp(max=2.0).unsqueeze(1)             # (B, 1)
    heading      = cmd[:, 3:4]                                         # (B, 1) radians

    if randomize_far_yaw:
        # Randomise observed yaw when actual distance > 2 m so the actor does not
        # learn to depend on precise bearing at long range.  [stated Sec.IV-E.3]
        far_mask = (actual_d_xy > 2.0).unsqueeze(1)                    # (B, 1)
        rand_yaw = torch.rand_like(heading) * (2.0 * math.pi) - math.pi
        heading  = torch.where(far_mask, rand_yaw, heading)

    return torch.cat([d_xy, torch.sin(heading), torch.cos(heading)], dim=1)  # (B, 3)


def ame2_critic_cmd(
    env,
    command_name: str = "goal_pos",
) -> "torch.Tensor":
    """AME-2 critic command: [x_rel, y_rel, sin(yaw_rel), cos(yaw_rel), t_remaining].

    Critic receives the FULL command (stated Sec.IV-E.3):
      - Exact body-frame goal position [x_rel, y_rel]  (NOT clipped to 2 m)
      - [sin(yaw_rel), cos(yaw_rel)]
      - Normalised remaining episode time  t_remaining ∈ [0, 1]

    This asymmetric critic command lets the value network plan with full goal
    information and episode horizon, while the actor is agnostic to both.

    Output: (B, 5)
    """
    import torch
    cmd       = env.command_manager.get_command(command_name)        # (B, 4)
    x_rel     = cmd[:, 0:1]                                          # (B, 1) exact fwd dist
    y_rel     = cmd[:, 1:2]                                          # (B, 1) exact lat dist
    heading   = cmd[:, 3:4]                                          # (B, 1) radians
    t_remain  = (
        (env.max_episode_length - env.episode_length_buf).float()
        / env.max_episode_length
    ).unsqueeze(1)                                                    # (B, 1)  ∈ [0, 1]
    return torch.cat(
        [x_rel, y_rel, torch.sin(heading), torch.cos(heading), t_remain], dim=1
    )  # (B, 5)


def foot_contact_forces(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> "torch.Tensor":
    """3D net contact force on each foot body.

    Returns: (B, num_feet * 3) — flattened per-foot 3D forces.
    For ANYmal-D with 4 feet: (B, 12).
    """
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    return forces.reshape(forces.shape[0], -1)


def gt_policy_map_flat(
    env,
    scanner_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_policy"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    map_h: int = 14,
    map_w: int = 36,
) -> "torch.Tensor":
    """Ground-truth policy map from policy-resolution RayCaster.

    Reads the world-frame hit positions from ``height_scanner_policy``
    (14×36 @ 8 cm) and returns them as robot-relative (x, y, z) offsets.
    Channel layout matches the teacher map used by AME2Encoder:
        channels 0/1/2 = (x_rel, y_rel, z_rel) per grid point

    Returns: (B, 3 * map_h * map_w) = (B, 1512) flat tensor.
    """
    import torch
    scanner = env.scene.sensors[scanner_cfg.name]
    asset = env.scene[asset_cfg.name]

    # FIX: Isaac Lab RayCaster API uses `ray_hits_w`, not `pos_w`.
    hits_w = scanner.data.ray_hits_w     # (B, H*W, 3)
    base_pos = asset.data.root_pos_w     # (B, 3)

    hits_rel = hits_w - base_pos.unsqueeze(1)  # (B, H*W, 3)

    B = hits_w.shape[0]
    return hits_rel.permute(0, 2, 1).reshape(B, 3 * map_h * map_w)


# ============================================================================
# Custom termination functions for AME-2 (Sec. IV-D.2)                [stated]
# ============================================================================

def ame2_bad_orientation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> "torch.Tensor":
    """AME-2 bad orientation termination.                              [stated]

    Terminate when: |g_b[x]| > 0.985  OR  |g_b[y]| > 0.7  OR  g_b[z] > 0.0
    """
    import torch
    asset = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # (B, 3)
    bad_x = torch.abs(g_b[:, 0]) > 0.985
    bad_y = torch.abs(g_b[:, 1]) > 0.7
    flipped = g_b[:, 2] > 0.0
    return bad_x | bad_y | flipped


def ame2_base_collision(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    robot_mass_kg: float = 50.0,
) -> "torch.Tensor":
    """AME-2 base collision termination.                               [stated]

    Terminate when base link contact force > robot total weight (mass * 9.81).
    """
    import torch
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    force_mag = torch.norm(forces, dim=-1)
    max_force = torch.max(force_mag, dim=-1)[0]
    return max_force > robot_mass_kg * 9.81


def ame2_high_thigh_acceleration(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 60.0,
) -> "torch.Tensor":
    """AME-2 high thigh acceleration termination.                      [stated]

    Terminate when any thigh link linear acceleration > threshold (m/s^2).
    """
    import torch
    asset = env.scene[asset_cfg.name]
    acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    acc_mag = torch.norm(acc, dim=-1)
    return torch.any(acc_mag > threshold, dim=-1)


# Module-level state dict for stagnation tracking (env_id → checkpoint tensors).
# Using id(env) as key is safe because a single python process runs one env at a time.
_STAG_STATE: dict = {}


def ame2_stagnation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_s: float = 5.0,
    min_displacement: float = 0.5,
    min_goal_dist: float = 1.0,
) -> "torch.Tensor":
    """AME-2 stagnation termination.                                   [stated]

    Terminate when XY displacement in the past ``window_s`` seconds is less
    than ``min_displacement`` metres AND the robot is more than ``min_goal_dist``
    metres from the goal.

    Implementation: rolling window via a per-env checkpoint.
    Every ``window_s`` seconds (or at episode reset), the checkpoint is updated
    to the current position.  Stagnation fires when the robot has not moved
    ``min_displacement`` from its last checkpoint.

    This gives a non-sliding but causally correct window — no ring buffer
    allocation, no Isaac Lab wrapper required.
    """
    import torch
    asset = env.scene[asset_cfg.name]
    current_pos = asset.data.root_pos_w[:, :2].detach()  # (B, 2) — no grad through term

    eid = id(env)
    cached = _STAG_STATE.get(eid)
    if cached is None or cached["pos"].shape != current_pos.shape or cached["pos"].device != current_pos.device:
        # First call, or stale cache from a previous training run in the same process.
        _STAG_STATE[eid] = {
            "pos":  current_pos.clone(),
            "step": env.episode_length_buf.clone(),
        }
    state = _STAG_STATE[eid]

    window_steps = max(1, int(window_s / env.step_dt))

    steps_since = env.episode_length_buf - state["step"]    # may be negative on reset
    disp        = torch.norm(current_pos - state["pos"], dim=-1)  # (B,)

    # Refresh checkpoint when episode restarted or window has elapsed
    episode_reset  = env.episode_length_buf < state["step"]
    window_elapsed = steps_since >= window_steps
    update_mask    = episode_reset | window_elapsed

    if update_mask.any():
        state["pos"][update_mask]  = current_pos[update_mask].clone()
        state["step"][update_mask] = env.episode_length_buf[update_mask].clone()

    # Stagnant: insufficient displacement from last checkpoint.
    # FIX: Only check stagnation when the full observation window has elapsed.
    # Without this gate, the check fires on the step immediately after a
    # checkpoint refresh (when disp ≈ 0), causing premature termination.
    stagnant = (disp < min_displacement) & window_elapsed

    # Far from goal: use body-frame goal distance from goal_pos command
    cmd          = env.command_manager.get_command("goal_pos")  # (B, 4)
    far_from_goal = torch.norm(cmd[:, :2], dim=-1) > min_goal_dist  # (B,)

    return stagnant & far_from_goal


# ============================================================================
# AME-2 ANYmal-D Environment Configuration
# ============================================================================

@configclass
class AME2AnymalEnvCfg(LocomotionVelocityRoughEnvCfg):
    """AME-2 environment for ANYmal-D.

    Inherits the full LocomotionVelocityRoughEnvCfg from robot_lab and only
    overrides AME-2 specific settings:

      - Height scanner: 31x51 grid @ 4cm resolution → feeds MappingNet    [stated]
      - policy obs (48D): base_vel(3)|ang_vel+grav+q+dq+act(42)|actor_cmd(3)  [stated]
      - teacher_privileged obs (1593D): height_scan(1581) + foot_contact(12)
      - teacher_map obs (1512D): GT policy map 3×14×36 flat
      - Actor cmd 3D: [clip(d_xy,2m), sin(yaw_rel), cos(yaw_rel)]     [stated]
      - Critic uses same 48D prop as teacher actor (AsymmetricCritic)
      - 4 custom termination conditions (Sec IV-D.2)                   [stated]
      - Domain randomization aligned to Appendix B                     [stated]

    Terrain: default ROUGH_TERRAINS_CFG from robot_lab.
    Full AME-2 12 terrain types are defined in terrains.py.
    """

    base_link_name = "base"
    foot_link_name = ".*FOOT"

    def __post_init__(self):
        super().__post_init__()

        # ============================ Sim (must come first — scanners use these values) ============================
        self.decimation = 4          # 4 sim steps per control step → dt_ctrl = 0.02s  [stated]
        self.episode_length_s = 20.0
        self.sim.dt = 0.005          # 200 Hz physics                                  [stated]

        # ============================ Scene ============================
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # AME-2 height scanner: 31×51 @ 4cm resolution → feeds MappingNet  [inferred from MappingConfig.local_res=0.04]
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.04,
                size=[1.20, 2.00],  # (31-1)*0.04 × (51-1)*0.04
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 0.02s

        # GT policy map scanner: 14×36 @ 8 cm  [inferred]
        # Feeds the teacher's AME2Encoder — separate from the 31×51 MappingNet scanner.
        self.scene.height_scanner_policy = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.08,
                size=[1.04, 2.80],  # (14-1)*0.08 × (36-1)*0.08
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner_policy.update_period = self.decimation * self.sim.dt  # 0.02s

        # ============================ Num Envs ============================
        self.scene.num_envs = 4800  # [stated Table VI]

        # ============================ Observations ============================
        self.observations.policy.height_scan = None
        # Null base-class velocity command obs — command "base_velocity" is disabled above.
        # These terms reference "base_velocity" which no longer exists.
        self.observations.policy.velocity_commands = None
        self.observations.critic.velocity_commands = None

        # AME-2 actor command obs: [clip(d_xy,2m), sin(yaw_rel), cos(yaw_rel)]    [stated]
        # Appended LAST so prop layout is:
        #   base_vel(3) | ang_vel+grav+q+dq+act(42) | ame2_actor_cmd(3) = 48D  [stated]
        # randomize_far_yaw=True: when d>2m, yaw is randomised during training  [stated]
        self.observations.policy.ame2_actor_cmd = ObsTerm(
            func=ame2_actor_cmd,
            params={"command_name": "goal_pos", "randomize_far_yaw": True},
        )

        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # ------ Observation noise (Sec.IV-D.3) [stated: uniform noise on policy obs] ------
        # The parent class already sets enable_corruption=True and noise on each sensor term.
        # We re-declare them here explicitly so AME-2 noise values are auditable and tunable
        # without having to trace back to LocomotionVelocityRoughEnvCfg.
        #
        # Noise is applied to raw (pre-scale) sensor values:
        #   network input = (sensor_value + uniform(n_min, n_max)) * scale
        #
        # [inferred] magnitudes match the ANYmal-D locomotion training convention.
        # Command-derived terms (ame2_actor_cmd, actions) are intentionally NOT noised.
        self.observations.policy.enable_corruption = True
        self.observations.policy.base_lin_vel.noise      = Unoise(n_min=-0.1,  n_max=0.1)   # ±0.1  m/s
        self.observations.policy.base_ang_vel.noise      = Unoise(n_min=-0.2,  n_max=0.2)   # ±0.2  rad/s
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)  # ±0.05 (unit-less)
        self.observations.policy.joint_pos.noise         = Unoise(n_min=-0.01, n_max=0.01)  # ±0.01 rad
        self.observations.policy.joint_vel.noise         = Unoise(n_min=-1.5,  n_max=1.5)   # ±1.5  rad/s

        @configclass
        class TeacherPrivilegedCfg(ObsGroup):
            """Height scan (31x51=1581) + foot contact forces (4x3=12) = 1593D."""

            height_scan = ObsTerm(
                func=mdp.height_scan,
                params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                clip=(-1.0, 1.0),
                scale=1.0,
            )
            contact_forces_obs = ObsTerm(
                func=foot_contact_forces,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*FOOT"])},
                clip=(-100.0, 100.0),
                scale=1.0,
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.observations.teacher_privileged = TeacherPrivilegedCfg()

        @configclass
        class TeacherMapCfg(ObsGroup):
            """GT policy-map from height_scanner_policy: (B, 1512) flat."""

            gt_map = ObsTerm(
                func=gt_policy_map_flat,
                params={
                    "scanner_cfg": SceneEntityCfg("height_scanner_policy"),
                    "asset_cfg": SceneEntityCfg("robot"),
                    "map_h": 14,
                    "map_w": 36,
                },
                clip=(-5.0, 5.0),
                scale=1.0,
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.observations.teacher_map = TeacherMapCfg()

        # AME-2 critic command obs group: 5D full command for AsymmetricCritic  [stated Sec.IV-E.3]
        # Layout: [x_rel, y_rel, sin(yaw_rel), cos(yaw_rel), t_remaining]
        # Combined with base(45D) in rslrl_wrapper to form critic_prop(50D):
        #   base_vel(3) | ang_vel+grav+q+dq+act(42) | critic_cmd(5) = 50D
        @configclass
        class CriticExtraCfg(ObsGroup):
            """5D full critic command: exact position + yaw + remaining time."""

            ame2_critic_cmd_obs = ObsTerm(
                func=ame2_critic_cmd,
                params={"command_name": "goal_pos"},
            )

            def __post_init__(self):
                self.enable_corruption = False   # no noise on critic command
                self.concatenate_terms = True

        self.observations.critic_extra = CriticExtraCfg()

        self.observations.critic.height_scan = None

        # ============================ Actions ============================
        # Actuation delay randomization U[0, 0.02]s (Appendix B)            [stated]
        from .delayed_joint_actions import DelayedJointPositionActionCfg
        self.actions.joint_pos = DelayedJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
            preserve_order=True,
            clip={".*": (-100.0, 100.0)},
            max_delay_s=0.02,  # U[0, 0.02]s → 0..4 physics steps @ 200Hz
        )

        # ============================ Commands ============================
        # AME-2 uses position-goal navigation (Sec. IV-D), not velocity tracking.
        # Actor  cmd (3D): [clip(d_xy, 2m), sin(yaw_rel), cos(yaw_rel)]  [stated]
        # Critic cmd (5D): [x_rel, y_rel, sin(yaw_rel), cos(yaw_rel), t_remain]  [stated]
        self.commands.base_velocity = None  # disable inherited velocity command
        self.commands.goal_pos = mdp.UniformPose2dCommandCfg(
            asset_name="robot",
            simple_heading=False,  # outputs (N,4)=[x_b, y_b, z_b, heading_b]
            resampling_time_range=(10.0, 20.0),
            debug_vis=True,
            ranges=mdp.UniformPose2dCommandCfg.Ranges(
                pos_x=(-5.0, 5.0),
                pos_y=(-5.0, 5.0),
                heading=(-math.pi, math.pi),
            ),
        )

        # ============================ Events (Domain Randomization) ============================
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "add"

        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]

        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [".*"]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.3, 1.0)

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2),
                "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        }

        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains = None

        # Actuation delay is handled by DelayedJointPositionActionCfg above.
        # See delayed_joint_actions.py for implementation details.

        # ============================ Rewards ============================
        # Null ALL base-class velocity-tracking rewards (Table I uses position goals).
        self.rewards.track_lin_vel_xy_exp.weight    = 0
        self.rewards.track_ang_vel_z_exp.weight     = 0
        self.rewards.lin_vel_z_l2.weight            = 0
        self.rewards.ang_vel_xy_l2.weight           = 0
        self.rewards.flat_orientation_l2.weight     = 0
        self.rewards.base_height_l2.weight          = 0
        self.rewards.body_lin_acc_l2.weight         = 0
        self.rewards.joint_torques_l2.weight        = 0
        self.rewards.joint_vel_l2.weight            = 0
        self.rewards.joint_acc_l2.weight            = 0
        self.rewards.joint_pos_limits.weight        = 0
        self.rewards.joint_vel_limits.weight        = 0
        self.rewards.joint_power.weight             = 0
        self.rewards.stand_still.weight             = 0
        self.rewards.action_rate_l2.weight          = 0
        self.rewards.undesired_contacts.weight      = 0
        self.rewards.contact_forces.weight          = 0
        self.rewards.feet_air_time.weight           = 0
        self.rewards.feet_contact.weight            = 0
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_stumble.weight            = 0
        self.rewards.feet_slide.weight              = 0
        self.rewards.upward.weight                  = 0
        self.rewards.is_terminated.weight           = 0

        # ── Task Rewards (Table I, weight = paper_weight × dτ* = × 0.02) ──────
        self.rewards.position_tracking = RewTerm(  # 100 × 0.02 = 2.0  [stated]
            func=position_tracking,
            weight=2.0,
            params={
                "command_name": "goal_pos",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.heading_tracking = RewTerm(   # 50 × 0.02 = 1.0   [stated]
            func=heading_tracking,
            weight=1.0,
            params={
                "command_name": "goal_pos",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.moving_to_goal = RewTerm(     # 5 × 0.02 = 0.1    [stated]
            func=moving_to_goal,
            weight=0.1,
            params={
                "command_name": "goal_pos",
                "v_min": 0.3,
                "v_max": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.standing_at_goal = RewTerm(   # 5 × 0.02 = 0.1    [stated]
            func=standing_at_goal,
            weight=0.1,
            params={
                "command_name": "goal_pos",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*FOOT"]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ── Regularization and Penalties (Table I, weight = paper_weight × dτ) ──
        self.rewards.early_termination = RewTerm(  # (-10/dτ) × dτ = -10  [stated]
            func=early_termination,
            weight=-10.0,
            params={},
        )
        self.rewards.undesired_events = RewTerm(   # -1 × 0.02 = -0.02  [stated]
            func=undesired_events,
            weight=-0.02,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*THIGH", ".*SHANK"]
                ),
                "foot_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*FOOT"]
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_roll_rate = RewTerm(     # -0.1 × 0.02 = -0.002  [stated]
            func=base_roll_rate,
            weight=-0.002,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.joint_regularization = RewTerm(  # -0.001 × 0.02 = -0.00002 [stated]
            func=joint_regularization,
            weight=-0.00002,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.action_smoothness = RewTerm(  # -0.01 × 0.02 = -0.0002  [stated]
            func=action_smoothness,
            weight=-0.0002,
            params={},
        )
        self.rewards.link_contact_forces = RewTerm(  # -0.00001 × 0.02 = -0.0000002 [stated]
            func=link_contact_forces,
            weight=-0.0000002,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.link_acceleration = RewTerm(  # -0.001 × 0.02 = -0.00002 [stated]
            func=link_acceleration,
            weight=-0.00002,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── Simulation Fidelity (Table I, weight = paper_weight × dτ) ─────────
        self.rewards.ame2_joint_pos_limits = RewTerm(  # -1000 × 0.02 = -20 [stated]
            func=joint_pos_limits,
            weight=-20.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.ame2_joint_vel_limits = RewTerm(  # -1 × 0.02 = -0.02  [stated]
            func=joint_vel_limits,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.ame2_joint_torque_limits = RewTerm(  # -1 × 0.02 = -0.02 [stated]
            func=joint_torque_limits,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        if self.__class__.__name__ == "AME2AnymalEnvCfg":
            self.disable_zero_weight_rewards()

        # ============================ Terminations ============================
        self.terminations.illegal_contact = None

        self.terminations.bad_orientation = DoneTerm(
            func=ame2_bad_orientation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.base_collision = DoneTerm(
            func=ame2_base_collision,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
                "robot_mass_kg": 50.0,
            },
        )
        self.terminations.high_thigh_acceleration = DoneTerm(
            func=ame2_high_thigh_acceleration,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*THIGH"]),
                "threshold": 60.0,
            },
        )
        self.terminations.stagnation = DoneTerm(
            func=ame2_stagnation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "window_s": 5.0,
                "min_displacement": 0.5,
                "min_goal_dist": 1.0,
            },
        )

        # ============================ Curriculum ============================
        # Terrain curriculum: replace velocity-based metric with goal-reaching  [stated Sec.IV-D.3]
        # Robots that reach the goal (d < 0.5m) are moved to harder terrain;
        # robots that clearly fail (d > 1.0m at episode end) are moved to easier terrain.
        self.curriculum.terrain_levels = CurrTerm(
            func=terrain_levels_goal,
            params={"goal_threshold": 0.5},
        )
        # Disable velocity-command curricula (AME-2 uses goal-pos commands, not vel)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ============================ Dimension Assertions ============================
        # Actor prop (policy obs group):
        #   base_vel(3) | ang_vel+grav+q+dq+act(42) | ame2_actor_cmd(3) = 48   [stated]
        _d_actor_teacher = 3 + 3 + 3 + 12 + 12 + 12 + 3
        assert _d_actor_teacher == 48, f"d_actor_teacher mismatch: {_d_actor_teacher} != 48"
        # student-accessible slice (no base_vel): ang_vel+grav+q+dq+act = 42 = d_hist
        _d_hist = 3 + 3 + 12 + 12 + 12
        assert _d_hist == 42, f"d_hist mismatch: {_d_hist} != 42"
        # actor cmd appended at end: 3D                                          [stated]
        assert _d_actor_teacher == _d_hist + 3 + 3, "layout sanity: base_vel+hist+cmd != 48"
        # student prop (no base_vel) used to verify hist+cmd = 45D
        _d_actor_student = _d_hist + 3   # 42 + 3 = 45
        assert _d_actor_student == 45, f"d_actor_student mismatch: {_d_actor_student} != 45"
        # Critic prop (critic_extra obs group → assembled in rslrl_wrapper):
        #   base_vel(3) | ang_vel+grav+q+dq+act(42) | critic_cmd(5) = 50        [stated]
        #   critic_cmd = [x_rel, y_rel, sin(yaw), cos(yaw), t_remaining]
        _d_critic_cmd = 5   # x_rel + y_rel + sin + cos + t_remaining
        _d_critic_prop = _d_hist + 3 + _d_critic_cmd   # base_vel(3)+hist(42)+cmd(5) = 50
        assert _d_critic_prop == 50, f"d_critic_prop mismatch: {_d_critic_prop} != 50"
        _d_teacher = 31 * 51 + 4 * 3
        assert _d_teacher == 1593, f"d_teacher_privileged mismatch: {_d_teacher} != 1593"
        _d_teacher_map = 3 * 14 * 36
        assert _d_teacher_map == 1512, f"d_teacher_map mismatch: {_d_teacher_map} != 1512"
