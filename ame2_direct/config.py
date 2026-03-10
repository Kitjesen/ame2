"""AME-2 Direct Environment Config — V43 Paper-Faithful.

Based on anymal_c_env_cfg.py pattern from IsaacLab 0.46.x:
  - observation_space / action_space are int (SpaceType)
  - state_space for critic
  - prim_path format: /World/envs/env_.*/Robot

V43: Match AME-2 paper exactly (Table I + Table VI + Sec.IV-D).
     No simplifications, no bootstrap, no custom curriculum.
     Only deviation: 2048 envs (RTX 3090 limit vs paper's 4800).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG, ANYDRIVE_3_SIMPLE_ACTUATOR_CFG  # isort: skip



@configclass
class AME2DirectEnvCfg(DirectRLEnvCfg):
    """AME-2 direct environment configuration (ANYmal-D).

    observation_space = 48   (teacher actor prop)           [stated]
    action_space      = 12   (joint position targets)       [stated]
    state_space       = 0    (asymmetric critic in network)
    """

    # ── Env ─────────────────────────────────────────────────────────────────
    episode_length_s:  float = 20.0    # Paper: avg goal dist 4m, needs time
    decimation:        int   = 4       # 50 Hz control                [stated]
    action_scale:      float = 0.5     # joint targets = default + scale*action

    # SpaceType (int → flat Box)
    observation_space: int = 48        # teacher actor prop dim       [stated]
    action_space:      int = 12        # 12-DoF joint position targets
    state_space:       int = 0         # asymmetric critic handled in network
    # critic_prop: 45(base) + 5(critic_cmd) + 5(nav_extra) = 55D

    # ── Simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,                 # 200 Hz physics               [stated]
        render_interval=4,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**21,  # 2M patches — avoids overflow with 2048+ envs
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── Scene ────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4800,                  # [stated Table VI]
        env_spacing=3.0,
        replicate_physics=False,
    )

    # ── Terrain ──────────────────────────────────────────────────────────────
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        debug_vis=False,
    )

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG},
    )

    # ── Contact Sensor ────────────────────────────────────────────────────────
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ── Height Scanners ───────────────────────────────────────────────────────
    # MappingNet scanner: 31×51 @ 4cm                                 [stated]
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.04,
            size=[1.20, 2.00],  # (31-1)*0.04 × (51-1)*0.04
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # GT policy map scanner: 14×36 @ 8cm                              [stated]
    height_scanner_policy: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.08,
            size=[1.04, 2.80],  # (14-1)*0.08 × (36-1)*0.08
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ── Goal Command ─────────────────────────────────────────────────────────
    fallen_start_ratio:  float = 0.0
    fallen_roll_range:   tuple = (-3.14, 3.14)
    fallen_pitch_range:  tuple = (-0.5, 0.5)

    # Paper Sec.IV-D.3: "average starting distance is 4m"
    # Goals sampled anywhere on terrain; we use annulus [2, 6]m.
    goal_pos_range_min:  float = 2.0     # minimum goal distance
    goal_pos_range_max:  float = 6.0     # maximum goal distance (no curriculum)

    # Paper Eq.(4): v_min = 0.3 m/s, v_max = 2.0 m/s
    moving_to_goal_v_min: float = 0.3    # Paper value, no bootstrap

    # ── Reward Weights (V43 Paper-Faithful, Table I) ──────────────────────────
    # All weights are RAW values; dt=0.02 scaling applied in env.py __init__.
    # "We set all weights to integer powers of 10" — Paper Sec.IV-D.1
    #
    # === Task Rewards (Paper Table I) ===
    w_position_tracking:    float = 100.0    # Eq.(1): terminal, last 4s
    w_heading_tracking:     float = 50.0     # Eq.(3): heading at goal, last 2s
    w_moving_to_goal:       float = 5.0      # Eq.(4): binary walk signal
    w_standing_at_goal:     float = 5.0      # Eq.(5): stand still at goal
    #
    # === NOT in paper — disabled ===
    w_bias_goal:            float = 0.0      # V42 custom, not in paper
    w_anti_stall:           float = 0.0      # V42 custom, not in paper
    w_upward:               float = 0.0
    w_goal_coarse:          float = 0.0
    w_goal_fine:            float = 0.0
    w_vel_toward_goal:      float = 0.0
    w_position_approach:    float = 0.0
    w_base_height:          float = 0.0
    w_feet_air_time:        float = 0.0
    w_anti_stagnation:      float = 0.0
    w_lin_vel_z_l2:         float = 0.0      # not in paper Table I
    #
    # === Regularization (Paper Table I) ===
    w_early_termination:    float = -500.0   # Paper: -10/dτ = -10/0.02 = -500
    w_undesired_contacts:   float = -1.0     # Paper: -1 per event
    w_ang_vel_xy_l2:        float = -0.1     # Paper: Base Roll Rate -0.1
    w_joint_reg_l2:         float = -0.001   # Paper: Joint Regularization -0.001
    w_action_rate_l2:       float = -0.01    # Paper: Action Smoothness -0.01
    w_link_contact_forces:  float = -0.00001 # Paper: Link Contact Forces -0.00001
    w_link_acceleration:    float = -0.001   # Paper: Link Acceleration -0.001
    #
    # === Simulation Fidelity (Paper Table I) ===
    w_joint_pos_limits:     float = -1000.0  # Paper: -1000
    w_joint_vel_limits:     float = -1.0     # Paper: -1
    w_joint_torque_limits:  float = -1.0     # Paper: -1
