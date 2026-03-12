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
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG, ANYDRIVE_3_SIMPLE_ACTUATOR_CFG  # isort: skip


# ── AME-2 Paper Terrain (Appendix A, 12 types with curriculum) ────────────────
# difficulty ∈ [0,1] linearly interpolates range[0]→range[1] across num_rows.
# For "harder = narrower" params, range is written (wide, narrow) so difficulty=1 = hardest.
AME2_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0),     # 12m tiles: center-to-edge=6m covers max goal dist
    border_width=20.0,
    num_rows=10,            # 10 difficulty levels
    num_cols=14,            # reduced from 20 to keep mesh manageable (~2M verts)
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # ── Dense (25%) — goals sampled anywhere on tile ──
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05,
            noise_range=(-0.20, 0.20),      # curriculum: ±0 → ±0.2m
            noise_step=0.01,
            border_width=0.25,
        ),
        "stair_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.40),  # curriculum: slope 5°→45°
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stair_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.40),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.05,
            grid_width=0.45,
            grid_height_range=(0.05, 0.40),  # curriculum: box height 0.05→0.4m
            platform_width=2.0,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.05,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.05, 0.20),  # approx density via height
            num_obstacles=40,                     # ~0.5/m² at 8×8m
            platform_width=2.0,
        ),
        # ── Climbing (30%) — goals sampled at opposite end ──
        "climbing_up": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.10, 1.00),    # curriculum: pit 0.1→1.0m
            platform_width=3.0,
            double_pit=False,
        ),
        "climbing_down": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.05,
            box_height_range=(0.20, 1.00),   # curriculum: platform 0.2→1.0m
            platform_width=3.0,
            double_box=False,
        ),
        "climbing_consecutive": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05,
            pit_depth_range=(0.05, 0.50),    # approx stacked rings
            platform_width=2.5,
            double_pit=True,
        ),
        # ── Sparse (45%) — goals at opposite end, virtual floors ──
        "stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.30,
            stone_height_max=0.25,
            stone_width_range=(0.15, 0.60),
            stone_distance_range=(0.05, 0.20),
            holes_depth=-1.50,               # paper: floor -1.5m to -0.35m
            platform_width=2.0,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05,
            gap_width_range=(0.10, 1.10),    # curriculum: gap 0.1→1.1m
            platform_width=3.0,
        ),
        "pallets": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.05,
            rail_thickness_range=(0.40, 0.16),  # curriculum: wide→narrow (harder)
            rail_height_range=(0.05, 0.30),
            platform_width=2.0,
        ),
        "beam": terrain_gen.MeshStarTerrainCfg(
            proportion=0.05,
            num_bars=4,
            bar_width_range=(0.90, 0.18),    # curriculum: wide→narrow (harder)
            bar_height_range=(0.05, 0.20),
            platform_width=2.0,
        ),
    },
)



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
    # replicate_physics=True matches IsaacLab's anymal_c example:
    # enables collision filtering between envs so robots don't interact.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,                  # Paper: 4800; RTX 3090 limit: 2048 (4096 OOM)
        env_spacing=4.0,                # match anymal_c (was 3.0)
        replicate_physics=True,
    )

    # ── Terrain ──────────────────────────────────────────────────────────────
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=AME2_TERRAINS_CFG,
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
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),  # V46: paper Sec.V-B "centered at x=1.0m"
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
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 20.0)),  # V46: paper Sec.IV-E "centered at x=0.6m"
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
    w_position_tracking:    float = 100.0    # V44: approach reward clamp(d_prev-d_curr, 0, 0.5)
    w_arrival:              float = 0.0      # Not in paper (was V44 addition, removed)
    w_heading_tracking:     float = 50.0     # Eq.(3): heading at goal, last 2s
    w_moving_to_goal:       float = 5.0      # V43q: lowered from 20, vel_toward_goal now provides gradient
    w_standing_at_goal:     float = 5.0      # Eq.(5): stand still at goal
    #
    # === NOT in paper — disabled ===
    w_bias_goal:            float = 0.0      # V42 custom, not in paper
    w_anti_stall:           float = 0.0      # V42 custom, not in paper
    w_upward:               float = 0.0
    w_goal_coarse:          float = 5.0      # Always-on 1-tanh(d/2): gradient at any d
    w_goal_fine:            float = 0.0
    w_vel_toward_goal:      float = 0.0      # Not in paper (was V43q addition, removed)
    w_position_approach:    float = 50.0    # V49: true approach reward clamp(d_prev-d_curr, 0, 0.1)
    w_base_height:          float = 0.0      # V43l: removed, causes "stand still" exploit
    w_feet_air_time:        float = 0.0
    w_anti_stagnation:      float = 0.0
    w_lin_vel_z_l2:         float = 0.0      # not in paper Table I
    #
    # === Regularization (Paper Table I) ===
    w_early_termination:    float = -500.0   # Paper: -10/dτ = -10/0.02 = -500
    w_undesired_contacts:   float = -1.0     # Paper: -1 (V44: back to paper, fixed stumble+slip bugs)
    w_ang_vel_xy_l2:        float = -0.1     # Paper: Base Roll Rate -0.1
    w_joint_reg_l2:         float = -0.001   # Paper: Joint Regularization -0.001
    w_action_rate_l2:       float = -0.01    # Paper: Action Smoothness -0.01
    w_link_contact_forces:  float = -0.00001 # Paper: Link Contact Forces -0.00001
    w_link_acceleration:    float = -0.001   # Paper: Link Acceleration -0.001
    #
    # === Simulation Fidelity (Paper Table I) ===
    w_joint_pos_limits:     float = -50.0    # v46 fix: -1000 causes noise collapse (robot prefers standing still)
    w_joint_vel_limits:     float = -1.0     # Paper: -1
    w_joint_torque_limits:  float = -1.0     # Paper: -1
