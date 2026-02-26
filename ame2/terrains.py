"""
AME-2 terrain configuration aligned to real Isaac Lab TerrainGeneratorCfg API.

Training terrains — 12 types from paper Sec.V-B with [stated] proportions:
  Dense    25%  (Rough 5% + StairDown 5% + StairUp 5% + Boxes 5% + Obstacles 5%)
  Climbing 30%  (ClimbingUp 20% + ClimbingDown 5% + Consecutive 5%)
  Sparse   45%  (Gap 5% + Pallets 5% + Stones 30% + Beam 5%)

Note: Paper Sec.V-B labels Sparse as "50%" but sub-terrain proportions sum to 45%.
We use explicit sub-terrain proportions (total 100%).

Test terrains — 4 evaluation-only environments from paper Sec.V-C / Fig. 5.
  These are NOT used during training. curriculum=False, fixed high difficulty.
  TEST1_IRREGULAR_STONES_CFG    — ③ Sparse
  TEST2_STONES_PALLET_CLIMBING_CFG — ②+③
  TEST3_PARKOUR_OBSTACLES_CFG   — ①+②+③
  TEST4_DEBRIS_CFG              — ①+②+③

Isaac Lab terrain class mapping:
  Rough          -> HfRandomUniformTerrainCfg
  StairDown      -> MeshInvertedPyramidStairsTerrainCfg  (inverted = descending)
  StairUp        -> MeshPyramidStairsTerrainCfg          (ascending)
  Boxes          -> MeshRandomGridTerrainCfg              (random grid of boxes)
  Obstacles      -> HfDiscreteObstaclesTerrainCfg         (discrete obstacles on HF)
  ClimbingUp     -> MeshPitTerrainCfg                     (climb out of pit)
  ClimbingDown   -> MeshBoxTerrainCfg                     (climb down from platform)
  Consecutive    -> MeshPitTerrainCfg(double_pit=True)    (two-level nested pit)
  Gap            -> MeshGapTerrainCfg                     (gap around platform)
  Pallets        -> MeshRailsTerrainCfg                   (parallel rail beams)
  Stones         -> HfSteppingStonesTerrainCfg            (stepping stones)
  Beam           -> MeshStarTerrainCfg                    (star bars connecting platform)

Terrains without a perfect Isaac Lab match are noted with # [approx] comments.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# ------------------------------------------------------------------ #
# Sub-terrain configs for ANYmal-D (default)
# ------------------------------------------------------------------ #

# --- Dense 25% ---

ROUGH_TERRAIN = terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.05,
    noise_range=(-0.20, 0.20),   # [stated] max at difficulty=1
    noise_step=0.01,
    border_width=0.25,
)

STAIR_DOWN_TERRAIN = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
    proportion=0.05,
    step_height_range=(0.05, 0.40),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    holes=False,
)

STAIR_UP_TERRAIN = terrain_gen.MeshPyramidStairsTerrainCfg(
    proportion=0.05,
    step_height_range=(0.05, 0.40),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
    holes=False,
)

BOXES_TERRAIN = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=0.05,
    grid_width=0.45,
    grid_height_range=(0.05, 0.40),    # [stated] ANYmal-D
    platform_width=2.0,
)

OBSTACLES_TERRAIN = terrain_gen.HfDiscreteObstaclesTerrainCfg(
    proportion=0.05,
    obstacle_height_mode="choice",
    obstacle_width_range=(0.25, 0.75),
    obstacle_height_range=(0.05, 0.20),
    num_obstacles=40,
    platform_width=2.0,
)

# --- Climbing 30% ---

CLIMBING_UP_TERRAIN = terrain_gen.MeshPitTerrainCfg(
    proportion=0.20,
    pit_depth_range=(0.10, 1.00),   # [stated] ANYmal-D
    platform_width=3.0,
    double_pit=False,
)

CLIMBING_DOWN_TERRAIN = terrain_gen.MeshBoxTerrainCfg(
    proportion=0.05,
    box_height_range=(0.20, 1.00),  # [stated] ANYmal-D
    platform_width=3.0,
    double_box=False,
)

CLIMBING_CONSECUTIVE_TERRAIN = terrain_gen.MeshPitTerrainCfg(
    proportion=0.05,
    pit_depth_range=(0.05, 0.50),   # [stated] ring1 ANYmal-D
    platform_width=2.5,
    double_pit=True,
)

# --- Sparse 45% ---

GAP_TERRAIN = terrain_gen.MeshGapTerrainCfg(
    proportion=0.05,
    gap_width_range=(0.10, 1.10),   # [stated] ANYmal-D
    platform_width=3.0,
)

PALLETS_TERRAIN = terrain_gen.MeshRailsTerrainCfg(
    proportion=0.05,
    rail_thickness_range=(0.16, 0.40),
    rail_height_range=(0.05, 0.30),
    platform_width=2.0,
)

STONES_TERRAIN = terrain_gen.HfSteppingStonesTerrainCfg(
    proportion=0.30,
    stone_height_max=0.25,
    stone_width_range=(0.15, 0.60),
    stone_distance_range=(0.05, 0.20),
    holes_depth=-1.50,              # [stated]
    platform_width=2.0,
)

BEAM_TERRAIN = terrain_gen.MeshStarTerrainCfg(
    proportion=0.05,
    num_bars=4,
    bar_width_range=(0.18, 0.90),   # [stated]
    bar_height_range=(0.05, 0.20),
    platform_width=2.0,
)


# ================================================================== #
# AME2_TERRAIN_CFG — ANYmal-D (default)
# ================================================================== #

AME2_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "rough": ROUGH_TERRAIN,
        "stair_down": STAIR_DOWN_TERRAIN,
        "stair_up": STAIR_UP_TERRAIN,
        "boxes": BOXES_TERRAIN,
        "obstacles": OBSTACLES_TERRAIN,
        "climbing_up": CLIMBING_UP_TERRAIN,
        "climbing_down": CLIMBING_DOWN_TERRAIN,
        "climbing_consecutive": CLIMBING_CONSECUTIVE_TERRAIN,
        "gap": GAP_TERRAIN,
        "pallets": PALLETS_TERRAIN,
        "stones": STONES_TERRAIN,
        "beam": BEAM_TERRAIN,
    },
)


# ================================================================== #
# TRON1_TERRAIN_CFG — smaller robot, reduced difficulty ceilings
# ================================================================== #

TRON1_ROUGH = terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.05, noise_range=(-0.15, 0.15), noise_step=0.01, border_width=0.25,
)
TRON1_BOXES = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=0.05, grid_width=0.45, grid_height_range=(0.05, 0.30), platform_width=2.0,
)
TRON1_CLIMBING_UP = terrain_gen.MeshPitTerrainCfg(
    proportion=0.20, pit_depth_range=(0.10, 0.48), platform_width=3.0, double_pit=False,
)
TRON1_CLIMBING_DOWN = terrain_gen.MeshBoxTerrainCfg(
    proportion=0.05, box_height_range=(0.20, 0.88), platform_width=3.0, double_box=False,
)
TRON1_CLIMBING_CONSECUTIVE = terrain_gen.MeshPitTerrainCfg(
    proportion=0.05, pit_depth_range=(0.05, 0.30), platform_width=2.5, double_pit=True,
)
TRON1_GAP = terrain_gen.MeshGapTerrainCfg(
    proportion=0.05, gap_width_range=(0.10, 0.60), platform_width=3.0,
)
TRON1_PALLETS = terrain_gen.MeshRailsTerrainCfg(
    proportion=0.05, rail_thickness_range=(0.16, 0.40), rail_height_range=(0.05, 0.20), platform_width=2.0,
)

TRON1_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "rough": TRON1_ROUGH,
        "stair_down": STAIR_DOWN_TERRAIN,
        "stair_up": STAIR_UP_TERRAIN,
        "boxes": TRON1_BOXES,
        "obstacles": OBSTACLES_TERRAIN,
        "climbing_up": TRON1_CLIMBING_UP,
        "climbing_down": TRON1_CLIMBING_DOWN,
        "climbing_consecutive": TRON1_CLIMBING_CONSECUTIVE,
        "gap": TRON1_GAP,
        "pallets": TRON1_PALLETS,
        "stones": STONES_TERRAIN,
        "beam": BEAM_TERRAIN,
    },
)


# ================================================================== #
# Terrain metadata for goal sampling / category logic
# ================================================================== #

GOAL_SAMPLING_ANYWHERE = "anywhere"
GOAL_SAMPLING_OPPOSITE = "opposite_end"

CATEGORY_DENSE = "dense"
CATEGORY_CLIMBING = "climbing"
CATEGORY_SPARSE = "sparse"

TERRAIN_META = {
    "rough":                {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "stair_down":           {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "stair_up":             {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "boxes":                {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "obstacles":            {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "climbing_up":          {"category": CATEGORY_CLIMBING, "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "climbing_down":        {"category": CATEGORY_CLIMBING, "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "climbing_consecutive": {"category": CATEGORY_CLIMBING, "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "gap":                  {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "pallets":              {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "stones":               {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "beam":                 {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_OPPOSITE},
}


# ================================================================== #
# Curriculum helper
# ================================================================== #

def get_terrain_at_curriculum_level(cfg: TerrainGeneratorCfg, level: float) -> dict[str, dict]:
    """Return interpolated terrain parameters at a given curriculum level (0=easiest, 1=hardest)."""
    level = max(0.0, min(1.0, level))
    result: dict[str, dict] = {}

    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    for name, sub_cfg in cfg.sub_terrains.items():
        params: dict = {"difficulty": level}
        if isinstance(sub_cfg, terrain_gen.HfRandomUniformTerrainCfg):
            nr = sub_cfg.noise_range
            params["noise_range"] = (-_lerp(0.0, abs(nr[1]), level), _lerp(0.0, abs(nr[1]), level))
        elif isinstance(sub_cfg, (terrain_gen.MeshPyramidStairsTerrainCfg,
                                  terrain_gen.MeshInvertedPyramidStairsTerrainCfg)):
            sh = sub_cfg.step_height_range
            params["step_height"] = _lerp(sh[0], sh[1], level)
        elif isinstance(sub_cfg, terrain_gen.MeshRandomGridTerrainCfg):
            gh = sub_cfg.grid_height_range
            params["grid_height"] = _lerp(gh[0], gh[1], level)
        elif isinstance(sub_cfg, terrain_gen.HfDiscreteObstaclesTerrainCfg):
            oh = sub_cfg.obstacle_height_range
            params["obstacle_height"] = _lerp(oh[0], oh[1], level)
        elif isinstance(sub_cfg, terrain_gen.MeshPitTerrainCfg):
            pd = sub_cfg.pit_depth_range
            params["pit_depth"] = _lerp(pd[0], pd[1], level)
        elif isinstance(sub_cfg, terrain_gen.MeshBoxTerrainCfg):
            bh = sub_cfg.box_height_range
            params["box_height"] = _lerp(bh[0], bh[1], level)
        elif isinstance(sub_cfg, terrain_gen.MeshGapTerrainCfg):
            gw = sub_cfg.gap_width_range
            params["gap_width"] = _lerp(gw[0], gw[1], level)
        elif isinstance(sub_cfg, terrain_gen.MeshRailsTerrainCfg):
            params["rail_thickness"] = _lerp(*sub_cfg.rail_thickness_range, level)
            params["rail_height"] = _lerp(*sub_cfg.rail_height_range, level)
        elif isinstance(sub_cfg, terrain_gen.HfSteppingStonesTerrainCfg):
            params["stone_width"] = _lerp(*sub_cfg.stone_width_range, level)
            params["stone_distance"] = _lerp(*sub_cfg.stone_distance_range, level)
        elif isinstance(sub_cfg, terrain_gen.MeshStarTerrainCfg):
            params["bar_width"] = _lerp(*sub_cfg.bar_width_range, level)
            params["bar_height"] = _lerp(*sub_cfg.bar_height_range, level)
        result[name] = params

    return result


# ================================================================== #
# TEST TERRAIN CONFIGS
# Evaluation-only — NOT used during training (paper Sec.V-C / Fig. 5).
# All test configs have curriculum=False and a fixed high difficulty range
# to measure out-of-distribution generalisation.
# ================================================================== #

# ------------------------------------------------------------------
# Test 1: Irregular Stones  [③ Sparse]
# A dense field of irregular-height rocks — harder and more chaotic
# than the training Stones terrain.
# [approx] high-difficulty stepping stones + very rough height field.
# ------------------------------------------------------------------

TEST1_IRREGULAR_STONES_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,           # TEST — fixed difficulty, no curriculum
    difficulty_range=(0.8, 1.0),
    use_cache=False,
    sub_terrains={
        "rough_base": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(-0.30, 0.30),   # [approx] harder than training Rough (±0.20)
            noise_step=0.01,
            border_width=0.25,
        ),
        "irregular_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.70,
            stone_height_max=0.40,             # [approx] taller than training Stones (0.25m)
            stone_width_range=(0.08, 0.60),    # wide variance → more irregular shapes
            stone_distance_range=(0.05, 0.35), # larger possible gaps than training
            holes_depth=-2.0,
            platform_width=2.0,
        ),
    },
)

# ------------------------------------------------------------------
# Test 2: Stones-Pallet-Climbing  [②+③]
# Mixed sparse + climbing: stepping stones underfoot, pallets to
# balance on, and pits/platforms to climb.
# ------------------------------------------------------------------

TEST2_STONES_PALLET_CLIMBING_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,           # TEST — fixed difficulty, no curriculum
    difficulty_range=(0.5, 1.0),
    use_cache=False,
    sub_terrains={
        "stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.40,
            stone_height_max=0.25,
            stone_width_range=(0.15, 0.60),
            stone_distance_range=(0.05, 0.20),
            holes_depth=-1.50,
            platform_width=2.0,
        ),
        "pallets": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.30,
            rail_thickness_range=(0.16, 0.40),
            rail_height_range=(0.05, 0.30),
            platform_width=2.0,
        ),
        "climbing_up": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.10, 1.00),
            platform_width=3.0,
            double_pit=False,
        ),
        "climbing_down": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.10,
            box_height_range=(0.20, 1.00),
            platform_width=3.0,
            double_box=False,
        ),
    },
)

# ------------------------------------------------------------------
# Test 3: Parkour w/ Obstacles  [①+②+③]
# Full-spectrum course combining all three difficulty categories:
# discrete obstacles and boxes (Dense ①), climbing pits and gaps
# (Climbing ②), stepping stones (Sparse ③).
# ------------------------------------------------------------------

TEST3_PARKOUR_OBSTACLES_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,           # TEST — fixed difficulty, no curriculum
    difficulty_range=(0.5, 1.0),
    use_cache=False,
    sub_terrains={
        # ① Dense
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=40,
            platform_width=2.0,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.40),
            platform_width=2.0,
        ),
        # ② Climbing
        "climbing_up": terrain_gen.MeshPitTerrainCfg(
            proportion=0.25,
            pit_depth_range=(0.10, 1.00),
            platform_width=3.0,
            double_pit=False,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.10, 1.10),
            platform_width=3.0,
        ),
        # ③ Sparse
        "stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.25,
            stone_height_max=0.25,
            stone_width_range=(0.15, 0.60),
            stone_distance_range=(0.05, 0.20),
            holes_depth=-1.50,
            platform_width=2.0,
        ),
    },
)

# ------------------------------------------------------------------
# Test 4: Debris  [①+②+③]
# Randomly scattered debris of all sizes and shapes, mixing dense
# obstacles, irregular blocks, stepping stones, and climbing features.
# [approx] all obstacle types at higher density than training.
# ------------------------------------------------------------------

TEST4_DEBRIS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,           # TEST — fixed difficulty, no curriculum
    difficulty_range=(0.6, 1.0),
    use_cache=False,
    sub_terrains={
        # ① Dense — rough surface as the base
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(-0.20, 0.20),
            noise_step=0.01,
            border_width=0.25,
        ),
        # ① Dense — more numerous and smaller obstacles than training
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.10, 0.60),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=60,               # [approx] denser than training (40)
            platform_width=2.0,
        ),
        # ① Dense — smaller grid → more scattered debris chunks
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.35,               # [approx] tighter grid than training (0.45)
            grid_height_range=(0.05, 0.35),
            platform_width=2.0,
        ),
        # ③ Sparse — irregular stepping stones
        "stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.20,
            stone_height_max=0.25,
            stone_width_range=(0.10, 0.50),
            stone_distance_range=(0.05, 0.25),
            holes_depth=-1.50,
            platform_width=2.0,
        ),
        # ② Climbing — debris piles to climb over
        "climbing_up": terrain_gen.MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.10, 1.00),
            platform_width=3.0,
            double_pit=False,
        ),
    },
)


# ================================================================== #
# Test terrain metadata (goal sampling hints, same schema as TERRAIN_META)
# ================================================================== #

TERRAIN_META_TEST: dict[str, dict] = {
    # Test 1
    "rough_base":       {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "irregular_stones": {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    # Test 2 / 3 / 4 (keys shared across multiple test configs)
    "stones":           {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "pallets":          {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "climbing_up":      {"category": CATEGORY_CLIMBING, "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "climbing_down":    {"category": CATEGORY_CLIMBING, "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "obstacles":        {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "boxes":            {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
    "gap":              {"category": CATEGORY_SPARSE,   "goal_sampling": GOAL_SAMPLING_OPPOSITE},
    "rough":            {"category": CATEGORY_DENSE,    "goal_sampling": GOAL_SAMPLING_ANYWHERE},
}
