# Copyright (c) 2024-2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""AME-2 curriculum functions.

Reference: Zhang et al., "AME-2", arXiv:2601.08485, Sec. IV-D.3.

Two training curricula:
  1. terrain_levels_goal — move to harder terrain based on goal-reaching success.
  2. heading_curriculum_frac stored in env.extras for external control (train_ame2.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_goal(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_threshold: float = 0.5,
) -> torch.Tensor:
    """Terrain curriculum based on goal-reaching success (Sec. IV-D.3).

    At each episode end the robot is assigned to:
      - harder terrain  if it reached within ``goal_threshold`` m of the goal.
      - easier terrain  if it failed to reach the goal AND its final distance
                        exceeded 2× the threshold (clear failure case).
      - same level otherwise (ambiguous outcome — e.g. episode timed out near goal).

    The goal command is read from the "goal_pos" command manager key which
    outputs [x_base, y_base, z_base, heading_base].  The x/y components are
    the goal position expressed in the robot base frame, so their L2-norm
    equals the current distance-to-goal.

    .. note::
        Only usable with ``terrain_type="generator"`` and
        ``terrain_generator.curriculum=True``.

    Returns:
        Mean terrain level across all active environments (scalar) — logged by
        Isaac Lab's curriculum manager.
    """
    asset: Articulation = env.scene[asset_cfg.name]  # noqa: F841 — needed for SceneEntityCfg resolution
    terrain: TerrainImporter = env.scene.terrain

    # goal command in base frame: [x_b, y_b, z_b, heading_b]
    cmd = env.command_manager.get_command("goal_pos")  # (N, 4)
    d_xy = torch.norm(cmd[env_ids, :2], dim=1)        # (K,) distance to goal

    move_up   = d_xy < goal_threshold                      # reached goal → harder
    move_down = (d_xy > 2.0 * goal_threshold) & ~move_up  # clear failure → easier

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
