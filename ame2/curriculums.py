# Copyright (c) 2024-2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""AME-2 curriculum functions.

Reference: Zhang et al., "AME-2", arXiv:2601.08485, Sec. IV-D.3.

Two training curricula:
  1. terrain_levels_goal — move to harder terrain based on EMA goal-reaching
     success rate (matches paper Sec. IV-D.3: promote if EMA > 0.5, demote
     if EMA ≤ 0.5 AND d_xy > 4 m).
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


# ---------------------------------------------------------------------------
# Module-level EMA state — persists across curriculum calls within a run.
# Key: id(env)  →  Tensor(num_envs,) of per-env EMA success rates in [0, 1].
# ---------------------------------------------------------------------------
_ema_state: dict[int, torch.Tensor] = {}


def terrain_levels_goal(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_threshold: float = 0.5,       # success: reached within this distance (m)
    ema_alpha: float = 0.1,            # EMA update rate  (≈10-episode window)
    promote_threshold: float = 0.5,    # advance level if EMA success rate > this
    demote_distance: float = 4.0,      # demote only if ALSO d_xy > this (m)
) -> torch.Tensor:
    """Terrain curriculum based on EMA goal-reaching success (Sec. IV-D.3).

    At each episode end the EMA success rate is updated for every finishing env:

    .. code-block::

        ema[i]  ←  α · success[i]  +  (1 − α) · ema[i]
        success  =  1  if  d_xy < goal_threshold  else  0

    Level transitions (paper, Sec. IV-D.3):

    * **Promote** : ``ema[i] > promote_threshold``  (default 0.5)
    * **Demote**  : ``ema[i] ≤ promote_threshold``  AND  ``d_xy > demote_distance``
      (must also be a *clear* failure — far from goal — to avoid penalising
      near-miss episodes on hard terrain)
    * **Stay**    : otherwise

    The EMA buffer is lazily initialised to zero on first call (conservative
    start — env will need to succeed before advancing).

    .. note::
        Only usable with ``terrain_type="generator"`` and
        ``terrain_generator.curriculum=True``.

    Args:
        env: The RL environment.
        env_ids: Indices of environments that just ended their episode.
        asset_cfg: Unused (kept for API compatibility with Isaac Lab).
        goal_threshold: Distance (metres) to goal that counts as success.
        ema_alpha: EMA update step size.  ``α=0.1`` corresponds to roughly
            a 10-episode rolling window.
        promote_threshold: EMA success rate above which the env advances to
            a harder terrain level.
        demote_distance: Minimum ``d_xy`` required for a demotion to occur;
            prevents penalising near-misses on hard terrain.

    Returns:
        Mean terrain level across all environments (scalar) — logged by
        Isaac Lab's curriculum manager.
    """
    global _ema_state

    asset: Articulation = env.scene[asset_cfg.name]  # noqa: F841
    terrain: TerrainImporter = env.scene.terrain

    # Lazy-init per-env EMA buffer (zeros → conservative start)
    key = id(env)
    if key not in _ema_state:
        _ema_state[key] = torch.zeros(env.num_envs, device=env.device)
    ema = _ema_state[key]

    # goal command in base frame: [x_b, y_b, z_b, heading_b]
    cmd = env.command_manager.get_command("goal_pos")  # (N, 4)
    d_xy = torch.norm(cmd[env_ids, :2], dim=1)         # (K,) distance to goal

    # Success: reached within goal_threshold
    success = (d_xy < goal_threshold).float()

    # EMA update for this batch of finished environments
    ema[env_ids] = ema_alpha * success + (1.0 - ema_alpha) * ema[env_ids]

    # Paper Sec. IV-D.3 level transitions
    move_up   = ema[env_ids] > promote_threshold
    move_down = (ema[env_ids] <= promote_threshold) & (d_xy > demote_distance)

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
