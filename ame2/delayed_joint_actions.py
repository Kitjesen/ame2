# Copyright (c) 2024-2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Delayed joint position action term for AME-2 actuation delay randomization.

Reference: Zhang et al., "AME-2", arXiv:2601.08485, Appendix B.
    "Actuation delay is sampled uniformly from U[0, 0.02] s."

This wraps Isaac Lab's standard JointPositionActionCfg with a per-environment
random delay buffer.  At each control step the action is stored in a ring
buffer and the actually-applied action is read from a randomly delayed slot.

The delay is re-sampled at each episode reset (one value per environment,
constant within an episode) to match the paper's domain randomization protocol.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch

from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.assets import Articulation
from isaaclab.utils import configclass


class DelayedJointPositionAction(ActionTerm):
    """Joint position action with per-environment actuation delay (Appendix B).

    Each environment gets a random integer delay d ∈ [0, max_delay_steps] at
    episode reset.  The action applied to the simulator at step t is the action
    that was commanded at step (t − d).  A FIFO ring buffer stores the last
    ``max_delay_steps + 1`` actions for each environment.

    This simulates realistic actuator communication latency without modifying
    the physics timestep or the policy control frequency.
    """

    cfg: "DelayedJointPositionActionCfg"

    def __init__(self, cfg: "DelayedJointPositionActionCfg", env):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        # Convert max delay in seconds to physics steps
        physics_dt = env.sim.get_physics_dt() if hasattr(env.sim, "get_physics_dt") else env.step_dt / env.cfg.decimation
        self._max_delay_steps = max(0, math.ceil(cfg.max_delay_s / physics_dt))

        buf_len = self._max_delay_steps + 1
        B = env.num_envs
        n_joints = self._asset.num_joints

        # Ring buffer: (B, buf_len, n_joints)
        default_pos = self._asset.data.default_joint_pos.clone()
        self._buf = default_pos.unsqueeze(1).expand(B, buf_len, n_joints).clone()
        self._write_idx = torch.zeros(B, dtype=torch.long, device=env.device)

        # Per-env delay in steps: sampled at reset
        self._delay = torch.zeros(B, dtype=torch.long, device=env.device)

        self._buf_len = buf_len
        self._device = env.device

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-sample delays and clear buffers for reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self._buf.shape[0], device=self._device)
        if len(env_ids) == 0:
            return

        # Re-sample delay for each reset environment: U[0, max_delay_steps]
        self._delay[env_ids] = torch.randint(
            0, self._max_delay_steps + 1, (len(env_ids),), device=self._device
        )

        # Reset ring buffer to default joint positions
        default_pos = self._asset.data.default_joint_pos[env_ids]
        self._buf[env_ids] = default_pos.unsqueeze(1).expand(
            len(env_ids), self._buf_len, -1
        )
        self._write_idx[env_ids] = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store the commanded action and retrieve the delayed one."""
        # Scale and offset (same as standard JointPositionAction)
        target = actions * self.cfg.scale
        if self.cfg.use_default_offset:
            target = target + self._asset.data.default_joint_pos

        # Write into ring buffer
        B = target.shape[0]
        b_idx = torch.arange(B, device=self._device)
        self._buf[b_idx, self._write_idx, :] = target

        # Read from delayed slot
        read_idx = (self._write_idx - self._delay) % self._buf_len
        self._processed_actions = self._buf[b_idx, read_idx, :]

        # Advance write pointer
        self._write_idx = (self._write_idx + 1) % self._buf_len

    def apply_actions(self) -> None:
        """Apply the delayed action to the articulation."""
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )


@configclass
class DelayedJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`DelayedJointPositionAction`.

    Attributes:
        max_delay_s: Maximum actuation delay in seconds.  Per-env delay is
            sampled uniformly from [0, max_delay_s] at each episode reset.
            Paper (Appendix B): 0.02 s.
        scale: Action scaling factor.
        use_default_offset: If True, add default joint positions to actions.
    """

    class_type: type = DelayedJointPositionAction

    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    scale: float = 0.5
    use_default_offset: bool = True
    preserve_order: bool = True
    clip: dict | None = None
    max_delay_s: float = 0.02  # [stated] Appendix B: U[0, 0.02] s
