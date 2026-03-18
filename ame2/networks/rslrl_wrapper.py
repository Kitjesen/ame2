"""
RSL-RL compatible wrapper for AME-2 Actor-Critic.

Aligned to the real RSL-RL ActorCritic interface (rsl_rl.modules.actor_critic)
from the rsl_rl library bundled with IsaacLab.

Real RSL-RL interface contract (verified against source):
    Constructor:
        __init__(obs: TensorDict, obs_groups: dict, num_actions: int, ...)
    Required attributes:
        is_recurrent: bool
        actor: nn.Module         (for optimizer param groups)
        critic: nn.Module        (for optimizer param groups)
        distribution: Normal     (cached, set by act() via _update_distribution())
        obs_groups: dict         (maps "policy"/"critic" -> list of obs group names)
    Required methods:
        act(obs: TensorDict, **kw) -> actions          stochastic
        act_inference(obs: TensorDict) -> actions       deterministic
        evaluate(obs: TensorDict, **kw) -> values (B,1) value estimate
        get_actions_log_prob(actions) -> log_prob (B,)
        reset(dones)                                    no-op for non-recurrent
        update_normalization(obs: TensorDict)           called by PPO each step
    Properties:
        action_mean -> distribution.mean
        action_std  -> distribution.stddev
        entropy     -> distribution.entropy().sum(dim=-1)

Key design notes:
    - PPO calls act(obs) and evaluate(obs) with the SAME obs TensorDict.
      The policy uses obs_groups to extract actor vs critic observations.
    - distribution is Normal (per-dimension), NOT Independent.
    - log_prob and entropy are summed over action dim explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Import AME-2 model classes (relative import — inside robot_lab package).
# ---------------------------------------------------------------------------
from .ame2_model import (
    PolicyConfig,
    MappingConfig,
    AME2Policy,
    AsymmetricCritic,
    StudentLoss,
    WTAMapFusion,
    MappingNet,
    ANYMAL_D_WTA_KWARGS,
)


# Binary contact detection threshold (Newtons) — must match rewards.CONTACT_FORCE_THRESHOLD.
CONTACT_FORCE_THRESHOLD: float = 1.0

# ---------------------------------------------------------------------------
# Left-Right Symmetry Augmentation (Sec. IV-B)
# ---------------------------------------------------------------------------
# ANYmal-D joint ordering (Isaac Lab / Isaac Sim USD traversal — confirmed from
# isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/anymal.py):
#
#   ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',   ← idx 0,1,2,3
#    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',   ← idx 4,5,6,7
#    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']   ← idx 8,9,10,11
#
# Leg indices: LF=(0,4,8)  LH=(1,5,9)  RF=(2,6,10)  RH=(3,7,11)
# Ordering within each type group is [LF, LH, RF, RH] — NOT [LF, RF, LH, RH].
#
# L-R flip: LF↔RF  and  LH↔RH
#   HAA(0-3): LF(0)↔RF(2), LH(1)↔RH(3)  → perm [2,3,0,1]
#   HFE(4-7): LF(4)↔RF(6), LH(5)↔RH(7)  → perm [6,7,4,5]
#   KFE(8-11):LF(8)↔RF(10),LH(9)↔RH(11) → perm [10,11,8,9]
_LR_JOINT_PERM   = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
# HAA joints negate under L-R mirror (abduction axis flips sign); HFE/KFE stay positive.
# With interleaved ordering, HAA joints occupy the first 4 indices (0-3).
_LR_JOINT_SIGN   = [-1., -1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1.]
# All-link contact order: base(0), thigh×4(1-4), shank×4(5-8), foot×4(9-12)
# Within each 4-element group, find_bodies(".*THIGH" etc.) returns bodies in USD
# traversal order, which matches the joint interleaving: [LF, LH, RF, RH].
# L-R mirror: swap LF↔RF (pos 0↔2) and LH↔RH (pos 1↔3) in each group.
#   base(0): unchanged
#   thigh(1-4): [LF=1,LH=2,RF=3,RH=4] → [RF=3,RH=4,LF=1,LH=2]
#   shank(5-8): [LF=5,LH=6,RF=7,RH=8] → [RF=7,RH=8,LF=5,LH=6]
#   foot(9-12): [LF=9,LH=10,RF=11,RH=12]→[RF=11,RH=12,LF=9,LH=10]
#
# RUNTIME VERIFICATION: env.py prints [ContactOrder] lines at startup showing
# actual body_names so this assumption can be confirmed against the live USD.
_LR_CONTACT_PERM = [0, 3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10]


def _flip_lr(
    map_feat: torch.Tensor,
    prop:     torch.Tensor,
    contact:  torch.Tensor,
    is_teacher: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply left-right mirror to critic inputs (Sec. IV-B symmetry augmentation).

    Args:
        map_feat:   (B, C, H, W)  — flip the W (left-right) axis.
        prop:       (B, d_prop)   — negate lateral dims, permute joint order.
        contact:    (B, 13)       — swap left/right contacts (all links).
        is_teacher: True  → teacher prop layout (48): base_vel(3)|ang_vel(3)|grav(3)|q(12)|dq(12)|act(12)|cmd(3)
                    False → student prop layout (45):             ang_vel(3)|grav(3)|q(12)|dq(12)|act(12)|cmd(3)

    Returns:
        (map_f, prop_f, contact_f) — mirrored tensors, same device/dtype.
    """
    # Map: lateral flip (H = y/lateral axis in robot frame)
    # Coordinate convention (from WTAMapFusion):
    #   dim -2 (H, rows)  → y (lateral)   e.g. policy_h=14 covers ±0.52 m
    #   dim -1 (W, cols)  → x (forward)   e.g. policy_w=36 covers −0.8..2.0 m
    # L-R mirror flips y → −y = flip rows (dim -2), NOT columns (dim -1).
    # BUG FIX: was torch.flip(dims=[-1]) which flipped forward axis instead.
    map_f = torch.flip(map_feat, dims=[-2])
    # After row-flip, y_rel values still carry the original sign (e.g. a cell at +0.5m
    # body-y is now placed at the -0.5m row position, but its stored y_rel is still +0.5).
    # Negate channel 1 (y_rel / lateral) to restore physical consistency.
    # Map channel layout: [x_rel, y_rel, z_rel] (teacher, 3ch) or [x, y, z, u] (student, 4ch).
    # z_rel / u are NOT negated (vertical and uncertainty are even under L-R).
    if map_f.shape[1] >= 2:
        map_f = map_f.clone()          # detach from flip's storage
        map_f[:, 1] = -map_f[:, 1]    # y_rel (lateral) — odd under L-R flip

    # Proprioception: negate lateral-sensitive dims, permute + sign-flip joints
    prop_f = prop.clone()
    sign   = torch.tensor(_LR_JOINT_SIGN, device=prop.device, dtype=prop.dtype)

    if is_teacher:
        # Common base: base_vel[0:3] | ang_vel[3:6] | grav[6:9] | q[9:21] | dq[21:33] | act[33:45]
        # ang_vel: negate [3]=roll(wx) and [5]=yaw(wz); leave [4]=pitch(wy) (symmetric under L-R)
        prop_f[:, 1]  *= -1.0   # base_lin_vel y (lateral — odd under L-R)
        prop_f[:, 3]  *= -1.0   # ang_vel [3] = roll rate (wx) — odd under L-R
        prop_f[:, 5]  *= -1.0   # ang_vel [5] = yaw rate  (wz) — odd under L-R
        prop_f[:, 7]  *= -1.0   # gravity y (lateral — odd under L-R)
        for sl in (slice(9, 21), slice(21, 33), slice(33, 45)):
            prop_f[:, sl] = prop_f[:, sl][:, _LR_JOINT_PERM] * sign

        d = prop.shape[-1]
        if d == 48:
            # Actor prop — cmd layout: [45]=d_xy, [46]=sin(yaw), [47]=cos(yaw)
            prop_f[:, 46] *= -1.0   # sin(yaw_rel) — odd
            # [45]=d_xy: no flip (positive scalar); [47]=cos: even, no flip
        elif d == 50:
            # Critic prop — cmd layout: [45]=x_rel, [46]=y_rel, [47]=sin(yaw), [48]=cos(yaw), [49]=t_rem
            prop_f[:, 46] *= -1.0   # y_rel — lateral, odd under L-R
            prop_f[:, 47] *= -1.0   # sin(yaw_rel) — odd
            # [45]=x_rel: forward, symmetric, no flip
            # [48]=cos(yaw): even, no flip
            # [49]=t_remaining: scalar, no flip
        elif d == 55:
            # Extended critic prop: 50D above + nav_extra(5D) at [50-54]
            # nav_extra = [v_toward_goal, d_progress_rate, heading_align_rate, vel_w_x, vel_w_y]
            prop_f[:, 46] *= -1.0   # y_rel — lateral, odd under L-R
            prop_f[:, 47] *= -1.0   # sin(yaw_rel) — odd
            prop_f[:, 54] *= -1.0   # world_vel_y — lateral, odd under L-R
            # [50]=v_toward_goal: scalar dot-product, even; [51]=d_progress: even
            # [52]=heading_align_rate: even (abs angle magnitude); [53]=vel_w_x: even
        else:
            raise ValueError(
                f"_flip_lr is_teacher=True: unexpected prop dim {d}, expected 48 (actor) or 50/55 (critic)"
            )
    else:
        # Offsets: ang_vel[0:3] | grav[3:6] | q[6:18] | dq[18:30] | act[30:42] | cmd[42:45]
        # cmd layout: [42]=d_xy (positive, no flip), [43]=sin(yaw_rel) (odd→negate), [44]=cos(yaw_rel) (even→keep)
        # ang_vel: negate [0]=roll(wx) and [2]=yaw(wz); leave [1]=pitch(wy) unchanged (symmetric under L-R)
        prop_f[:, 0]  *= -1.0   # ang_vel [0] = roll rate (wx) — odd under L-R
        prop_f[:, 2]  *= -1.0   # ang_vel [2] = yaw rate  (wz) — odd under L-R
        prop_f[:, 4]  *= -1.0   # gravity y (lateral — odd under L-R)
        for sl in (slice(6, 18), slice(18, 30), slice(30, 42)):
            prop_f[:, sl] = prop_f[:, sl][:, _LR_JOINT_PERM] * sign
        prop_f[:, 43] *= -1.0   # sin(yaw_rel) — odd function, flips sign
        # prop_f[:, 44] unchanged — cos(yaw_rel) is even, no sign change under L-R mirror

    # Contact: LF↔RF, LH↔RH
    contact_f = contact[:, _LR_CONTACT_PERM]

    return map_f, prop_f, contact_f


def _shift_map_batch(
    maps: torch.Tensor,
    di: torch.Tensor,
    dj: torch.Tensor,
) -> torch.Tensor:
    """Per-env shift of (B, C, H, W) via bilinear grid_sample.

    Used to simulate mapping drift (localization error) in domain randomization
    (Sec.IV-D.3 stated).  Each environment gets an independent integer-cell offset
    (di[b], dj[b]).  Border cells are replicated at the edges (no wrap-around).

    Args:
        maps: (B, C, H, W)  — map to shift (float tensor).
        di:   (B,) int tensor — row offsets in cells (+down).
        dj:   (B,) int tensor — col offsets in cells (+right).

    Returns:
        (B, C, H, W) shifted maps, same dtype/device as ``maps``.
    """
    B, _, H, W = maps.shape
    # Build identity grid (H, W, 2) in normalised coords ∈ [−1, 1]
    ys = torch.linspace(-1.0, 1.0, H, device=maps.device, dtype=maps.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=maps.device, dtype=maps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')     # (H, W)

    # (B, H, W, 2): channel 0 = x (W-axis), channel 1 = y (H-axis)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).clone()

    # Normalised shift: one cell maps to 2/(dim−1) in grid_sample coords
    grid[:, :, :, 0] += (2.0 * dj.to(maps.dtype) / max(W - 1, 1)).view(B, 1, 1)
    grid[:, :, :, 1] += (2.0 * di.to(maps.dtype) / max(H - 1, 1)).view(B, 1, 1)

    return F.grid_sample(maps, grid, mode='bilinear',
                         align_corners=True, padding_mode='border')


# ===================================================================
# RSL-RL compatible AME-2 Actor-Critic
# ===================================================================

class AME2ActorCritic(nn.Module):
    """RSL-RL compatible Actor-Critic wrapping AME-2 policy and Asymmetric
    MoE Critic.

    Matches the real rsl_rl.modules.ActorCritic interface:
        - Constructor: (obs, obs_groups, num_actions, ...)
        - act/evaluate both receive the same TensorDict obs
        - obs_groups maps "policy" and "critic" to lists of obs group names
        - Distribution is Normal (not Independent); log_prob summed over dims

    Observation groups expected from the environment TensorDict:

        "prop"         : (B, 48)            actor prop  (base_vel+hist+actor_cmd_3D)
        "critic_prop"  : (B, 50)            critic prop (base_vel+hist+critic_cmd_5D)
        "map"          : (B, C, 14, 36)     map (C=3 teacher GT, C=4 student neural)
        "history"      : (B, 20, 42)        prop history without base_vel/cmd (student only)
        "commands"     : (B, 3)             actor cmd [clip(d_xy,2), sin(yaw), cos(yaw)] (student only)
        "map_teacher"  : (B, 3, 14, 36)     GT map for AsymmetricCritic
        "contact"      : (B, 13)            all-link binary contact states for critic [stated Sec.IV-B]

    obs_groups example:
        {
            "policy": ["prop", "map"],                     # teacher
            "critic": ["prop", "map_teacher", "contact"],
        }
    or for student:
        {
            "policy": ["map", "history", "commands"],
            "critic": ["prop", "map_teacher", "contact"],
        }
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs,       # TensorDict — initial observation from env
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        ame2_cfg: Optional[PolicyConfig] = None,
        is_student: bool = False,
        critic_kwargs: Optional[dict] = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # Accept and ignore standard RSL-RL kwargs we don't use
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ):
        """
        Args:
            obs:            TensorDict from env.get_observations() — used to
                            verify observation shapes. Matches real RSL-RL API.
            obs_groups:     Maps "policy" -> [obs group names], "critic" -> [...].
            num_actions:    Must equal cfg.num_joints (12).
            ame2_cfg:       PolicyConfig. Defaults to ANYmal-D.
            is_student:     If True, builds student policy with LSIO encoder.
            critic_kwargs:  Extra kwargs for AsymmetricCritic.
            init_noise_std: Initial action noise std (default 1.0).
            noise_std_type: "scalar" or "log" (how std is parameterised).
        """
        if kwargs:
            print(
                "AME2ActorCritic.__init__ got unexpected arguments, "
                "which will be ignored: " + str(list(kwargs.keys()))
            )
        super().__init__()

        if ame2_cfg is None:
            ame2_cfg = PolicyConfig()
        self.cfg = ame2_cfg
        self.is_student = is_student
        self.obs_groups = obs_groups

        assert num_actions == ame2_cfg.num_joints, (
            f"num_actions ({num_actions}) must match cfg.num_joints ({ame2_cfg.num_joints})"
        )

        # ---- Actor (AME-2 policy network) ----
        self.actor = AME2Policy(ame2_cfg, is_student=is_student)

        # ---- Critic (asymmetric MoE, always uses teacher-level inputs) ----
        _ck = critic_kwargs or {}
        self.critic = AsymmetricCritic(ame2_cfg, **_ck)

        # ---- Learnable action std ----
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown noise_std_type: {noise_std_type}. "
                "Should be 'scalar' or 'log'"
            )

        # ---- Cached distribution (set by act / _update_distribution) ----
        self.distribution: Optional[Normal] = None

        # Disable args validation for speedup (matches real RSL-RL)
        Normal.set_default_validate_args(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_std(self) -> torch.Tensor:
        """Get current action std from parameters.

        Floor of 0.2 prevents noise collapse (stand-still local optima).
        Ceiling of 0.5 prevents entropy bonus from pushing std up (V52c fix):
        with std>0.5 the return variance is too high for the critic to learn,
        causing value loss to explode 100x.
        """
        if self.noise_std_type == "scalar":
            return torch.clamp(self.std, min=0.2, max=0.5)
        else:
            return torch.clamp(torch.exp(self.log_std), min=0.2, max=0.5)

    def _forward_actor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run actor policy on obs TensorDict.

        Returns: (action_mean, map_emb, prop_emb)
        """
        if self.is_student:
            return self.actor(
                obs["map"],
                prop_hist=obs["history"],
                commands=obs["commands"],
            )
        else:
            return self.actor(obs["map"], prop=obs["prop"])

    def _update_distribution(self, obs) -> None:
        """Compute action mean from actor network, build Normal distribution."""
        mean, _, _ = self._forward_actor(obs)
        std = self._get_std().expand_as(mean)
        self.distribution = Normal(mean, std)

    # ------------------------------------------------------------------
    # RSL-RL interface methods
    # ------------------------------------------------------------------

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        """No-op for non-recurrent policy. Required by RSL-RL."""
        pass

    def act(self, obs, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Stochastic action sampling (rollout phase).

        Args:
            obs: TensorDict with observation groups.
        Returns:
            actions: (B, num_joints)
        """
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        """Deterministic inference -- returns distribution mean.

        Args:
            obs: TensorDict with observation groups.
        Returns:
            actions: (B, num_joints)
        """
        mean, _, _ = self._forward_actor(obs)
        return mean

    def evaluate(self, obs, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Value estimate from the Asymmetric MoE Critic with L-R symmetry augmentation.

        Augmentation (Sec. IV-B): the critic is applied to both the original
        observations and their left-right mirror.  Averaging the two estimates
        enforces V(s) = V(flip(s)), doubling effective critic sample efficiency
        without touching the actor.

        The critic receives the FULL command (50D critic_prop) which includes
        exact [x_rel, y_rel] and remaining episode time — not available to actor.
        [stated Sec.IV-E.3]

        Args:
            obs: TensorDict with keys "critic_prop", "map_teacher", "contact".
        Returns:
            value: (B, 1)  — mean of original and mirrored value estimates.
        """
        map_t   = obs["map_teacher"]
        prop    = obs["critic_prop"]   # (B, 50) — full critic command
        contact = obs["contact"]

        v_orig = self.critic(map_t, prop, contact)

        map_f, prop_f, cont_f = _flip_lr(
            map_t, prop, contact, is_teacher=True  # critic always uses teacher layout (50D)
        )
        v_flip = self.critic(map_f, prop_f, cont_f)

        return 0.5 * (v_orig + v_flip)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of actions under the cached distribution.

        Returns: (B,) -- sum of per-dimension log probs.
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs) -> None:
        """Update observation normalizers. No-op for AME-2 (no normalization)."""
        pass

    # ------------------------------------------------------------------
    # Properties (match real RSL-RL exactly)
    # ------------------------------------------------------------------

    @property
    def action_mean(self) -> torch.Tensor:
        """Mean of the cached distribution."""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Std of the cached distribution."""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the cached distribution (B,). Summed over action dim."""
        return self.distribution.entropy().sum(dim=-1)

    # ------------------------------------------------------------------
    # State dict compatibility
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load parameters. Returns True to indicate training can resume."""
        super().load_state_dict(state_dict, strict=strict)
        return True


# ===================================================================
# Student stage Actor-Critic with distillation loss support
# ===================================================================

class AME2StudentActorCritic(AME2ActorCritic):
    """Student training stage with distillation loss.

    Extends AME2ActorCritic with ``act_and_embed()`` and
    ``compute_student_loss()`` for the two-phase student training
    schedule (Sec. IV-C, Table VI).

    Training schedule:
        Phase 1 (iter 0..4999):   pure distillation, PPO disabled, LR=0.001
        Phase 2 (iter 5000..39999): PPO + distillation, adaptive LR
    """

    PHASE1_ITERS: int = 5000
    PHASE1_LR:   float = 1e-3   # [stated] Table VI: "LR When Surrogate Loss Disabled"
    KL_TARGET:   float = 0.01   # [stated] Table VI: "Desired KL Divergence"

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        ame2_cfg: Optional[PolicyConfig] = None,
        critic_kwargs: Optional[dict] = None,
        lam_dist: float = 0.02,
        lam_repr: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            ame2_cfg=ame2_cfg,
            is_student=True,
            critic_kwargs=critic_kwargs,
            **kwargs,
        )
        self.student_loss_fn = StudentLoss(lam_dist=lam_dist, lam_repr=lam_repr)
        self._iteration = 0

    @property
    def in_phase1(self) -> bool:
        return self._iteration < self.PHASE1_ITERS

    def set_iteration(self, iteration: int) -> None:
        self._iteration = iteration

    def step_iteration(
        self,
        optimizer: torch.optim.Optimizer,
        it: int,
    ) -> bool:
        """Advance iteration counter and handle Phase 1 → 2 transition.

        Call once per training iteration (after each ``alg.update()`` call).
        Detects the Phase 1 → 2 boundary (PPO surrogate enabled).

        LR scheduling (Table VI):
          Phase 1: fixed at PHASE1_LR=1e-3 (enforced by train_ame2.py)
          Phase 2: starts at 1e-3 with adaptive KL-based scheduling
                   (desired_kl=0.01, same as teacher PPO)

        Args:
            optimizer: Optimizer managing this policy's parameters.
            it:        0-based current iteration index (``runner.current_learning_iteration``).

        Returns:
            True if the Phase 1 → 2 transition happened at this call.
        """
        was_phase1 = self.in_phase1
        self.set_iteration(it)
        transitioned = was_phase1 and not self.in_phase1
        if transitioned:
            # LR stays at PHASE1_LR (1e-3) which is also the initial adaptive LR.
            # Adaptive scheduling (desired_kl=0.01) takes effect in Phase 2.
            print(
                f"[AME2 Student] Phase 1 -> 2 at iter {it}: "
                f"PPO surrogate enabled, LR={self.PHASE1_LR} "
                f"(adaptive scheduling, desired_kl={self.KL_TARGET})"
            )
        return transitioned

    def act_and_embed(self, obs):
        """Like act(), but also returns embeddings for distillation.

        Returns: (actions, map_emb, prop_emb)
        """
        mean, map_emb, prop_emb = self._forward_actor(obs)
        std = self._get_std().expand_as(mean)
        self.distribution = Normal(mean, std)
        actions = self.distribution.sample()
        return actions, map_emb, prop_emb

    def compute_student_loss(
        self,
        teacher_map_emb: torch.Tensor,
        teacher_prop_emb: torch.Tensor,
        student_map_emb: torch.Tensor,
        student_prop_emb: torch.Tensor,
        teacher_actions: torch.Tensor,
        student_actions: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_old: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined student training loss.

        Handles phase 1/2 schedule: PPO zeroed in phase 1.

        Returns: dict with "total", "ppo", "dist", "repr" scalars.
        """
        import torch.nn.functional as F

        assert self.distribution is not None, (
            "Call act() or act_and_embed() before compute_student_loss()"
        )
        log_probs_new = self.distribution.log_prob(student_actions).sum(dim=-1)
        ratio = (log_probs_new - log_probs_old).exp()
        clip_eps = 0.2
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()

        if self.in_phase1:
            ppo_loss = torch.zeros_like(ppo_loss)

        l_dist = F.mse_loss(student_actions, teacher_actions.detach())
        l_repr = F.mse_loss(student_map_emb, teacher_map_emb.detach())

        total = (
            ppo_loss
            + self.student_loss_fn.lam_dist * l_dist
            + self.student_loss_fn.lam_repr * l_repr
        )

        return {
            "total": total,
            "ppo": ppo_loss.detach(),
            "dist": l_dist.detach(),
            "repr": l_repr.detach(),
        }


# ===================================================================
# WTA Map Manager -- multi-environment batch processing
# ===================================================================

class WTAMapManager:
    """Manages N parallel WTAMapFusion instances for batched environments.

    Each environment maintains its own episode-relative global map.
    Handles update/crop/reset across all envs.
    """

    def __init__(
        self,
        num_envs: int,
        wta_kwargs: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.device = device
        _kw = wta_kwargs or dict(ANYMAL_D_WTA_KWARGS)

        self.wta = WTAMapFusion(B=num_envs, **_kw).to(device)
        self.wta.reset()

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset global maps for terminated episodes."""
        if env_ids is None or len(env_ids) == 0:
            self.wta.reset()
        else:
            self.wta.reset(batch_idx=env_ids)

    def _normalize_pose(self, poses: torch.Tensor) -> torch.Tensor:
        """Convert (N,3) or (N,4) poses to (N,3) [x,y,yaw]."""
        if poses.shape[-1] == 4:
            return torch.stack([poses[:, 0], poses[:, 1], poses[:, 3]], dim=-1)
        return poses

    def update(
        self,
        elev_local: torch.Tensor,
        log_var_local: torch.Tensor,
        poses: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ):
        """Ingest MappingNet output into global maps."""
        poses_3 = self._normalize_pose(poses)

        if env_ids is not None and len(env_ids) < self.num_envs:
            for i, eid in enumerate(env_ids):
                e = int(eid)
                self._update_single(
                    e,
                    elev_local[i:i + 1],
                    log_var_local[i:i + 1],
                    poses_3[i:i + 1],
                )
        else:
            self.wta.update(elev_local, log_var_local, poses_3)

    def _update_single(
        self,
        env_idx: int,
        elev_local: torch.Tensor,
        log_var_local: torch.Tensor,
        poses: torch.Tensor,
    ):
        """Update a single environment's global map via probabilistic WTA (Eq. 6-8).

        FIX: Previously used deterministic min-variance instead of the paper's
        probabilistic WTA logic. Now matches WTAMapFusion.update() exactly:
          Eq.6: sigma2_eff = max(sigma2_t, 0.5 * sigma2_prior)
          Validity: sigma2_eff < 1.5 * sigma2_prior OR sigma2_eff < 0.04
          Eq.7: p_win = prec_new / (prec_new + prec_prior)
          Eq.8: stochastic overwrite where xi < p_win
        """
        var_local = log_var_local.exp()
        elev_flat = elev_local.reshape(1, -1)
        var_flat = var_local.reshape(1, -1)

        pts = self.wta._local_pts.unsqueeze(0)
        pts_w = self.wta._to_world(pts, poses)
        gi, gj = self.wta._world_to_idx(pts_w)

        lin = gi[0] * self.wta.global_w + gj[0]
        gvar = self.wta.global_var[env_idx, 0].view(-1)
        gelev = self.wta.global_elev[env_idx, 0].view(-1)

        sigma2_prior = gvar[lin]                                    # (N,)
        sigma2_t = var_flat[0]                                      # (N,)

        # Eq. 6: effective measurement variance, lower-bounded by half the prior
        sigma2_eff = torch.max(sigma2_t, 0.5 * sigma2_prior)       # (N,)

        # Validity check (paper Sec. V-A)
        valid = (sigma2_eff < 1.5 * sigma2_prior) | (sigma2_eff < 0.04)
        valid_idx = valid.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return

        v_lin = lin[valid_idx]
        v_sigma2_eff = sigma2_eff[valid_idx]
        v_sigma2_pr = sigma2_prior[valid_idx]
        v_elev = elev_flat[0][valid_idx]

        # Eq. 7: p_win = precision_new / (precision_new + precision_prior)
        prec_new = 1.0 / (v_sigma2_eff + 1e-8)
        prec_prior = 1.0 / (v_sigma2_pr + 1e-8)
        p_win = prec_new / (prec_new + prec_prior)

        # Eq. 8: stochastic overwrite
        xi = torch.rand_like(p_win)
        wins = xi < p_win
        win_idx = wins.nonzero(as_tuple=True)[0]
        if win_idx.numel() == 0:
            return

        w_lin = v_lin[win_idx]
        w_var = v_sigma2_eff[win_idx]
        w_elev = v_elev[win_idx]

        # Sort descending by var so minimum-var wins last when indices collide
        order = w_var.argsort(descending=True)
        gvar.scatter_(0, w_lin[order], w_var[order])
        gelev.scatter_(0, w_lin[order], w_elev[order])

    def get_policy_maps(
        self,
        poses: torch.Tensor,
        gt_map_flat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Crop policy-sized windows from all environments' global maps.

        Args:
            poses:       Robot poses (N, 3) [x, y, yaw] or (N, 4) [x, y, z, yaw].
            gt_map_flat: Ground-truth teacher map from ``height_scanner_policy``
                         (the ``teacher_map`` obs group in AME2AnymalEnvCfg),
                         shape (N, 3*H*W).  Reshaped internally to (N, 3, H, W).
                         If None, falls back to ``student_map[:, :3]`` — this is
                         ONLY valid for offline / standalone testing; never use
                         None during real teacher or student training.

        Returns:
            dict with:
                "student_map"  (N, 4, H, W) — neural WTA map crop
                "teacher_map"  (N, 3, H, W) — GT RayCaster map (or fallback)
        """
        poses_3 = self._normalize_pose(poses)
        student_map = self.wta.crop(poses_3)       # (N, 4, policy_h, policy_w)
        N = student_map.shape[0]
        # Use explicit policy grid dims from WTA config, NOT runtime shape from student_map.
        # This prevents silent data reinterpretation if the WTA and GT scanner dims diverge.
        ph, pw = self.wta.policy_h, self.wta.policy_w

        if gt_map_flat is not None:
            # Reshape (N, 3*ph*pw) → (N, 3, ph, pw) from GT RayCaster
            teacher_map = gt_map_flat.reshape(N, 3, ph, pw)
        else:
            # Fallback for offline testing only — NOT ground-truth
            teacher_map = student_map[:, :3]

        return {
            "student_map": student_map,
            "teacher_map": teacher_map,
        }


# ===================================================================
# Factory function
# ===================================================================

def make_ame2_rslrl_agent(
    obs,
    obs_groups: dict[str, list[str]],
    num_actions: int,
    stage: str = "teacher",
    ame2_cfg: Optional[PolicyConfig] = None,
    device: str = "cuda",
    **kwargs,
) -> AME2ActorCritic:
    """Factory function matching RSL-RL's construction pattern.

    Called as: actor_critic_class(obs, obs_groups, num_actions, **policy_cfg)

    Args:
        obs:         TensorDict from env.get_observations().
        obs_groups:  Observation group mapping.
        num_actions: Number of action dims (12).
        stage:       "teacher" or "student".
        ame2_cfg:    PolicyConfig. Defaults to ANYmal-D.
        device:      Target device.

    Returns:
        AME2ActorCritic or AME2StudentActorCritic on device.
    """
    if ame2_cfg is None:
        ame2_cfg = PolicyConfig()

    if stage == "teacher":
        agent = AME2ActorCritic(
            obs, obs_groups, num_actions,
            ame2_cfg=ame2_cfg,
            is_student=False,
            **kwargs,
        )
    elif stage == "student":
        agent = AME2StudentActorCritic(
            obs, obs_groups, num_actions,
            ame2_cfg=ame2_cfg,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown stage '{stage}', expected 'teacher' or 'student'")

    return agent.to(device)


# ===================================================================
# AME2MapEnvWrapper — hooks the mapping pipeline into every env step
# ===================================================================

class AME2MapEnvWrapper:
    """Wraps a gymnasium Isaac Lab env to inject the AME-2 mapping pipeline.

    Replaces ``RslRlVecEnvWrapper`` in the training loop. Implements the
    same 5-value step/reset interface that RSL-RL's OnPolicyRunner expects:

        reset() → (obs_td, obs_td, None, None, extras)
        step(a) → (obs_td, obs_td, rewards, dones, extras)

    Pipeline executed at every control step:
        raw_scan  (B, 1, 31, 51)  ← teacher_privileged obs group
        → MappingNet              → (elev, log_var)
        → WTAMapFusion.update()   (student map update)
        → get_policy_maps(gt_map_flat=...)  ← teacher_map obs group (GT)
        → build TensorDict        for AME2ActorCritic

    Also maintains the per-env LSIO history buffer (B, T, d_hist).

    Contact (manager-based path):  foot_contact_forces (B,12) → per-foot magnitude (B,4).
    NOTE: The manager-based path only produces 4D foot contacts, NOT the full 13D
    all-link contact used by the direct env.  When Phase 2 manager-based training is
    activated, this path must be updated to match the direct env's 13D contact output.
    See TODO at the contact_4 computation below.
    History:  teacher prop layout = base_vel(3)|hist(42)|cmd(3)
              student prop layout = hist(42)|cmd(3)
    """

    # Must match AME2AnymalEnvCfg height_scanner dimensions
    SCAN_H: int = 31
    SCAN_W: int = 51

    def __init__(
        self,
        gym_env,
        mapping_net: MappingNet,
        wta_manager: "WTAMapManager",
        policy_cfg: PolicyConfig,
        *,
        is_student: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            gym_env:      Raw gymnasium env from ``gym.make("AME2-ANYmal-D-v0")``.
                          Do NOT wrap with RslRlVecEnvWrapper first.
            mapping_net:  Pretrained (or frozen) MappingNet instance.
            wta_manager:  WTAMapManager for all envs.
            policy_cfg:   PolicyConfig (ANYmal-D or TRON1).
            is_student:   If True, builds student obs (adds history + commands).
            device:       Torch device string.
        """
        self._gym_env = gym_env
        # Raw ManagerBasedRLEnv — obs_buf is populated after each step/reset
        self._il_env = gym_env.unwrapped
        self.mapping_net = mapping_net
        self.wta_manager = wta_manager
        self.cfg = policy_cfg
        self.is_student = is_student
        self._device = device

        self.num_envs: int = self._il_env.num_envs
        self.num_actions: int = policy_cfg.num_joints
        self.max_episode_length: int = self._il_env.max_episode_length

        # RSL-RL runner reads these to init its internal book-keeping.
        # AME2ActorCritic reads TensorDict keys directly, so the exact values
        # don't affect network shapes — set to 1 as a safe placeholder.
        self.num_obs: int = 1
        self.num_privileged_obs: Optional[int] = None

        # LSIO history ring buffer: (B, T, d_hist)
        B, T, d = self.num_envs, policy_cfg.prop_history, policy_cfg.d_hist
        self.history: torch.Tensor = torch.zeros(B, T, d, device=device)

        # Perception noise curriculum (0 = clean, 1 = full noise)
        self._scan_noise_scale: float = 0.0

        # Initial heading curriculum: fraction ∈ [0, 1].
        # 0 = robot always faces goal at reset; 1 = full random heading.
        self._heading_curriculum_frac: float = 1.0  # start unconstrained; set via set_heading_curriculum()

        # ── Student depth scan degradation (Sec.IV-D.3, Appendix B) ────────
        self._scan_dropout_rate: float = 0.0   # fraction of depth pixels zeroed (missing)
        self._artifact_rate: float = 0.0        # fraction of depth pixels with spike artifacts
        self._artifact_std: float = 0.0         # spike artifact magnitude (std of Gaussian)

        # ── Map domain randomization (Sec.IV-D.3, Appendix B) ───────────────
        self._partial_map_fraction: float = 0.0  # fraction of envs with local-only access
        self._map_drop_fraction: float = 0.0     # per-cell corruption probability
        self._drift_max_m: float = 0.0            # max map drift in metres (Appendix B: 0.03 m)
        self._corrupt_var_min: float = 1.0        # min variance for corrupted cells (m²)

        # Per-env mask: True = full global WTA map, False = local-only.
        # Re-sampled at episode reset.  Starts with full map access for all envs.
        self._partial_map_mask: torch.Tensor = torch.ones(
            self.num_envs, dtype=torch.bool, device=device
        )

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Perception noise curriculum (Sec. IV-D3)
    # ------------------------------------------------------------------

    #: Maximum additive uniform noise magnitude applied to raw depth scans.
    #: [stated] Appendix B: "zero-mean uniform noise ... 0.05 m for map observations".
    #: Ramped linearly during Phase 1 scan noise curriculum (Sec. IV-D.3).
    SCAN_NOISE_MAX: float = 0.05       # [stated] Appendix B

    def set_scan_noise_scale(self, scale: float) -> None:
        """Set perception noise curriculum scale for the raw depth scan.

        Called once per training iteration.  During teacher training the scale
        ramps linearly from 0.0 (clean) to 1.0 (full noise) over the first 20 %
        of iterations (Sec. IV-D3 stated).  After that it stays at 1.0.

        The noise is applied *before* MappingNet so the teacher learns to
        produce reliable maps despite realistic sensor imperfections.

        Args:
            scale: Value in [0, 1].  0 = no added noise, 1 = SCAN_NOISE_MAX.
        """
        self._scan_noise_scale = float(max(0.0, min(1.0, scale)))

    def set_heading_curriculum(self, frac: float) -> None:
        """Set initial heading curriculum fraction (Sec. IV-D.3).

        During the first 20 % of training the robot is initialised facing
        (approximately) toward the goal, making early exploration easier.
        As training progresses the allowed heading deviation expands linearly
        until it covers the full circle [-π, π].

        Implementation: after each episode reset we override the goal command's
        heading component for the *just-reset* environments by clamping the
        sampled heading to ±(frac × π).  At frac = 1.0 (full training) the
        clamp covers ±π which is the full range — no constraint.

        Called once per training iteration from train_ame2.py via the same
        ramp schedule used for perception noise.

        Args:
            frac: Curriculum progress ∈ [0, 1].
                  0 → heading is forced to face goal (clamp to ±0 rad).
                  1 → full random heading (no clamp, ±π).
        """
        self._heading_curriculum_frac = float(max(0.0, min(1.0, frac)))

    def set_student_scan_degradation(
        self,
        dropout_rate: float = 0.15,
        artifact_rate: float = 0.02,
        artifact_std: float = 0.5,
    ) -> None:
        """Configure student depth scan degradation (Appendix B, Sec.IV-D.3).

        Applies two types of sensor degradation to the raw depth scan
        BEFORE it enters MappingNet, so the student learns to build reliable
        neural maps despite realistic sensor failures.

        Args:
            dropout_rate: Fraction of depth pixels zeroed out (missing returns).
                          [stated] Appendix B: 15 %.
            artifact_rate: Fraction of depth pixels replaced by spike artifacts
                           (multi-path / erroneous returns).
                           [stated] Appendix B: 2 %.
            artifact_std:  Gaussian std of spike artifacts.  [inferred] 0.5 m.
        """
        self._scan_dropout_rate = float(dropout_rate)
        self._artifact_rate = float(artifact_rate)
        self._artifact_std = float(artifact_std)

    def set_map_randomization(
        self,
        partial_fraction: float = 0.90,
        drop_fraction: float = 0.01,
        drift_max_m: float = 0.03,
        corrupt_var_min: float = 1.0,
    ) -> None:
        """Configure map domain randomization for student Phase 2 training.

        Three independent randomization mechanisms (Appendix B, Sec.IV-D.3):

        1. **Partial map access**: ``partial_fraction`` of envs can only access
           the current local MappingNet scan (no global WTA accumulation).
           Simulates robots that have not yet built a full map of the terrain.
           [stated] Appendix B: 90 % local-only (10 % get complete maps).

        2. **Map corruption**: In every step, ``drop_fraction`` of map cells are
           replaced with random elevation + random high variance (> corrupt_var_min).
           [stated] Appendix B: 1 % of cells corrupted, variance > 1 m².

        3. **Map drift**: The student crop centre is shifted by a continuous
           random offset uniform in [−drift_max_m, +drift_max_m] metres each
           step.  Teacher GT crop is shifted by the same range (rounded to
           integer cells at policy-map resolution).
           [stated] Appendix B: ±0.03 m (= ±3 cm).

        Call this once before Phase 2 student training begins.  Call again with
        all zeros to disable.

        Args:
            partial_fraction: Fraction of envs in local-only map mode ∈ [0, 1].
                              Default 0.90 (Appendix B: 10 % get complete maps).
            drop_fraction:    Per-cell corruption probability ∈ [0, 1].
                              Default 0.01 (Appendix B: 1 %).
            drift_max_m:      Max crop-centre drift in metres ≥ 0.
                              Default 0.03 (Appendix B: ±3 cm).
            corrupt_var_min:  Minimum variance for corrupted cells (m²).
                              Default 1.0 (Appendix B: "larger than 1 m²").
        """
        self._partial_map_fraction = float(partial_fraction)
        self._map_drop_fraction = float(drop_fraction)
        self._drift_max_m = float(drift_max_m)
        self._corrupt_var_min = float(corrupt_var_min)
        # Immediately re-sample the partial-map mask for all envs
        self._resample_partial_mask(
            torch.arange(self.num_envs, device=self._device)
        )

    # ------------------------------------------------------------------
    # Internal helpers for map randomization
    # ------------------------------------------------------------------

    def _apply_heading_curriculum(self, env_ids: torch.Tensor) -> None:
        """Restrict goal heading for newly-reset envs (Sec. IV-D.3).

        When ``_heading_curriculum_frac < 1.0``, the goal command heading is
        constrained to ±(frac × π) around the approach direction (the angle
        pointing from the robot origin toward the goal).  This biases early
        training toward goals that the robot will face after locomoting directly
        toward them, reducing the need to rotate first.

        At ``_heading_curriculum_frac = 1.0`` (full training) the full ±π
        range is restored and this function is a no-op.

        The approach direction is computed from the ``goal_pos`` command's
        [x_b, y_b] components (goal expressed in base frame).

        Args:
            env_ids: Indices of just-reset environments to apply the curriculum to.
        """
        if self._heading_curriculum_frac >= 1.0 or len(env_ids) == 0:
            return

        cmd_term = self._il_env.command_manager.get_term("goal_pos")
        cmd = cmd_term.command[env_ids]                                        # (K, 4)

        # Direction angle from base origin to goal in base frame
        approach = torch.atan2(cmd[:, 1], cmd[:, 0])                          # (K,)

        # Allowed deviation from approach direction: ±(frac × π)
        max_dev = self._heading_curriculum_frac * math.pi
        dev = (torch.rand(len(env_ids), device=self._device) * 2.0 - 1.0) * max_dev

        # New heading wrapped to [-π, π]
        new_h = approach + dev
        new_h = torch.atan2(torch.sin(new_h), torch.cos(new_h))

        cmd_term.command[env_ids, 3] = new_h

    def _resample_partial_mask(self, env_ids: torch.Tensor) -> None:
        """Re-draw full/partial map access for the given env IDs.

        Called at episode reset so each episode independently gets either
        full global map access or local-only access.
        """
        if self._partial_map_fraction <= 0.0:
            self._partial_map_mask[env_ids] = True
            return
        # True  = full global WTA map (probability = 1 - partial_fraction)
        # False = local-only (probability = partial_fraction)
        rand = torch.rand(len(env_ids), device=self._device)
        self._partial_map_mask[env_ids] = rand >= self._partial_map_fraction

    def _build_local_map(
        self,
        elev: torch.Tensor,     # (B, 1, SCAN_H, SCAN_W)  MappingNet elev output
        log_var: torch.Tensor,  # (B, 1, SCAN_H, SCAN_W)  MappingNet log_var output
    ) -> torch.Tensor:
        """Build a (B, 4, ph, pw) local-only student map from the current scan.

        Used for partial-map envs.  Bilinearly interpolates the MappingNet
        local output to policy map resolution and computes surface normals.

        Channel layout matches student_map: [elev, n_x, n_y, var].
        """
        ph = self.wta_manager.wta.policy_h
        pw = self.wta_manager.wta.policy_w
        res = self.wta_manager.wta.global_res   # policy map cell size (0.08 m)

        elev_p = F.interpolate(elev,          size=(ph, pw), mode='bilinear', align_corners=False)
        var_p  = F.interpolate(log_var.exp(), size=(ph, pw), mode='bilinear', align_corners=False)
        normals = WTAMapFusion._surface_normals(elev_p, res)                  # (B, 2, ph, pw)
        return torch.cat([elev_p, normals, var_p], dim=1)                     # (B, 4, ph, pw)

    # ------------------------------------------------------------------
    # RSL-RL VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset all envs.  Returns (obs_td, obs_td, None, None, extras)."""
        _, extras = self._gym_env.reset()
        self.wta_manager.reset()
        self.history.zero_()
        all_ids = torch.arange(self.num_envs, device=self._device)
        # Re-sample map access type (full vs partial) for all envs
        self._resample_partial_mask(all_ids)
        # Apply heading curriculum (constrain goal heading early in training)
        self._apply_heading_curriculum(all_ids)
        obs_td = self._make_obs_td()
        return obs_td, obs_td, None, None, extras

    def step(self, actions: torch.Tensor) -> tuple:
        """Step all envs, run mapping pipeline.

        Returns (obs_td, obs_td, rewards, dones, extras).
        Same obs TensorDict is passed for both actor (obs) and critic
        (privileged_obs) — AME2ActorCritic reads the appropriate keys.
        """
        _, rewards, terminated, truncated, extras = self._gym_env.step(actions)
        dones = terminated | truncated

        # Build obs (reads obs_buf populated by the step above)
        obs_td = self._make_obs_td()

        # Reset done episodes AFTER building obs so they get a clean state
        # on the *next* step, not this one.
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.wta_manager.reset(done_ids)
            self.history[done_ids] = 0.0
            # Re-draw full/partial map access for newly started episodes
            self._resample_partial_mask(done_ids)
            # Apply heading curriculum for newly started episodes
            self._apply_heading_curriculum(done_ids)

        return obs_td, obs_td, rewards, dones, extras

    def close(self) -> None:
        self._gym_env.close()

    # ------------------------------------------------------------------
    # Internal: build TensorDict from current obs_buf
    # ------------------------------------------------------------------

    def _robot_pose(self) -> torch.Tensor:
        """(B, 3) [x, y, yaw] robot pose in world frame (yaw from quaternion)."""
        robot = self._il_env.scene["robot"]
        pos  = robot.data.root_pos_w    # (B, 3)  world-frame position
        quat = robot.data.root_quat_w   # (B, 4)  [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1)

    def _make_obs_td(self) -> "TensorDict":
        """Run the full mapping pipeline and return obs TensorDict.

        Reads from ``_il_env.obs_buf`` which Isaac Lab populates on every
        step() and reset() call before returning.
        """
        from tensordict import TensorDict

        obs_buf = self._il_env.obs_buf   # dict[str, Tensor], set by IL env
        B   = self.num_envs
        cfg = self.cfg

        # ── Proprioception (policy obs group) ────────────────────────
        prop_flat = obs_buf["policy"].to(self._device)
        # prop_flat layout (always 48D, same for teacher and student envs):
        #   base_vel(3) | ang_vel+grav+q+dq+act(42) | ame2_actor_cmd(3) = 48D  [stated]
        # base_vel is available in the env for BOTH teacher and student training;
        # the student ACTOR never reads prop directly (uses history+commands+map).
        # The AsymmetricCritic always receives full 48D prop.
        commands = prop_flat[:, -cfg.d_commands:]                                 # (B, 3) = last 3D
        # hist_step: student-observable slice (no base_vel, no cmd) for LSIO history.
        # Same slice for teacher and student — history stores the 42D student-accessible part.
        hist_step = prop_flat[:, cfg.d_base_vel : cfg.d_base_vel + cfg.d_hist]  # (B, 42)

        # ── Privileged obs: height scan + foot contact forces ─────────
        priv = obs_buf.get("teacher_privileged")
        if priv is not None:
            priv = priv.to(self._device)
            n_scan = self.SCAN_H * self.SCAN_W                    # 1581
            raw_scan_flat = priv[:, :n_scan]                      # (B, 1581)
            contact_12    = priv[:, n_scan:]                      # (B, 12)
        else:
            raw_scan_flat = torch.zeros(B, self.SCAN_H * self.SCAN_W, device=self._device)
            contact_12    = torch.zeros(B, 12, device=self._device)

        # Per-foot binary contact state: (B, 12) → reshape (B,4,3) → norm → threshold → (B, 4)
        # Paper (Sec IV-B): "contact state of each link" — binary contact indicators.
        # FIX: was continuous force magnitude; now thresholded to binary (0/1).
        #
        # TODO (manager-based Phase 2): The manager-based env only exposes foot contact
        # forces (4 feet × 3D = 12 floats) in the teacher_privileged obs group.  The
        # AsymmetricCritic in AME2ActorCritic expects a 13D all-link contact vector
        # (base+thigh+shank+foot = 13 links, [stated Sec.IV-B]).  When Phase 2 training
        # is activated via the manager-based env, the "contact" obs group must be extended
        # to include base(1) + thigh(4) + shank(4) + foot(4) = 13D binary contact states,
        # and _LR_CONTACT_PERM must be used for the full 13D vector.  Until then, the
        # manager-based path passes a 4D foot-only contact tensor, which is incompatible
        # with the trained direct-workflow model.
        contact_force_mag = contact_12.reshape(B, 4, 3).norm(dim=-1)  # (B, 4)
        contact_4 = (contact_force_mag > CONTACT_FORCE_THRESHOLD).float()  # (B, 4) binary

        # ── GT teacher map (teacher_map obs group, F15) ───────────────
        gt_map_flat = obs_buf.get("teacher_map")
        if gt_map_flat is not None:
            gt_map_flat = gt_map_flat.to(self._device)            # (B, 1512)

        # ── MappingNet (always eval / no grad during rollout) ─────────
        raw_scan = raw_scan_flat.reshape(B, 1, self.SCAN_H, self.SCAN_W)

        # Perception noise curriculum (Sec. IV-D3): linearly ramp from
        # 0 → SCAN_NOISE_MAX over the first 20 % of teacher training.
        # set_scan_noise_scale() is called each iteration from train_ame2.py.
        # BUG FIX: was Gaussian (randn); paper says "zero-mean uniform noise" (Appendix B).
        if self._scan_noise_scale > 0.0:
            noise_max = self._scan_noise_scale * self.SCAN_NOISE_MAX
            raw_scan = raw_scan + (torch.rand_like(raw_scan) * 2.0 - 1.0) * noise_max

        # Student depth scan degradation (Sec.IV-D.3): missing returns + artifacts
        # Applied BEFORE MappingNet so the network learns to handle sensor failures.
        if self.is_student:
            if self._scan_dropout_rate > 0.0:
                # [stated] Appendix B: 15 % of depth pixels missing
                drop_mask = torch.rand_like(raw_scan) < self._scan_dropout_rate
                raw_scan = raw_scan.masked_fill(drop_mask, 0.0)
            if self._artifact_rate > 0.0 and self._artifact_std > 0.0:
                # [stated] Appendix B: 2 % artifact pixels (random returns)
                spike_mask = torch.rand_like(raw_scan) < self._artifact_rate
                raw_scan = raw_scan + spike_mask.float() * (
                    torch.randn_like(raw_scan) * self._artifact_std
                )

        self.mapping_net.eval()
        with torch.no_grad():
            elev, log_var = self.mapping_net(raw_scan)            # (B,1,31,51) each

        # ── WTA update + policy map crop ──────────────────────────────
        poses = self._robot_pose()                                # (B, 3)
        self.wta_manager.update(elev, log_var, poses)

        # Student map drift: continuous ±drift_max_m metres to crop centre.
        # [stated] Appendix B: "random drift within [−0.03, 0.03] m when querying
        # map observations" — simulates localization error in the student.
        poses_crop = poses
        if self._drift_max_m > 0.0 and self.is_student:
            xy_off = (torch.rand(B, 2, device=self._device) * 2.0 - 1.0) * self._drift_max_m
            poses_crop = poses.clone()
            poses_crop[:, :2] = poses_crop[:, :2] + xy_off

        maps = self.wta_manager.get_policy_maps(poses_crop, gt_map_flat=gt_map_flat)
        student_map = maps["student_map"].clone()   # (B, 4, 14, 36) — writable copy
        teacher_map = maps["teacher_map"]           # (B, 3, 14, 36)

        # Teacher map drift: same ±drift_max_m applied to GT crop centre.
        # [stated] Appendix B: drift applies to "both the teacher and the student".
        # At 8 cm/cell, 3 cm = 0.375 cells → integer approximation = 0..1 cells.
        if self._drift_max_m > 0.0:
            _drift_cells = max(0, round(self._drift_max_m / self.wta_manager.wta.global_res))
            if _drift_cells > 0:
                di = torch.randint(-_drift_cells, _drift_cells + 1,
                                   (B,), device=self._device)
                dj = torch.randint(-_drift_cells, _drift_cells + 1,
                                   (B,), device=self._device)
                teacher_map = _shift_map_batch(teacher_map.float(), di, dj)

        # ── Student map domain randomization (Sec.IV-D.3, Phase 2 only) ──────
        if self.is_student:
            # 1. Partial map access: envs assigned local-only get the current
            #    MappingNet output (no global WTA accumulation) instead of the
            #    WTA crop.  Simulates robots without prior map data.  [stated]
            partial_envs = ~self._partial_map_mask             # True = local-only
            if partial_envs.any():
                local_map = self._build_local_map(elev, log_var)  # (B, 4, ph, pw)
                student_map[partial_envs] = local_map[partial_envs]

            # 1b. Complete-map environments: override variance to 0.0025 m²
            # [stated] Appendix B: "complete maps with variance 0.0025 m²"
            # This signals low uncertainty to the student policy for well-observed terrain.
            if self._partial_map_fraction > 0.0:
                complete_envs = self._partial_map_mask          # True = complete map
                if complete_envs.any():
                    student_map[complete_envs, 3] = 0.0025

            # 2. Map corruption: replace a random fraction of cells with random
            #    elevation + high uncertainty sentinel.  Forces the policy to
            #    be robust to locally corrupted map data.  [stated]
            if self._map_drop_fraction > 0.0:
                ph, pw = student_map.shape[2], student_map.shape[3]
                drop_mask = (
                    torch.rand(B, ph, pw, device=self._device) < self._map_drop_fraction
                )  # (B, H, W) boolean
                drop_4 = drop_mask.unsqueeze(1).expand_as(student_map)  # (B, 4, H, W)
                # [stated] Appendix B: "random values with a random variance larger than 1 m²"
                corrupt = torch.zeros_like(student_map)
                corrupt[:, 0] = torch.randn(B, ph, pw, device=self._device) * 0.3
                # Variance sampled from [corrupt_var_min, corrupt_var_min + 9] m²
                corrupt[:, 3] = (
                    self._corrupt_var_min
                    + torch.rand(B, ph, pw, device=self._device) * 9.0
                )
                student_map = torch.where(drop_4, corrupt, student_map)

        # ── Update LSIO history ring buffer ───────────────────────────
        # Roll left by 1 step, insert newest at the end
        self.history = torch.roll(self.history, -1, dims=1)
        self.history[:, -1, :] = hist_step.detach()

        # ── Critic prop: base(45D) + critic_cmd(5D) = 50D  [stated Sec.IV-E.3] ──
        # prop_flat = [base_vel(3) | hist(42) | actor_cmd(3)] = 48D
        # base(45D) = prop_flat[:, :45]  (drops the 3D actor cmd)
        # critic_cmd(5D) = [x_rel, y_rel, sin(yaw), cos(yaw), t_remaining]
        #   from the "critic_extra" obs group in Isaac Lab env
        prop_base = prop_flat[:, : cfg.d_base_vel + cfg.d_hist]       # (B, 45)
        critic_cmd_raw = obs_buf.get("critic_extra")
        if critic_cmd_raw is not None:
            critic_cmd  = critic_cmd_raw.to(self._device)              # (B, 5)
            critic_prop = torch.cat([prop_base, critic_cmd], dim=-1)   # (B, 50)
        else:
            # Fallback (standalone testing): pad actor_cmd with zeros for t_remaining
            critic_prop = torch.cat(
                [prop_base, prop_flat[:, -cfg.d_commands:],
                 torch.zeros(B, 2, device=self._device)], dim=-1
            )  # (B, 50) — approximation, only valid offline

        # ── Assemble TensorDict ───────────────────────────────────────
        td_data: dict[str, torch.Tensor] = {
            # Actor map: teacher uses GT 3-ch; student uses neural 4-ch
            "map":         student_map if self.is_student else teacher_map,
            "map_teacher": teacher_map,    # critic always sees GT map
            "prop":        prop_flat,      # (B, 48) — actor + teacher critic
            "critic_prop": critic_prop,    # (B, 50) — AsymmetricCritic input
            "contact":     contact_4,
        }
        if self.is_student:
            td_data["history"]  = self.history.clone()
            td_data["commands"] = commands

        return TensorDict(td_data, batch_size=[B], device=self._device)


# ===================================================================
# Sanity check (no Isaac Sim dependency)
# ===================================================================

if __name__ == "__main__":
    from tensordict import TensorDict

    device = "cpu"
    cfg = PolicyConfig()
    B = 4

    # Build mock TensorDict matching what the env would provide
    obs_td = TensorDict({
        "prop":         torch.randn(B, cfg.d_prop_raw),
        "critic_prop":  torch.randn(B, cfg.d_prop_critic),   # 50D: base(45)+critic_cmd(5)
        "map":          torch.randn(B, cfg.d_map_teacher, cfg.map_h, cfg.map_w),
        "map_teacher":  torch.randn(B, cfg.d_map_teacher, cfg.map_h, cfg.map_w),
        "contact":      torch.zeros(B, 13),
    }, batch_size=[B])

    obs_groups_teacher = {
        "policy": ["prop", "map"],
        "critic": ["critic_prop", "map_teacher", "contact"],
    }

    # --- Teacher ---
    print("=== AME2ActorCritic (teacher) ===")
    teacher_ac = AME2ActorCritic(
        obs_td, obs_groups_teacher, num_actions=cfg.num_joints,
        ame2_cfg=cfg, is_student=False,
    ).to(device)

    actions = teacher_ac.act(obs_td)
    print(f"  act():           {actions.shape}")
    assert actions.shape == (B, cfg.num_joints)

    actions_det = teacher_ac.act_inference(obs_td)
    print(f"  act_inference(): {actions_det.shape}")
    assert actions_det.shape == (B, cfg.num_joints)

    # evaluate uses same obs TensorDict (critic extracts its own groups)
    value = teacher_ac.evaluate(obs_td)
    print(f"  evaluate():      {value.shape}")
    assert value.shape == (B, 1)

    log_prob = teacher_ac.get_actions_log_prob(actions)
    print(f"  log_prob:        {log_prob.shape}")
    assert log_prob.shape == (B,)

    # Properties
    print(f"  action_mean:     {teacher_ac.action_mean.shape}")
    assert teacher_ac.action_mean.shape == (B, cfg.num_joints)
    print(f"  action_std:      {teacher_ac.action_std.shape}")
    assert teacher_ac.action_std.shape == (B, cfg.num_joints)
    print(f"  entropy:         {teacher_ac.entropy.shape}")
    assert teacher_ac.entropy.shape == (B,)

    print(f"  is_recurrent:    {teacher_ac.is_recurrent}")
    print(f"  actor params:    {sum(p.numel() for p in teacher_ac.actor.parameters()):,}")
    print(f"  critic params:   {sum(p.numel() for p in teacher_ac.critic.parameters()):,}")

    # update_normalization (no-op but must exist)
    teacher_ac.update_normalization(obs_td)
    print("  update_normalization(): OK")

    # load_state_dict returns True
    sd = teacher_ac.state_dict()
    assert teacher_ac.load_state_dict(sd) is True
    print("  load_state_dict(): OK")

    # --- Student ---
    print("\n=== AME2StudentActorCritic (student) ===")

    obs_td_student = TensorDict({
        "prop":        torch.randn(B, cfg.d_prop_raw),
        "critic_prop": torch.randn(B, cfg.d_prop_critic),   # 50D: base(45)+critic_cmd(5)
        "map":         torch.randn(B, cfg.d_map_student, cfg.map_h, cfg.map_w),
        "history":     torch.randn(B, cfg.prop_history, cfg.d_hist),
        "commands":    torch.randn(B, cfg.d_commands),
        "map_teacher": torch.randn(B, cfg.d_map_teacher, cfg.map_h, cfg.map_w),
        # NOTE: manager-based env only provides 4D foot contact; direct env provides 13D.
        # This test uses 4D as placeholder — real Phase 2 training requires 13D.
        "contact":     torch.zeros(B, 4),
    }, batch_size=[B])

    obs_groups_student = {
        "policy": ["map", "history", "commands"],
        "critic": ["prop", "map_teacher", "contact"],
    }

    student_ac = AME2StudentActorCritic(
        obs_td_student, obs_groups_student, num_actions=cfg.num_joints,
        ame2_cfg=cfg,
    ).to(device)

    s_actions, s_map_emb, s_prop_emb = student_ac.act_and_embed(obs_td_student)
    print(f"  act_and_embed(): actions={s_actions.shape}, map_emb={s_map_emb.shape}")
    assert s_actions.shape == (B, cfg.num_joints)

    # Test act() via standard RSL-RL interface
    s_actions2 = student_ac.act(obs_td_student)
    print(f"  act():           {s_actions2.shape}")
    assert s_actions2.shape == (B, cfg.num_joints)

    # Evaluate
    s_value = student_ac.evaluate(obs_td_student)
    print(f"  evaluate():      {s_value.shape}")
    assert s_value.shape == (B, 1)

    # Simulate teacher embeddings for distillation
    t_map_emb = torch.randn(B, cfg.d_map_emb)
    t_prop_emb = torch.randn(B, cfg.d_prop_emb)
    t_actions = torch.randn(B, cfg.num_joints)
    advantages = torch.randn(B)
    log_probs_old = torch.randn(B)
    returns = torch.randn(B)

    # Phase 1: PPO disabled
    student_ac.set_iteration(0)
    loss_dict = student_ac.compute_student_loss(
        teacher_map_emb=t_map_emb, teacher_prop_emb=t_prop_emb,
        student_map_emb=s_map_emb, student_prop_emb=s_prop_emb,
        teacher_actions=t_actions, student_actions=s_actions,
        advantages=advantages, log_probs_old=log_probs_old, returns=returns,
    )
    print(f"  Phase 1 loss: total={loss_dict['total']:.4f}, "
          f"ppo={loss_dict['ppo']:.4f}, dist={loss_dict['dist']:.4f}, "
          f"repr={loss_dict['repr']:.4f}")
    assert loss_dict["ppo"].item() == 0.0, "PPO should be zero in phase 1"

    # Phase 2: PPO enabled
    student_ac.set_iteration(5000)
    student_ac.act(obs_td_student)  # rebuild distribution
    loss_dict2 = student_ac.compute_student_loss(
        teacher_map_emb=t_map_emb, teacher_prop_emb=t_prop_emb,
        student_map_emb=s_map_emb, student_prop_emb=s_prop_emb,
        teacher_actions=t_actions, student_actions=s_actions,
        advantages=advantages, log_probs_old=log_probs_old, returns=returns,
    )
    print(f"  Phase 2 loss: total={loss_dict2['total']:.4f}, "
          f"ppo={loss_dict2['ppo']:.4f}, dist={loss_dict2['dist']:.4f}, "
          f"repr={loss_dict2['repr']:.4f}")

    # --- Symmetry augmentation sanity check ---
    print("\n=== L-R Symmetry Augmentation ===")
    map_t   = obs_td["map_teacher"]   # (B, 3, 14, 36)
    prop_t  = obs_td["prop"]          # (B, 48)
    cont_t  = obs_td["contact"]       # (B, 13)

    map_f, prop_f, cont_f = _flip_lr(map_t, prop_t, cont_t, is_teacher=True)
    # Double flip must recover original
    map_ff, prop_ff, cont_ff = _flip_lr(map_f, prop_f, cont_f, is_teacher=True)
    assert torch.allclose(map_ff,  map_t,  atol=1e-6), "Map double-flip failed"
    assert torch.allclose(prop_ff, prop_t, atol=1e-6), "Prop double-flip failed"
    assert torch.allclose(cont_ff, cont_t, atol=1e-6), "Contact double-flip failed"
    print("  double-flip identity: OK")

    # Verify H dimension (lateral/y axis) is flipped and ny channel negated
    expected_map_f = map_t.flip(-2).clone()
    expected_map_f[:, 2] = -expected_map_f[:, 2]   # ny negation
    assert torch.allclose(map_f, expected_map_f), "Map lateral-flip mismatch"
    print("  map H-flip (lateral): OK")

    # evaluate() output shape unchanged (B,1) despite internal augmentation
    v_aug = teacher_ac.evaluate(obs_td)
    assert v_aug.shape == (B, 1), f"evaluate() shape wrong: {v_aug.shape}"
    print(f"  evaluate() with augmentation: {v_aug.shape}")

    # Symmetry check: a perfectly symmetric critic gives V(s) == V(flip(s))
    # (not guaranteed before training, but shapes must be consistent)
    print("  L-R symmetry augmentation: OK")

    # --- PPO call-sequence simulation ---
    print("\n=== PPO call-sequence simulation ===")
    # This mimics what PPO.act() does:
    #   transition.actions = policy.act(obs).detach()
    #   transition.values = policy.evaluate(obs).detach()
    #   transition.actions_log_prob = policy.get_actions_log_prob(transition.actions).detach()
    #   transition.action_mean = policy.action_mean.detach()
    #   transition.action_sigma = policy.action_std.detach()
    act_out = teacher_ac.act(obs_td)
    val_out = teacher_ac.evaluate(obs_td)
    lp_out = teacher_ac.get_actions_log_prob(act_out)
    mu_out = teacher_ac.action_mean
    sigma_out = teacher_ac.action_std
    print(f"  actions:    {act_out.shape}")
    print(f"  values:     {val_out.shape}")
    print(f"  log_prob:   {lp_out.shape}")
    print(f"  action_mean:{mu_out.shape}")
    print(f"  action_std: {sigma_out.shape}")
    assert act_out.shape == (B, cfg.num_joints)
    assert val_out.shape == (B, 1)
    assert lp_out.shape == (B,)
    assert mu_out.shape == (B, cfg.num_joints)
    assert sigma_out.shape == (B, cfg.num_joints)
    print("  PPO.act() sequence: OK")

    # PPO.update() call sequence:
    #   policy.act(obs_batch, masks=None, hidden_state=None)
    #   policy.get_actions_log_prob(actions_batch)
    #   policy.evaluate(obs_batch, masks=None, hidden_state=None)
    #   policy.action_mean, policy.action_std, policy.entropy
    teacher_ac.act(obs_td, masks=None, hidden_state=None)
    lp2 = teacher_ac.get_actions_log_prob(act_out)
    val2 = teacher_ac.evaluate(obs_td, masks=None, hidden_state=None)
    ent = teacher_ac.entropy
    print(f"  PPO.update() entropy: {ent.shape}")
    assert ent.shape == (B,)
    print("  PPO.update() sequence: OK")

    # --- WTAMapManager ---
    print("\n=== WTAMapManager ===")
    map_mgr = WTAMapManager(num_envs=B, device=device)
    map_mgr.reset()

    mapping_net = MappingNet(MappingConfig()).to(device)
    for step in range(5):
        poses = torch.zeros(B, 3, device=device)
        poses[:, 0] = step * 0.3
        raw = torch.randn(B, 1, 31, 51, device=device)
        elev, lv = mapping_net(raw)
        map_mgr.update(elev, lv, poses)

    final_poses = torch.zeros(B, 3, device=device)
    final_poses[:, 0] = 5 * 0.3
    # Without GT map (offline fallback — student_map[:, :3])
    policy_maps = map_mgr.get_policy_maps(final_poses)
    print(f"  student_map: {policy_maps['student_map'].shape}")
    print(f"  teacher_map (fallback): {policy_maps['teacher_map'].shape}")
    assert policy_maps["student_map"].shape == (B, 4, cfg.map_h, cfg.map_w)
    assert policy_maps["teacher_map"].shape == (B, 3, cfg.map_h, cfg.map_w)

    # With GT map (correct path used during real training)
    gt_flat = torch.randn(B, 3 * cfg.map_h * cfg.map_w, device=device)
    policy_maps_gt = map_mgr.get_policy_maps(final_poses, gt_map_flat=gt_flat)
    print(f"  teacher_map (GT): {policy_maps_gt['teacher_map'].shape}")
    assert policy_maps_gt["teacher_map"].shape == (B, 3, cfg.map_h, cfg.map_w)
    # GT values must be preserved (no accidental overwrite from student_map)
    assert torch.allclose(
        policy_maps_gt["teacher_map"],
        gt_flat.reshape(B, 3, cfg.map_h, cfg.map_w),
    ), "GT teacher_map values were corrupted"

    # Test partial reset
    reset_ids = torch.tensor([0, 2], dtype=torch.long)
    map_mgr.reset(reset_ids)
    print(f"  Reset envs {reset_ids.tolist()} -- OK")

    # Test 4-column poses (with z)
    poses_4col = torch.zeros(B, 4, device=device)
    poses_4col[:, 0] = 1.0
    poses_4col[:, 3] = 0.1
    maps_4 = map_mgr.get_policy_maps(poses_4col)
    print(f"  4-col poses crop: {maps_4['student_map'].shape}")

    # --- Factory ---
    print("\n=== make_ame2_rslrl_agent ===")
    agent_t = make_ame2_rslrl_agent(
        obs_td, obs_groups_teacher, cfg.num_joints,
        stage="teacher", device=device,
    )
    agent_s = make_ame2_rslrl_agent(
        obs_td_student, obs_groups_student, cfg.num_joints,
        stage="student", device=device,
    )
    print(f"  teacher type: {type(agent_t).__name__}")
    print(f"  student type: {type(agent_s).__name__}")
    assert isinstance(agent_t, AME2ActorCritic)
    assert isinstance(agent_s, AME2StudentActorCritic)

    # --- Domain randomization helpers ---
    print("\n=== Domain Randomization Helpers ===")

    # _shift_map_batch: vectorised per-env shift
    map_orig = torch.randn(B, 3, 14, 36, device=device)
    di = torch.zeros(B, dtype=torch.long, device=device)
    dj = torch.zeros(B, dtype=torch.long, device=device)
    map_shifted = _shift_map_batch(map_orig.float(), di, dj)
    assert torch.allclose(map_shifted, map_orig.float(), atol=1e-5), \
        "zero-shift should be identity"
    print("  _shift_map_batch (zero-shift identity): OK")

    # Non-zero shift: output shape unchanged
    di2 = torch.randint(-3, 4, (B,), device=device)
    dj2 = torch.randint(-3, 4, (B,), device=device)
    map_s2 = _shift_map_batch(map_orig.float(), di2, dj2)
    assert map_s2.shape == map_orig.shape, "shift must preserve shape"
    print(f"  _shift_map_batch (random shift, shape ok): {map_s2.shape}")

    # WTAMapManager.get_policy_maps with drift via student poses offset
    drift_poses = final_poses.clone()
    drift_poses[:, :2] += 0.3 * (torch.rand(B, 2, device=device) * 2 - 1)
    maps_drift = map_mgr.get_policy_maps(drift_poses)
    assert maps_drift["student_map"].shape == (B, 4, cfg.map_h, cfg.map_w)
    print(f"  student_map with pose drift: {maps_drift['student_map'].shape} OK")

    # _build_local_map (used by partial map access)
    # Create a mock AME2MapEnvWrapper to test the helper method
    class _MockWTAWrapper:
        class wta:
            policy_h = cfg.map_h
            policy_w = cfg.map_w
            global_res = 0.08
    class _MockWrapper:
        wta_manager = _MockWTAWrapper()
        _device = device
    mock = _MockWrapper()
    elev_mock = torch.randn(B, 1, 31, 51, device=device)
    lv_mock   = torch.randn(B, 1, 31, 51, device=device)
    local_map = AME2MapEnvWrapper._build_local_map(mock, elev_mock, lv_mock)
    assert local_map.shape == (B, 4, cfg.map_h, cfg.map_w), \
        f"_build_local_map shape wrong: {local_map.shape}"
    print(f"  _build_local_map: {local_map.shape} OK")

    print("\nAll checks passed.")
