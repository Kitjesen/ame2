"""
AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding
Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter (ETH Zurich RSL)
arXiv: 2601.08485

1:1 architecture reproduction from paper description + Fig. 2, 3, 6, 8.
Dimensions not stated explicitly in the paper are marked with [inferred].

Verified against paper by multi-agent review (2026-02-24).
Fixes applied 2026-02-24:
  [F9]  AME2Encoder local-feature pipeline:
          paper: CNN(map) → pos_emb MLP(coords) → fusion MLP → pointwise_local
          (was: Linear+concat+CNN; missing independent fusion MLP)
  [F10] d_pe renamed to d_pos_emb (paper's dPE = proprioception embed dim, not positional)
  [F11] AME2Policy.forward() now returns (actions, map_emb, prop_emb)
          StudentLoss.L_repr requires real map embeddings, not dummy tensors
  [F12] LSIO replaced from 2-layer GRU → real dual-history 1D CNN architecture
          Short history: last 4 steps flattened (168 dims)
          Long  history: all T=20 steps → Conv1d[6,32,3]+Conv1d[4,16,2] → 16 dims
          LSIO out_size = 184; StudentPropEncoder MLP updated accordingly

NOT implemented (training infrastructure, not network architecture):
  - Asymmetric critic (MoE design, receives link contact states — paper Sec.IV-B)
  - PPO runner, terrain curriculum, Isaac Lab env wrappers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MappingConfig:
    """
    MappingNet operates on raw depth-projected local grids at 4cm resolution.
    Stated in paper (Appendix / Sec.V):
      ANYmal-D: 51×31 cells, center x=1.0m, y=0m (base frame)
      TRON1:    31×31 cells, center x=0.6m, y=0m
    Default: ANYmal-D.
    """
    map_h: int = 31           # stated: ANYmal-D local grid height
    map_w: int = 51           # stated: ANYmal-D local grid width
    cnn_channels: int = 16    # stated in paper (Fig. 8)
    pool_kernel: int = 3      # stated in paper (Fig. 8)
    pool_stride: int = 3      # stated in paper (Fig. 8)


@dataclass
class PolicyConfig:
    """
    Policy map (queried from fused global map) at 8cm resolution.
    Stated in paper (Appendix):
      ANYmal-D: 36×14 cells, center x=0.6m, y=0m
      TRON1:    18×13 cells, center x=0.32m, y=0m
    Default: ANYmal-D.
    """
    # Map dimensions (stated)
    map_h: int = 14           # stated: ANYmal-D policy map height
    map_w: int = 36           # stated: ANYmal-D policy map width
    d_map_teacher: int = 3    # stated: x, y, z (elevation + 2 normals)
    d_map_student: int = 4    # stated: teacher channels + uncertainty

    # AME-2 encoder internals
    # d_local=64: confirmed from AME-1 (He et al. 2025 [15]) Sec."Training": "d=64 for the MHA dimension"
    #   AME-2 builds on AME-1 and uses the same d for the local feature / attention embed dim.
    d_local: int = 64
    d_global: int = 128   # [inferred] — neither paper states this; d_map_emb = d_local + d_global
    # NOTE: paper's notation "dPE" = proprioception embedding dim (= d_prop_emb below).
    # d_pos_emb here is the POSITIONAL embedding dim for map coordinates [inferred].
    d_pos_emb: int = 64   # [inferred]
    # num_heads=16: confirmed from AME-1 Sec."Training": "h=16 for the number of heads" with d=64
    #   → 64/16 = 4 dims per head (unusual but explicit in source paper)
    num_heads: int = 16

    # Proprioception dimensions (ANYmal-D, Sec.III-B + Table I)
    #
    # Teacher actor (privileged, d_prop_raw=48):
    #   base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3)
    #   + joint_pos(12) + joint_vel(12) + prev_actions(12) + cmd_actor(3) = 48
    #   base_lin_vel is TEACHER-ONLY (stated); student never receives it.
    #
    # Student actor (deployed, no privileged info):
    #   history per step (d_hist=42): ang_vel(3)+gravity(3)+q(12)+dq(12)+actions(12)
    #   current commands (d_commands=3): fed separately to StudentPropEncoder
    #
    # Actor command (3D, STATED Sec.IV-E): [clip(d_xy, max=2m), sin(yaw), cos(yaw)]
    # Critic command (5D, STATED Sec.IV-E): [x_rel, y_rel, sin(yaw), cos(yaw), t_remaining]
    d_prop_raw: int = 48      # actor teacher prop:  base_vel(3)+hist(42)+cmd_actor(3)   [stated]
    d_prop_critic: int = 50   # critic prop:         base_vel(3)+hist(42)+cmd_critic(5)  [stated]
    d_prop_emb: int = 128     # proprioception embedding dim (paper symbol dPE) [inferred]
    d_base_vel: int = 3       # base linear velocity (teacher only, stated)
    d_commands: int = 3       # clip(d_xy,2), sin(yaw), cos(yaw) [stated Sec.IV-E]
    d_commands_critic: int = 5  # x_rel, y_rel, sin(yaw), cos(yaw), t_remaining [stated]
    prop_history: int = 20    # stated: student stacks past 20 steps

    # Derived: student per-step history = prop without base_lin_vel and commands
    @property
    def d_hist(self) -> int:
        return self.d_prop_raw - self.d_base_vel - self.d_commands  # = 42

    # Decoder
    d_map_emb: int = 192      # must equal d_local + d_global = 64 + 128
    num_joints: int = 12      # ANYmal-D: 3 joints × 4 legs = 12 DoF; confirmed by AME-1 Table 2 (j=[0:12])


# ---------------------------------------------------------------------------
# Robot-specific config presets  (Appendix, stated)
# ---------------------------------------------------------------------------

def anymal_d_mapping_cfg() -> MappingConfig:
    """ANYmal-D MappingNet config (default). local grid 51×31 @ 4cm, cx=1.0m."""
    return MappingConfig(map_h=31, map_w=51)

def tron1_mapping_cfg() -> MappingConfig:
    """TRON1 MappingNet config (Appendix, stated). local grid 31×31 @ 4cm, cx=0.6m."""
    return MappingConfig(map_h=31, map_w=31)

def anymal_d_policy_cfg() -> PolicyConfig:
    """ANYmal-D policy config (default). policy map 36×14 @ 8cm, cx=0.6m."""
    return PolicyConfig(map_h=14, map_w=36)

def tron1_policy_cfg() -> PolicyConfig:
    """
    TRON1 policy config (Appendix, stated). policy map 18×13 @ 8cm, cx=0.32m.
    d_map_emb kept at 192 (same encoder dims — [inferred] same as ANYmal-D).
    """
    return PolicyConfig(map_h=13, map_w=18)

# WTAMapFusion robot presets — pass as kwargs to WTAMapFusion.__init__
ANYMAL_D_WTA_KWARGS = dict(
    local_h=31, local_w=51, local_cx=1.0, local_cy=0.0,   # stated
    policy_h=14, policy_w=36, policy_cx=0.6, policy_cy=0.0,  # stated
)
TRON1_WTA_KWARGS = dict(
    local_h=31, local_w=31, local_cx=0.6, local_cy=0.0,   # stated
    policy_h=13, policy_w=18, policy_cx=0.32, policy_cy=0.0,  # stated
)


# ---------------------------------------------------------------------------
# Part 1: Neural Mapping Network  (Fig. 8 — lightweight U-Net)
# ---------------------------------------------------------------------------

class MappingNet(nn.Module):
    """
    Predicts per-cell elevation estimates + log-variance uncertainties
    from a noisy, partially observed local elevation grid.

    Architecture (Fig. 8):
        Input (1, H, W)
          → 2×Conv(ch=16, k=3, pad=1) + ReLU          [encoder]
          → MaxPool(k=3, stride=3)                      [bottleneck]
          → Upsample → Concat(skip)                     [skip connection]
          → 2×Conv(ch=16, k=3, pad=1) + ReLU           [decoder]
          → THREE parallel Conv(ch=1, k=1) heads:
              head_elev  → Raw Estimation
              head_unc   → Uncertainty Output (log-variance)
              head_gate  → Sigmoid → Gating Map           [FIX: independent head]
          → Estimation Output = gate * raw_elev + (1 - gate) * x

    Loss: β-NLL (β=0.5, eq.9), sample-weighted by terrain total variation (eq.10).
    """

    def __init__(self, cfg: MappingConfig = MappingConfig()):
        super().__init__()
        self.cfg = cfg
        ch = cfg.cnn_channels  # 16

        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(cfg.pool_kernel, cfg.pool_stride)

        # Decoder: skip (16) + upsampled (16) → 32 input channels
        self.dec = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Three independent 1×1 output heads (Fig. 8, stated in paper)
        self.head_elev = nn.Conv2d(ch, 1, 1)   # raw elevation estimate
        self.head_unc  = nn.Conv2d(ch, 1, 1)   # log-variance uncertainty
        self.head_gate = nn.Conv2d(ch, 1, 1)   # gating signal (FIX: separate head)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, H, W)  noisy raw elevation grid
        returns:
            elev:    (B, 1, H, W)  elevation estimate (gated)
            log_var: (B, 1, H, W)  log-variance uncertainty
        """
        skip   = self.enc(x)                                      # (B, 16, H, W)
        pooled = self.pool(skip)                                   # (B, 16, H/3, W/3)
        up     = F.interpolate(pooled, size=skip.shape[2:],
                               mode='bilinear', align_corners=False)
        feat   = self.dec(torch.cat([up, skip], dim=1))           # (B, 16, H, W)

        raw_elev = self.head_elev(feat)                           # (B, 1, H, W)
        log_var  = self.head_unc(feat)                            # (B, 1, H, W)

        # FIX: independent gate head → sigmoid → gated combination with input
        # Paper: "Estimation Output = Gating * Raw_Estimation + (1-Gating) * Input"
        gate = torch.sigmoid(self.head_gate(feat))                # (B, 1, H, W)
        elev = gate * raw_elev + (1.0 - gate) * x

        return elev, log_var

    @staticmethod
    def beta_nll_loss(pred_elev, log_var, target, beta=0.5,
                      tv_weights=None):
        """
        β-NLL loss (eq. 9 in paper, Seitzer et al. 2022):

            L_β = E[ sg[var^β] * (log_var/2 + (y - μ)² / (2·var)) ]

        With β=0.5:  sg[var^0.5] = sg[std]

        FIX: Apply stop-gradient to var^β (the full std), not just log_var.
             The beta parameter is now correctly used.

        tv_weights: (B,) per-sample weights from total variation (eq. 10).
        """
        var = torch.exp(log_var)
        nll = log_var / 2.0 + (target - pred_elev) ** 2 / (2.0 * var)
        # sg[var^β] = sg[std] when β=0.5
        weight = (var ** beta).detach()
        loss = (weight * nll).mean(dim=[1, 2, 3])   # (B,)
        if tv_weights is not None:
            loss = (loss * tv_weights).sum()
        else:
            loss = loss.mean()
        return loss

    @staticmethod
    def total_variation_weight(y: torch.Tensor, eps=1e-8):
        """
        Per-sample TV weight for batch reweighting (eq. 10).
            TV(Y_b) = (1/HW) * (||∇x Y_b||₁ + ||∇y Y_b||₁)
            w_b = TV(Y_b) / (Σ TV(Y_b') + ε)
        y: (B, 1, H, W) ground-truth elevations.
        """
        dy = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean(dim=[1, 2, 3])
        dx = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean(dim=[1, 2, 3])
        tv = (dy + dx) / 2.0
        w  = tv / (tv.sum() + eps)
        return w


# ---------------------------------------------------------------------------
# Part 1.5: Probabilistic WTA Global Map Fusion  (Sec. V, Eq. 6–8)
# ---------------------------------------------------------------------------

class WTAMapFusion(nn.Module):
    """
    Probabilistic Winner-Take-All global map fusion (Sec. V, Eq. 6–8).
    Stateful, no learnable parameters.  Call reset() at each episode start.

    Connects MappingNet output to AME2Policy (student) input:
        for each step:
            elev_local, log_var_local = mapping_net(raw_depth_grid)
            wta.update(elev_local, log_var_local, poses)
        policy_map = wta.crop(poses)   → (B, 4, policy_h, policy_w)

    Coordinate convention (robot base frame, paper Appendix):
        x = forward,  y = left
        Local grid center : (local_cx=1.0 m, 0 m) — ANYmal-D, stated
        Policy crop center: (policy_cx=0.6 m, 0 m) — ANYmal-D, stated

    Coverage (ANYmal-D):
        Local  (4 cm): x ∈ [0.0, 2.0] m,  y ∈ [−0.60, 0.60] m
        Policy (8 cm): x ∈ [−0.8, 2.0] m, y ∈ [−0.52, 0.52] m
        Global (8 cm): 400 × 400 cells = 32 m × 32 m  [inferred]

    Eq. 8 (WTA): for each cell, keep the observation with the lowest variance.
    When multiple local cells project to the same global cell in a single step,
    the one with the minimum variance wins (sort-by-descending-var + scatter).

    Output channels: [elevation, normal_x, normal_y, variance]  (d_map_student=4)
    """

    INF_VAR: float = 1e4   # sentinel for unobserved cells

    def __init__(
        self,
        B: int,
        *,
        global_h: int    = 400,
        global_w: int    = 400,
        local_res: float = 0.04,    # 4 cm/cell — stated
        global_res: float = 0.08,   # 8 cm/cell — stated
        local_h: int  = 31,         # stated
        local_w: int  = 51,         # stated
        local_cx: float = 1.0,      # stated
        local_cy: float = 0.0,      # stated
        policy_h: int = 14,         # stated
        policy_w: int = 36,         # stated
        policy_cx: float = 0.6,     # stated
        policy_cy: float = 0.0,     # stated
    ):
        super().__init__()
        self.B          = B
        self.global_h   = global_h
        self.global_w   = global_w
        self.global_res = global_res
        self.policy_h   = policy_h
        self.policy_w   = policy_w

        # Pre-compute local grid cell (x, y) in robot frame — (N, 2)
        r, c = torch.arange(local_h, dtype=torch.float32), torch.arange(local_w, dtype=torch.float32)
        gr, gc = torch.meshgrid(r, c, indexing='ij')
        x = (gc - (local_w - 1) / 2.0) * local_res + local_cx
        y = (gr - (local_h - 1) / 2.0) * local_res + local_cy
        self.register_buffer('_local_pts', torch.stack([x.flatten(), y.flatten()], dim=1))

        # Pre-compute policy crop cell (x, y) in robot frame — (M, 2)
        r, c = torch.arange(policy_h, dtype=torch.float32), torch.arange(policy_w, dtype=torch.float32)
        gr, gc = torch.meshgrid(r, c, indexing='ij')
        x = (gc - (policy_w - 1) / 2.0) * global_res + policy_cx
        y = (gr - (policy_h - 1) / 2.0) * global_res + policy_cy
        self.register_buffer('_policy_pts', torch.stack([x.flatten(), y.flatten()], dim=1))

        # Global map state buffers — (B, 1, Hg, Wg); not parameters
        self.register_buffer('global_elev', torch.zeros(B, 1, global_h, global_w))
        self.register_buffer('global_var',  torch.full((B, 1, global_h, global_w), self.INF_VAR))

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, batch_idx=None):
        """Reset global map to unobserved state.  Call at episode start."""
        if batch_idx is None:
            self.global_elev.zero_()
            self.global_var.fill_(self.INF_VAR)
        else:
            self.global_elev[batch_idx].zero_()
            self.global_var[batch_idx].fill_(self.INF_VAR)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_pts(pts: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """pts (B,N,2), yaw (B,) → rotated pts (B,N,2) via R(yaw)."""
        c, s = yaw.cos(), yaw.sin()
        R = torch.stack([torch.stack([c, -s], -1),
                         torch.stack([s,  c], -1)], -2)   # (B, 2, 2)
        return torch.bmm(pts, R.transpose(1, 2))

    def _to_world(self, pts_robot: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """pts_robot (B,N,2), poses (B,3=[x,y,yaw]) → world coords (B,N,2)."""
        pts_w = self._rotate_pts(pts_robot, poses[:, 2])
        pts_w = pts_w + poses[:, :2].unsqueeze(1)    # (B,1,2) broadcast
        return pts_w

    def _world_to_idx(self, pts_w: torch.Tensor):
        """
        World coords (B,N,2) → clamped integer grid indices gi,gj (B,N).

        COORDINATE CONVENTION (important):
          World origin (0, 0) maps to global grid center (global_h/2, global_w/2).
          This is correct because:
            • reset() is called at the start of each episode
            • poses passed to update()/crop() are episode-relative
              (robot spawns at world (0,0,0) at each episode reset)
          If absolute world coordinates were used, an origin offset would be needed.
        """
        gj = (pts_w[:, :, 0] / self.global_res + self.global_w / 2).long()
        gi = (pts_w[:, :, 1] / self.global_res + self.global_h / 2).long()
        return gi.clamp(0, self.global_h - 1), gj.clamp(0, self.global_w - 1)

    def _world_to_norm(self, pts_w: torch.Tensor) -> torch.Tensor:
        """World coords (B,N,2) → grid_sample normalised coords (B,N,2) ∈ [−1,1]."""
        nx = pts_w[:, :, 0] / self.global_res / (self.global_w / 2)
        ny = pts_w[:, :, 1] / self.global_res / (self.global_h / 2)
        return torch.stack([nx, ny], dim=-1)

    # ------------------------------------------------------------------
    # WTA update
    # ------------------------------------------------------------------

    def update(
        self,
        elev_local: torch.Tensor,
        log_var_local: torch.Tensor,
        poses: torch.Tensor,
    ):
        """
        Ingest one MappingNet output and update the global map via
        Probabilistic Winner-Take-All (Eq. 6–8, Sec. V-A).

        Eq.6:  σ̂²_t = max(σ²_t, 0.5 · σ²_prior)   — lower-bound by prior
        Valid: σ̂²_t < 1.5 · σ²_prior  OR  σ̂²_t < 0.2²
        Eq.7:  p_win = (1/σ̂²_t) / (1/σ̂²_t + 1/σ²_prior)
        Eq.8:  if ξ < p_win → overwrite cell with (h_t, σ̂²_t)

        Args:
            elev_local:    (B, 1, local_h, local_w)   MappingNet elevation
            log_var_local: (B, 1, local_h, local_w)   MappingNet log-variance
            poses:         (B, 3)  [x, y, yaw] robot pose in world frame (m, rad)
        """
        B         = elev_local.shape[0]
        var_local = log_var_local.exp()
        elev_flat = elev_local.reshape(B, -1)   # (B, N)
        var_flat  = var_local.reshape(B, -1)    # (B, N)

        pts   = self._local_pts.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        pts_w = self._to_world(pts, poses)
        gi, gj = self._world_to_idx(pts_w)                       # (B, N)

        for b in range(B):
            lin       = gi[b] * self.global_w + gj[b]   # (N,) linear global indices
            gvar_view = self.global_var[b, 0].view(-1)   # (Hg*Wg,) mutable view
            gelev_view = self.global_elev[b, 0].view(-1)

            sigma2_prior = gvar_view[lin]                 # (N,) prior variance at target cells
            sigma2_t     = var_flat[b]                    # (N,) new measurement variance

            # Eq. 6: effective measurement variance, lower-bounded by half the prior
            sigma2_eff = torch.max(sigma2_t, 0.5 * sigma2_prior)   # (N,)

            # Validity check (paper Sec. V-A):
            #   "valid only if σ̂² < 1.5·σ²_prior  OR  σ̂² < 0.2²"
            valid = (sigma2_eff < 1.5 * sigma2_prior) | (sigma2_eff < 0.04)  # 0.2² = 0.04

            valid_idx = valid.nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            # Gather valid candidates
            v_lin        = lin[valid_idx]
            v_sigma2_eff = sigma2_eff[valid_idx]
            v_sigma2_pr  = sigma2_prior[valid_idx]
            v_elev       = elev_flat[b][valid_idx]

            # Eq. 7: p_win = precision_new / (precision_new + precision_prior)
            prec_new   = 1.0 / (v_sigma2_eff + 1e-8)
            prec_prior = 1.0 / (v_sigma2_pr + 1e-8)
            p_win      = prec_new / (prec_new + prec_prior)           # (K,)

            # Eq. 8: stochastic overwrite — sample ξ ~ U[0,1]
            xi   = torch.rand_like(p_win)
            wins = (xi < p_win)                                       # (K,) bool

            win_idx = wins.nonzero(as_tuple=True)[0]
            if win_idx.numel() == 0:
                continue

            w_lin  = v_lin[win_idx]
            w_var  = v_sigma2_eff[win_idx]
            w_elev = v_elev[win_idx]

            # Sort descending by var → minimum-var wins last when indices collide
            order = w_var.argsort(descending=True)
            gvar_view.scatter_(0,  w_lin[order], w_var[order])
            gelev_view.scatter_(0, w_lin[order], w_elev[order])

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------

    @staticmethod
    def _surface_normals(elev: torch.Tensor, res: float) -> torch.Tensor:
        """
        Central-difference surface normals: n ∝ (−∂h/∂x, −∂h/∂y, 1).
        Returns (nx, ny) components.
        elev: (B, 1, H, W) → (B, 2, H, W)
        """
        p    = F.pad(elev, (1, 1, 1, 1), mode='replicate')   # (B,1,H+2,W+2)
        dhdx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) / (2.0 * res)
        dhdy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) / (2.0 * res)
        return torch.cat([-dhdx, -dhdy], dim=1)               # (B, 2, H, W)

    def crop(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Bilinear sample of the policy-size window from the accumulated global map.

        Args:
            poses: (B, 3)  [x, y, yaw]
        Returns: (B, 4, policy_h, policy_w)
            Channels: [elevation, normal_x, normal_y, variance]  ← d_map_student=4
        """
        B     = poses.shape[0]
        pts   = self._policy_pts.unsqueeze(0).expand(B, -1, -1)   # (B, M, 2)
        pts_w = self._to_world(pts, poses)
        grid  = self._world_to_norm(pts_w).reshape(
                    B, self.policy_h, self.policy_w, 2)            # (B, Ph, Pw, 2)

        src     = torch.cat([self.global_elev, self.global_var], 1)  # (B, 2, Hg, Wg)
        sampled = F.grid_sample(src, grid, mode='bilinear',
                                align_corners=True,
                                padding_mode='border')               # (B, 2, Ph, Pw)

        elev_c  = sampled[:, 0:1]                                    # (B, 1, Ph, Pw)
        var_c   = sampled[:, 1:2]
        normals = self._surface_normals(elev_c, self.global_res)     # (B, 2, Ph, Pw)

        # [elev, n_x, n_y, var] matches teacher layout (3 ch) + uncertainty (1 ch)
        return torch.cat([elev_c, normals, var_c], dim=1)            # (B, 4, Ph, Pw)


# ---------------------------------------------------------------------------
# Part 2: AME-2 Encoder  (Fig. 3 left-bottom)
# ---------------------------------------------------------------------------

class CoordPositionalEmbedding(nn.Module):
    """
    Positional embedding via MLP over each cell's 2-D grid coordinates.
    Paper (Sec.IV-A, Fig.3): "computes a positional embedding for each point with an MLP."
    Coordinates normalized to [-1, 1] over the map extent. [inferred details]
    """

    def __init__(self, map_h: int, map_w: int, d_pos_emb: int):
        super().__init__()
        # Pre-compute normalized (x, y) coords; registered as buffer (not param)
        ys = torch.linspace(-1, 1, map_h)
        xs = torch.linspace(-1, 1, map_w)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        coords = torch.stack([grid_x, grid_y], dim=-1)           # (H, W, 2)
        self.register_buffer('coords', coords)

        self.mlp = nn.Sequential(
            nn.Linear(2, d_pos_emb), nn.ELU(inplace=True),
            nn.Linear(d_pos_emb, d_pos_emb),
        )

    def forward(self, B: int) -> torch.Tensor:
        """
        Returns: (B, d_pos_emb, H, W) positional embeddings for each map cell.
        """
        pe = self.mlp(self.coords)            # (H, W, d_pos_emb)
        pe = pe.permute(2, 0, 1).unsqueeze(0) # (1, d_pos_emb, H, W)
        return pe.expand(B, -1, -1, -1)       # (B, d_pos_emb, H, W)


class AME2Encoder(nn.Module):
    """
    Attention-Based Map Encoder (Fig. 3, left-bottom).

    Three-step local feature extraction (Sec.IV-A):
      1. CNN(map)          → local features (per-point)
      2. MLP(coords)       → positional embedding (per-point)
      3. MLP(local + pe)   → pointwise local features  [K/V for attention]

    Full data flow:
      map (B, d_map, H, W)
        → local_cnn                    → (B, d_local, H, W)    [step 1]
        → CoordPositionalEmbedding     → (B, d_pos_emb, H, W) [step 2]
        → local_fusion MLP(local||pe)  → Pointwise Local Feats (B, d_local, H, W) [step 3]
        → global_mlp + MaxPool         → Global Features (B, d_global)
        → query_proj(global || prop_emb) → Query (B, 1, d_local)
        → MHA(Q, K=pointwise, V=pointwise) → Weighted Local (B, d_local)
        → cat(weighted_local, global_feat) → Map Embedding (B, d_map_emb)
    """

    def __init__(self, cfg: PolicyConfig, d_map: int):
        """
        d_map: map input channels (3 for teacher, 4 for student).
               Fixed at construction — no dynamic creation in forward().
        """
        super().__init__()
        self.cfg = cfg

        # Step 1: CNN on raw map input (d_map channels)
        # Paper: "first extracts local map features with a CNN"
        self.local_cnn = nn.Sequential(
            nn.Conv2d(d_map, cfg.d_local, 3, padding=1), nn.ELU(inplace=True),
            nn.Conv2d(cfg.d_local, cfg.d_local, 1),      nn.ELU(inplace=True),
        )

        # Step 2: Positional embedding (MLP over grid coordinates)
        # Paper: "computes a positional embedding for each point with an MLP"
        self.pos_emb = CoordPositionalEmbedding(cfg.map_h, cfg.map_w, cfg.d_pos_emb)

        # Step 3: Fusion MLP — fuses CNN local features + positional embedding
        # Paper: "These are then fused by another MLP to obtain pointwise local features"
        self.local_fusion = nn.Sequential(
            nn.Linear(cfg.d_local + cfg.d_pos_emb, cfg.d_local), nn.ELU(inplace=True),
        )

        # Global branch: "an additional MLP followed by max pooling"
        self.global_mlp = nn.Sequential(
            nn.Linear(cfg.d_local, cfg.d_global), nn.ELU(inplace=True),
            nn.Linear(cfg.d_global, cfg.d_global),
        )

        # Query projection: "combine global features with proprioceptive embedding through an MLP"
        self.query_proj = nn.Sequential(
            nn.Linear(cfg.d_global + cfg.d_prop_emb, cfg.d_local),
            nn.ELU(inplace=True),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_local,
            num_heads=cfg.num_heads,
            batch_first=True,
        )

        assert cfg.d_map_emb == cfg.d_local + cfg.d_global, \
            f"d_map_emb ({cfg.d_map_emb}) must equal d_local+d_global ({cfg.d_local+cfg.d_global})"

    def forward(self, map_feat: torch.Tensor, prop_emb: torch.Tensor):
        """
        map_feat: (B, d_map, H, W)
        prop_emb: (B, d_prop_emb)
        returns:  (B, d_map_emb)
        """
        B, C, H, W = map_feat.shape

        # Step 1: CNN on raw map → initial local features
        local_raw = self.local_cnn(map_feat)       # (B, d_local, H, W)

        # Step 2: Positional embedding per map cell
        pe = self.pos_emb(B)                       # (B, d_pos_emb, H, W)

        # Step 3: Fusion MLP → pointwise local features (these become K and V)
        lr_flat = local_raw.permute(0, 2, 3, 1).reshape(B * H * W, -1)  # (N, d_local)
        pe_flat = pe.permute(0, 2, 3, 1).reshape(B * H * W, -1)         # (N, d_pos_emb)
        pointwise = self.local_fusion(torch.cat([lr_flat, pe_flat], dim=-1))  # (N, d_local)
        local_feat = pointwise.reshape(B, H, W, -1).permute(0, 3, 1, 2) # (B, d_local, H, W)

        # Global features: MLP on pointwise features → spatial MaxPool
        gf_flat     = self.global_mlp(pointwise)                        # (N, d_global)
        gf          = gf_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2) # (B, d_global, H, W)
        global_feat = gf.flatten(2).max(dim=-1).values                  # (B, d_global)

        # Query: combine global + proprioception embedding
        q_in         = torch.cat([global_feat, prop_emb], dim=-1)       # (B, d_g+d_p)
        q            = self.query_proj(q_in).unsqueeze(1)               # (B, 1, d_local)

        # MHA: pointwise local features as K and V
        kv           = local_feat.flatten(2).permute(0, 2, 1)          # (B, H*W, d_local)
        weighted, _  = self.attn(q, kv, kv)
        weighted_local = weighted.squeeze(1)                            # (B, d_local)

        # Map embedding: concat weighted local + global
        return torch.cat([weighted_local, global_feat], dim=-1)         # (B, d_map_emb)


# ---------------------------------------------------------------------------
# Part 3: Proprioception Encoders  (Fig. 3 right)
# ---------------------------------------------------------------------------

class TeacherPropEncoder(nn.Module):
    """
    Teacher: plain MLP over all ground-truth proprioception observations.
    Observations include base_lin_vel (available to teacher only).

    Used for both the actor prop (48D) and the critic prop (50D).
    Pass ``d_prop_in`` to override the input dimension when using the
    critic's extended command layout.
    """

    def __init__(self, cfg: PolicyConfig, d_prop_in: int | None = None):
        super().__init__()
        d_in = d_prop_in if d_prop_in is not None else cfg.d_prop_raw
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 256), nn.ELU(inplace=True),
            nn.Linear(256, cfg.d_prop_emb), nn.ELU(inplace=True),
        )

    def forward(self, prop: torch.Tensor):
        """prop: (B, d_prop_in)  → (B, d_prop_emb)"""
        return self.mlp(prop)


class LSIO(nn.Module):
    """
    Long-Short I/O encoder (Li et al., IJRR 2024 [66]).

    Dual-history 1D CNN architecture (Sec. V + Fig. 3 of Li et al.):
      • Short history: last SHORT_LEN=4 steps directly flattened
      • Long  history: all T steps through 1D CNN temporal encoder

    1D CNN config (stated in Li et al. Sec.V):
      Layer 1: Conv1d(d_hist, 32, kernel_size=6, stride=3, no padding) + ReLU
      Layer 2: Conv1d(32,     16, kernel_size=4, stride=2, no padding) + ReLU

    AME-2 adaptation (T=20 vs. 66 in original paper):
      T1 = floor((T-6)/3)+1  = floor(14/3)+1  = 5
      T2 = floor((T1-4)/2)+1 = floor((5-4)/2)+1 = 1
      long_flat  = 16 × T2         = 16
      short_flat = SHORT_LEN × d_hist = 4 × 42 = 168
      out_size   = 168 + 16        = 184
    """

    SHORT_LEN: int = 4  # from Li et al. Sec.V

    def __init__(self, d_hist: int, T: int = 20):
        super().__init__()
        self.short_len = self.SHORT_LEN

        # 1D CNN for long history (architecture stated in Li et al.)
        self.long_cnn = nn.Sequential(
            nn.Conv1d(d_hist, 32, kernel_size=6, stride=3),  # no padding
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=4, stride=2),      # no padding
            nn.ReLU(inplace=True),
        )

        # Pre-compute flattened output sizes
        T1 = (T - 6) // 3 + 1
        T2 = (T1 - 4) // 2 + 1
        self._long_flat  = 16 * T2
        self._short_flat = self.SHORT_LEN * d_hist
        self.out_size    = self._short_flat + self._long_flat

    def forward(self, prop_hist: torch.Tensor) -> torch.Tensor:
        """
        prop_hist: (B, T, d_hist)
        returns:   (B, out_size)   [short_flat || long_cnn_flat]
        """
        B = prop_hist.shape[0]

        # Short: last SHORT_LEN steps, directly flattened
        short = prop_hist[:, -self.short_len:, :].reshape(B, -1)  # (B, short_len*d_hist)

        # Long: Conv1d expects (B, C, T) → permute time and channels
        long_in  = prop_hist.permute(0, 2, 1)   # (B, d_hist, T)
        long_emb = self.long_cnn(long_in)        # (B, 16, T2)
        long_flat = long_emb.reshape(B, -1)      # (B, 16*T2)

        return torch.cat([short, long_flat], dim=-1)   # (B, out_size)


class StudentPropEncoder(nn.Module):
    """
    Student proprioception encoder (Fig. 3 right, Sec.IV-A).

    Paper: LSIO processes stacked history (WITHOUT base_lin_vel and commands),
    then commands are concatenated with the temporal embedding and passed
    through a final MLP to produce the proprioception embedding.

    FIX: commands are now separate from LSIO history input.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.lsio = LSIO(cfg.d_hist, T=cfg.prop_history)
        # LSIO output (out_size=184) + commands (3) → proprioception embedding (128)
        self.out_mlp = nn.Sequential(
            nn.Linear(self.lsio.out_size + cfg.d_commands, cfg.d_prop_emb),
            nn.ELU(inplace=True),
        )

    def forward(self, prop_hist: torch.Tensor, commands: torch.Tensor):
        """
        prop_hist: (B, T=20, d_hist)  history WITHOUT base_lin_vel or commands
        commands:  (B, d_commands)    current goal commands
        returns:   (B, d_prop_emb)
        """
        temporal_feat = self.lsio(prop_hist)                            # (B, LSIO.out_size=184)
        return self.out_mlp(torch.cat([temporal_feat, commands], dim=-1))


# ---------------------------------------------------------------------------
# Part 4: Full Policy  (Fig. 3 left-top + Fig. 2 overview)
# ---------------------------------------------------------------------------

class AME2Policy(nn.Module):
    """
    Complete AME-2 locomotion policy.

    Teacher inputs:
        map_feat:  (B, 3,  H, W)   ground-truth elevation map
        prop:      (B, d_prop_raw)  full proprioception (inc. base_lin_vel)

    Student inputs:
        map_feat:  (B, 4,  H, W)   neural map (elevation + uncertainty)
        prop_hist: (B, T=20, d_hist) stacked history (no base_lin_vel/cmds)
        commands:  (B, d_commands)  goal position + heading

    Output:
        actions:   (B, num_joints)  joint PD targets (tracked at 400 Hz)
    Policy runs at 50 Hz.
    """

    def __init__(self, cfg: PolicyConfig, is_student: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_student = is_student

        d_map = cfg.d_map_student if is_student else cfg.d_map_teacher
        self.prop_encoder = (StudentPropEncoder(cfg)
                             if is_student else TeacherPropEncoder(cfg))
        # Each policy gets its own encoder with fixed d_map  [FIX]
        self.map_encoder = AME2Encoder(cfg, d_map=d_map)

        d_in = cfg.d_map_emb + cfg.d_prop_emb
        self.decoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.ELU(inplace=True),
            nn.Linear(512, 256),  nn.ELU(inplace=True),
            nn.Linear(256, cfg.num_joints),
        )

    def forward(self, map_feat: torch.Tensor, prop=None,
                prop_hist=None, commands=None):
        """
        Teacher: forward(map_feat, prop=prop_tensor)
        Student: forward(map_feat, prop_hist=hist_tensor, commands=cmd_tensor)

        Returns: (actions, map_emb, prop_emb)
          - actions:   (B, num_joints)   joint PD targets
          - map_emb:   (B, d_map_emb)   AME-2 encoder output   ← needed for L_repr
          - prop_emb:  (B, d_prop_emb)  proprioception embedding
        """
        if self.is_student:
            assert prop_hist is not None and commands is not None
            prop_emb = self.prop_encoder(prop_hist, commands)
        else:
            assert prop is not None
            prop_emb = self.prop_encoder(prop)

        map_emb = self.map_encoder(map_feat, prop_emb)
        actions = self.decoder(torch.cat([map_emb, prop_emb], dim=-1))
        return actions, map_emb, prop_emb


# ---------------------------------------------------------------------------
# Part 5: Teacher-Student training objective  (Sec. IV-C, Table VI)
# ---------------------------------------------------------------------------

class StudentLoss(nn.Module):
    """
    Student training loss (Sec. IV-C):

        L = L_PPO  +  λ_dist * L_distill  +  λ_repr * L_repr

    Coefficients from Table VI (stated in paper):
        λ_dist = 0.02   (action distillation)
        λ_repr = 0.2    (map embedding MSE)

    PPO schedule (stated in paper):
        First 5000 iterations: PPO surrogate loss disabled (ppo_loss=0),
        large learning rate = 0.001.
        Remaining 35000 iterations: PPO enabled, adaptive LR.
    Total student training: 40000 iterations.

    FIX: corrected default coefficients from 1.0 → 0.02 / 0.2.
    """

    def __init__(self, lam_dist: float = 0.02, lam_repr: float = 0.2):
        super().__init__()
        self.lam_dist = lam_dist
        self.lam_repr = lam_repr

    def forward(self, ppo_loss, student_actions, teacher_actions,
                student_map_emb, teacher_map_emb):
        l_dist = F.mse_loss(student_actions, teacher_actions.detach())
        l_repr = F.mse_loss(student_map_emb, teacher_map_emb.detach())
        return ppo_loss + self.lam_dist * l_dist + self.lam_repr * l_repr


# ---------------------------------------------------------------------------
# Part 6: Asymmetric MoE Critic  (Sec. IV-B — training only)
# ---------------------------------------------------------------------------

class AsymmetricCritic(nn.Module):
    """
    Asymmetric Mixture-of-Experts critic for PPO value estimation (Sec. IV-B).
    Training-only — not used during inference.

    "Asymmetric": privileged inputs unavailable to the student policy:
        • ground-truth elevation map  (d_map_teacher=3 channels)
        • full proprioception incl. base_lin_vel (d_prop_raw=48)
        • per-foot contact states  (d_contact, stated in paper)

    "MoE design" (stated in paper):
        Gate network  :  f(contact_states) → weights (B, N_experts)
        Expert MLPs   :  each maps state embedding → scalar value
        Output        :  V = Σ_i gate_i · expert_i(state)

    Rationale for contact-gated MoE:
        Contact pattern encodes the robot's locomotion mode (all 4 feet down,
        3-foot stance, 2-foot transition, …).  Different modes have very
        different value landscapes, so mode-specialised experts + soft gating
        produce a more accurate V(state) than a single monolithic MLP.

    Architecture details:
        N_experts = 4      [inferred — not stated]
        d_contact = 4      [inferred — one binary state per foot, ANYmal-D]
        d_hidden  = 256    [inferred]
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        N_experts: int = 4,    # [inferred]
        d_contact:  int = 4,   # [inferred] per-foot contact (ANYmal-D: 4 feet)
        d_hidden:   int = 256,  # [inferred]
    ):
        super().__init__()
        self.N = N_experts

        # Encoder: same architecture as actor, separate weights (asymmetric inputs).
        # Critic receives a WIDER prop (d_prop_critic=50D) with the full 5D command
        # instead of the actor's clipped 3D command.  [stated Sec.IV-E.3]
        self.map_encoder  = AME2Encoder(cfg, d_map=cfg.d_map_teacher)
        self.prop_encoder = TeacherPropEncoder(cfg, d_prop_in=cfg.d_prop_critic)

        d_state = cfg.d_map_emb + cfg.d_prop_emb   # 192 + 128 = 320

        # Gate: contact pattern → soft expert assignment
        # Contact states encode locomotion mode → natural gating signal
        self.gate = nn.Sequential(
            nn.Linear(d_contact, 64), nn.ELU(inplace=True),
            nn.Linear(64, N_experts),
            nn.Softmax(dim=-1),
        )

        # Expert MLPs: mode-specialised value estimators
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_state,      d_hidden),      nn.ELU(inplace=True),
                nn.Linear(d_hidden,     d_hidden // 2), nn.ELU(inplace=True),
                nn.Linear(d_hidden // 2, 1),
            )
            for _ in range(N_experts)
        ])

    def forward(
        self,
        map_feat:       torch.Tensor,   # (B, d_map_teacher, H, W)  ground-truth
        prop:           torch.Tensor,   # (B, d_prop_raw)  incl. base_lin_vel
        contact_states: torch.Tensor,   # (B, d_contact)   per-foot contact
    ) -> torch.Tensor:
        """
        Returns value estimate: (B, 1).
        """
        prop_emb = self.prop_encoder(prop)
        map_emb  = self.map_encoder(map_feat, prop_emb)
        state    = torch.cat([map_emb, prop_emb], dim=-1)   # (B, d_state)

        weights     = self.gate(contact_states)              # (B, N) — sums to 1
        assert not torch.isnan(weights).any(), \
            "AsymmetricCritic gate produced NaN — check contact_states for NaN/Inf"
        expert_vals = torch.stack(
            [e(state) for e in self.experts], dim=1
        )                                                    # (B, N, 1)

        # Soft mixture: Σ gate_i · expert_i
        return (weights.unsqueeze(-1) * expert_vals).sum(dim=1)  # (B, 1)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PolicyConfig()
    B, H, W = 4, cfg.map_h, cfg.map_w
    T = cfg.prop_history

    print("=== MappingNet ===")
    map_cfg = MappingConfig()
    mapping_net = MappingNet(map_cfg)
    raw_grid = torch.randn(B, 1, map_cfg.map_h, map_cfg.map_w)
    elev, log_var = mapping_net(raw_grid)
    print(f"  input:   {raw_grid.shape}")
    print(f"  elev:    {elev.shape}")
    print(f"  log_var: {log_var.shape}")

    gt = torch.randn_like(elev)
    tv_w = MappingNet.total_variation_weight(gt)
    loss = MappingNet.beta_nll_loss(elev, log_var, gt, beta=0.5, tv_weights=tv_w)
    print(f"  β-NLL loss: {loss.item():.4f}")

    print("\n=== Teacher Policy ===")
    teacher = AME2Policy(cfg, is_student=False)
    map_t  = torch.randn(B, cfg.d_map_teacher, H, W)
    prop_t = torch.randn(B, cfg.d_prop_raw)
    acts_t, map_emb_t, prop_emb_t = teacher(map_t, prop=prop_t)
    print(f"  map:      {map_t.shape}")
    print(f"  prop:     {prop_t.shape}")
    print(f"  actions:  {acts_t.shape}")
    print(f"  map_emb:  {map_emb_t.shape}")
    print(f"  prop_emb: {prop_emb_t.shape}")

    print("\n=== Student Policy ===")
    student   = AME2Policy(cfg, is_student=True)
    map_s     = torch.randn(B, cfg.d_map_student, H, W)
    prop_hist = torch.randn(B, T, cfg.d_hist)
    commands  = torch.randn(B, cfg.d_commands)
    acts_s, map_emb_s, prop_emb_s = student(map_s, prop_hist=prop_hist, commands=commands)
    print(f"  map:      {map_s.shape}")
    print(f"  history:  {prop_hist.shape}  (T={T}, d_hist={cfg.d_hist})")
    print(f"  commands: {commands.shape}")
    print(f"  actions:  {acts_s.shape}")
    print(f"  map_emb:  {map_emb_s.shape}")

    print("\n=== WTA Map Fusion  (MappingNet → WTA → Student Policy) ===")
    wta = WTAMapFusion(B)
    wta.reset()
    # Simulate 5 steps: robot advances forward, no yaw change
    for step in range(5):
        poses = torch.zeros(B, 3)
        poses[:, 0] = step * 0.3     # x advances 0.3 m/step
        raw_step = torch.randn(B, 1, map_cfg.map_h, map_cfg.map_w)
        elev_step, lv_step = mapping_net(raw_step)
        wta.update(elev_step, lv_step, poses)
    final_poses = torch.zeros(B, 3); final_poses[:, 0] = 5 * 0.3
    policy_map  = wta.crop(final_poses)
    print(f"  global_elev : {wta.global_elev.shape}")
    print(f"  policy_map  : {policy_map.shape}   (→ student policy input)")
    assert policy_map.shape == (B, cfg.d_map_student, cfg.map_h, cfg.map_w)
    # Feed WTA-produced map to student policy
    acts_wta, map_wta, _ = student(policy_map, prop_hist=prop_hist, commands=commands)
    print(f"  student acts: {acts_wta.shape}  ← end-to-end pipeline OK")

    print("\n=== Student Loss ===")
    loss_fn = StudentLoss()
    # Use real map embeddings from both policies (map dims differ: teacher d_map=3, student d_map=4,
    # but both output d_map_emb=192 from AME2Encoder)
    total = loss_fn(
        ppo_loss        = torch.tensor(0.1),
        student_actions = acts_s,
        teacher_actions = acts_t,
        student_map_emb = map_emb_s,
        teacher_map_emb = map_emb_t,
    )
    print(f"  lam_dist={loss_fn.lam_dist}, lam_repr={loss_fn.lam_repr}")
    print(f"  total student loss: {total.item():.4f}")

    print("\n=== Asymmetric MoE Critic ===")
    critic = AsymmetricCritic(cfg)
    # Critic receives 50D prop: base(45D) + critic_cmd(5D: x_rel,y_rel,sin,cos,t_remain)
    prop_critic = torch.randn(B, cfg.d_prop_critic)
    contact = torch.zeros(B, 4)   # (B, d_contact) — 4 foot contacts
    contact[:, :2] = 1.0          # front two feet on ground
    value = critic(map_t, prop_critic, contact)
    print(f"  map:          {map_t.shape}")
    print(f"  critic_prop:  {prop_critic.shape}  (50D = base_45 + critic_cmd_5)")
    print(f"  contact:      {contact.shape}")
    print(f"  value output: {value.shape}")
    assert value.shape == (B, 1)
    assert prop_critic.shape[-1] == cfg.d_prop_critic == 50
    # Gate weights should be a valid probability distribution
    with torch.no_grad():
        gate_w = critic.gate(contact)
    assert gate_w.shape == (B, critic.N)
    assert torch.allclose(gate_w.sum(dim=-1), torch.ones(B), atol=1e-5)
    print(f"  gate weights: {gate_w[0].tolist()}")  # show expert assignment

    print("\n=== TRON1 config ===")
    t1_map_cfg = tron1_mapping_cfg()
    t1_pol_cfg = tron1_policy_cfg()
    t1_mapping = MappingNet(t1_map_cfg)
    t1_raw     = torch.randn(B, 1, t1_map_cfg.map_h, t1_map_cfg.map_w)
    t1_elev, t1_lv = t1_mapping(t1_raw)
    print(f"  TRON1 local grid:   {t1_raw.shape}")
    print(f"  TRON1 policy map:   {t1_pol_cfg.map_h}×{t1_pol_cfg.map_w}")
    t1_wta = WTAMapFusion(B, **TRON1_WTA_KWARGS)
    t1_wta.reset()
    t1_poses = torch.zeros(B, 3)
    t1_wta.update(t1_elev, t1_lv, t1_poses)
    t1_pmap = t1_wta.crop(t1_poses)
    print(f"  TRON1 policy_map:   {t1_pmap.shape}")
    assert t1_pmap.shape == (B, t1_pol_cfg.d_map_student, t1_pol_cfg.map_h, t1_pol_cfg.map_w)
    t1_student = AME2Policy(t1_pol_cfg, is_student=True)
    t1_hist = torch.randn(B, T, t1_pol_cfg.d_hist)
    t1_cmd  = torch.randn(B, t1_pol_cfg.d_commands)
    t1_acts, _, _ = t1_student(t1_pmap, prop_hist=t1_hist, commands=t1_cmd)
    print(f"  TRON1 actions:      {t1_acts.shape}")

    print("\n=== Parameter count ===")
    print(f"  MappingNet (ANYmal-D):   {sum(p.numel() for p in mapping_net.parameters()):,}")
    print(f"  Teacher policy:          {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student policy:          {sum(p.numel() for p in student.parameters()):,}")
    print(f"  AsymmetricCritic:        {sum(p.numel() for p in critic.parameters()):,}")
    print(f"  MappingNet (TRON1):      {sum(p.numel() for p in t1_mapping.parameters()):,}")
    print(f"  Student policy (TRON1):  {sum(p.numel() for p in t1_student.parameters()):,}")

    print("\nAll checks passed.")
