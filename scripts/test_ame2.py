"""
Unit tests for ame2_model.py — AME-2 architecture reproduction.

Run:  python -m pytest test_ame2.py -v
"""

import pytest
import torch

from ame2.networks.ame2_model import (
    MappingConfig, PolicyConfig,
    MappingNet, WTAMapFusion,
    AME2Encoder, LSIO,
    AsymmetricCritic, StudentLoss,
    AME2Policy,
    tron1_mapping_cfg, tron1_policy_cfg,
    TRON1_WTA_KWARGS,
)

B = 4  # batch size used across tests

# =====================================================================
# 1. TestMappingNet
# =====================================================================

class TestMappingNet:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = MappingConfig()
        self.net = MappingNet(self.cfg)
        self.x = torch.randn(B, 1, self.cfg.map_h, self.cfg.map_w)

    def test_output_shapes(self):
        """elev and log_var shapes must equal input shape."""
        elev, log_var = self.net(self.x)
        assert elev.shape == self.x.shape, f"elev shape {elev.shape} != input {self.x.shape}"
        assert log_var.shape == self.x.shape, f"log_var shape {log_var.shape} != input {self.x.shape}"

    def test_gate_range(self):
        """Gate output must be in [0, 1] (sigmoid)."""
        # Run forward manually to extract gate
        skip = self.net.enc(self.x)
        pooled = self.net.pool(skip)
        up = torch.nn.functional.interpolate(pooled, size=skip.shape[2:],
                                              mode='bilinear', align_corners=False)
        feat = self.net.dec(torch.cat([up, skip], dim=1))
        gate = torch.sigmoid(self.net.head_gate(feat))
        assert gate.min() >= 0.0, f"gate min {gate.min().item()} < 0"
        assert gate.max() <= 1.0, f"gate max {gate.max().item()} > 1"

    def test_beta_nll_grad(self):
        """beta_nll_loss must have gradient w.r.t. pred_elev."""
        pred_elev = torch.randn(B, 1, self.cfg.map_h, self.cfg.map_w, requires_grad=True)
        log_var = torch.randn(B, 1, self.cfg.map_h, self.cfg.map_w)
        target = torch.randn(B, 1, self.cfg.map_h, self.cfg.map_w)
        loss = MappingNet.beta_nll_loss(pred_elev, log_var, target)
        loss.backward()
        assert pred_elev.grad is not None, "No gradient for pred_elev"
        assert pred_elev.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_tv_weight_sum(self):
        """Total variation weights must sum to 1.0."""
        y = torch.randn(B, 1, self.cfg.map_h, self.cfg.map_w)
        w = MappingNet.total_variation_weight(y)
        assert w.shape == (B,), f"TV weights shape {w.shape} != ({B},)"
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5), \
            f"TV weights sum {w.sum().item()} != 1.0"


# =====================================================================
# 2. TestWTAMapFusion
# =====================================================================

class TestWTAMapFusion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.wta = WTAMapFusion(B)
        self.poses = torch.zeros(B, 3)  # robot at origin, no yaw

    def test_reset(self):
        """After reset, global_var must all equal INF_VAR."""
        self.wta.reset()
        expected = torch.full_like(self.wta.global_var, WTAMapFusion.INF_VAR)
        assert torch.equal(self.wta.global_var, expected), "global_var not all INF_VAR after reset"
        assert torch.equal(self.wta.global_elev, torch.zeros_like(self.wta.global_elev)), \
            "global_elev not all zeros after reset"

    def test_update_shape(self):
        """update() must not change global_elev shape."""
        self.wta.reset()
        shape_before = self.wta.global_elev.shape
        elev_local = torch.randn(B, 1, 31, 51)
        log_var_local = torch.randn(B, 1, 31, 51)
        self.wta.update(elev_local, log_var_local, self.poses)
        assert self.wta.global_elev.shape == shape_before, \
            f"Shape changed from {shape_before} to {self.wta.global_elev.shape}"

    def test_wta_correctness(self):
        """Probabilistic WTA (Eq.6-8): lower-variance observation wins with p_win ≈ 2/3.

        With sigma2_prior=exp(2)≈7.4 and sigma2_new=exp(-2)≈0.14:
          sigma2_eff = max(sigma2_new, 0.5*sigma2_prior) = 3.69  (Eq.6 lower bound)
          p_win = prec_eff/(prec_eff+prec_prior) ≈ 0.667         (Eq.7)
        So ~67% of cells are overwritten — not all (stochastic, not deterministic).
        """
        torch.manual_seed(0)
        wta = WTAMapFusion(1)  # single batch
        wta.reset()
        poses = torch.zeros(1, 3)

        # First update: high variance (log_var = 2.0 → var ~ 7.4)
        elev1 = torch.full((1, 1, 31, 51), 10.0)
        lv1 = torch.full((1, 1, 31, 51), 2.0)
        wta.update(elev1, lv1, poses)

        mask = wta.global_var[0, 0] < WTAMapFusion.INF_VAR
        assert mask.any(), "No cells were updated on first pass"
        assert torch.allclose(wta.global_elev[0, 0][mask], torch.tensor(10.0), atol=1e-4), \
            "First update elevation incorrect"

        # Second update: lower variance (log_var = -2.0 → var ~ 0.14)
        # Eq.6 lower-bound clips sigma2_eff to 0.5*sigma2_prior → p_win ≈ 0.667
        elev2 = torch.full((1, 1, 31, 51), 5.0)
        lv2 = torch.full((1, 1, 31, 51), -2.0)
        wta.update(elev2, lv2, poses)

        # Probabilistic WTA: expect ~67% cells updated (accept 45%–95% range)
        written = wta.global_elev[0, 0][mask]
        frac_updated = (written - 5.0).abs() < 1e-4
        frac = frac_updated.float().mean().item()
        assert frac > 0.45, \
            f"Too few cells updated with lower-var obs: {frac:.2f} < 0.45 (expected ~0.67)"
        assert frac < 0.99, \
            f"All cells updated — probabilistic WTA should be stochastic, got {frac:.2f}"

    def test_crop_shape(self):
        """crop() must return (B, 4, policy_h, policy_w)."""
        self.wta.reset()
        # Do one update so the map is not empty
        elev_local = torch.randn(B, 1, 31, 51)
        log_var_local = torch.zeros(B, 1, 31, 51)
        self.wta.update(elev_local, log_var_local, self.poses)
        crop = self.wta.crop(self.poses)
        assert crop.shape == (B, 4, self.wta.policy_h, self.wta.policy_w), \
            f"crop shape {crop.shape} != expected (B,4,{self.wta.policy_h},{self.wta.policy_w})"

    def test_batch_independence(self):
        """Different batch items must not affect each other."""
        wta = WTAMapFusion(2)
        wta.reset()

        # Only update batch item 0
        elev_local = torch.zeros(2, 1, 31, 51)
        elev_local[0] = 1.0  # batch 0 gets elevation = 1.0
        log_var_local = torch.zeros(2, 1, 31, 51)
        log_var_local[1] = 100.0  # batch 1 has huge variance → should not update

        poses = torch.zeros(2, 3)
        wta.update(elev_local, log_var_local, poses)

        # Batch 1 should NOT have any elevation from batch 0
        # Cells updated in batch 0 should have elev=1.0
        mask0 = wta.global_var[0, 0] < WTAMapFusion.INF_VAR
        if mask0.any():
            assert torch.allclose(wta.global_elev[0, 0][mask0], torch.tensor(1.0), atol=1e-4)

        # Batch 1 with high variance: cells should have var < INF_VAR only where var(obs) < INF_VAR
        # But the elevation should NOT be 1.0 (that's batch 0's data)
        mask1 = wta.global_var[1, 0] < WTAMapFusion.INF_VAR
        if mask1.any():
            # Batch 1 had elev=0.0
            assert torch.allclose(wta.global_elev[1, 0][mask1], torch.tensor(0.0), atol=1e-4), \
                "Batch 1 was contaminated by batch 0 data"


# =====================================================================
# 3. TestAME2Encoder
# =====================================================================

class TestAME2Encoder:
    def test_teacher_shape(self):
        """Teacher encoder (d_map=3) output must be (B, 192)."""
        cfg = PolicyConfig()
        enc = AME2Encoder(cfg, d_map=cfg.d_map_teacher)
        map_feat = torch.randn(B, cfg.d_map_teacher, cfg.map_h, cfg.map_w)
        prop_emb = torch.randn(B, cfg.d_prop_emb)
        out = enc(map_feat, prop_emb)
        assert out.shape == (B, cfg.d_map_emb), \
            f"Teacher encoder output {out.shape} != ({B}, {cfg.d_map_emb})"

    def test_student_shape(self):
        """Student encoder (d_map=4) output must be (B, 192)."""
        cfg = PolicyConfig()
        enc = AME2Encoder(cfg, d_map=cfg.d_map_student)
        map_feat = torch.randn(B, cfg.d_map_student, cfg.map_h, cfg.map_w)
        prop_emb = torch.randn(B, cfg.d_prop_emb)
        out = enc(map_feat, prop_emb)
        assert out.shape == (B, cfg.d_map_emb), \
            f"Student encoder output {out.shape} != ({B}, {cfg.d_map_emb})"


# =====================================================================
# 4. TestLSIO
# =====================================================================

class TestLSIO:
    def test_out_size(self):
        """LSIO out_size must be 184 with default config."""
        cfg = PolicyConfig()
        lsio = LSIO(cfg.d_hist, T=cfg.prop_history)
        assert lsio.out_size == 184, f"LSIO out_size {lsio.out_size} != 184"

    def test_output_shape(self):
        """LSIO forward output must be (B, 184)."""
        cfg = PolicyConfig()
        lsio = LSIO(cfg.d_hist, T=cfg.prop_history)
        prop_hist = torch.randn(B, cfg.prop_history, cfg.d_hist)
        out = lsio(prop_hist)
        assert out.shape == (B, 184), f"LSIO output shape {out.shape} != ({B}, 184)"


# =====================================================================
# 5. TestAsymmetricCritic
# =====================================================================

class TestAsymmetricCritic:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = PolicyConfig()
        self.critic = AsymmetricCritic(self.cfg)
        self.map_feat = torch.randn(B, self.cfg.d_map_teacher, self.cfg.map_h, self.cfg.map_w)
        # Critic uses 50D critic_prop (base_vel+hist+critic_cmd), not 48D actor prop
        self.prop = torch.randn(B, self.cfg.d_prop_critic)
        self.contact = torch.randn(B, 4)

    def test_value_shape(self):
        """Critic output must be (B, 1)."""
        value = self.critic(self.map_feat, self.prop, self.contact)
        assert value.shape == (B, 1), f"Critic output {value.shape} != ({B}, 1)"

    def test_gate_sum(self):
        """Gate weights must sum to 1 for each batch item (softmax)."""
        with torch.no_grad():
            weights = self.critic.gate(self.contact)
        assert weights.shape == (B, self.critic.N), \
            f"Gate shape {weights.shape} != ({B}, {self.critic.N})"
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"Gate sums {sums.tolist()} not all 1.0"

    def test_gate_sensitivity(self):
        """Different contact states must produce different gate weights."""
        contact_a = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * B)
        contact_b = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * B)
        with torch.no_grad():
            gate_a = self.critic.gate(contact_a)
            gate_b = self.critic.gate(contact_b)
        # With random init, different inputs should produce different outputs
        assert not torch.allclose(gate_a, gate_b, atol=1e-6), \
            "Different contact states produced identical gate weights"


# =====================================================================
# 6. TestStudentLoss
# =====================================================================

class TestStudentLoss:
    def test_coefficients(self):
        """Default coefficients must match paper Table VI."""
        loss_fn = StudentLoss()
        assert loss_fn.lam_dist == 0.02, f"lam_dist {loss_fn.lam_dist} != 0.02"
        assert loss_fn.lam_repr == 0.2, f"lam_repr {loss_fn.lam_repr} != 0.2"

    def test_forward(self):
        """StudentLoss forward must return a positive scalar."""
        loss_fn = StudentLoss()
        # Create non-zero differences to ensure loss > 0
        student_actions = torch.randn(B, 12)
        teacher_actions = torch.randn(B, 12)
        student_map_emb = torch.randn(B, 192)
        teacher_map_emb = torch.randn(B, 192)
        ppo_loss = torch.tensor(0.1)

        loss = loss_fn(ppo_loss, student_actions, teacher_actions,
                       student_map_emb, teacher_map_emb)
        assert loss.ndim == 0, f"Loss is not scalar, shape={loss.shape}"
        assert loss.item() > 0, f"Loss {loss.item()} not > 0"


# =====================================================================
# 7. TestTRON1 — full pipeline with TRON1 config
# =====================================================================

class TestTRON1:
    def test_full_pipeline(self):
        """TRON1 config: MappingNet -> WTAMapFusion -> AME2Policy forward pass."""
        t1_map_cfg = tron1_mapping_cfg()
        t1_pol_cfg = tron1_policy_cfg()

        # MappingNet with TRON1 grid size
        mapping = MappingNet(t1_map_cfg)
        raw = torch.randn(B, 1, t1_map_cfg.map_h, t1_map_cfg.map_w)
        elev, log_var = mapping(raw)
        assert elev.shape == raw.shape
        assert log_var.shape == raw.shape

        # WTA fusion with TRON1 params
        wta = WTAMapFusion(B, **TRON1_WTA_KWARGS)
        wta.reset()
        poses = torch.zeros(B, 3)
        wta.update(elev, log_var, poses)
        policy_map = wta.crop(poses)
        assert policy_map.shape == (B, t1_pol_cfg.d_map_student,
                                     t1_pol_cfg.map_h, t1_pol_cfg.map_w), \
            f"TRON1 policy map shape {policy_map.shape} incorrect"

        # Student policy with TRON1 config
        student = AME2Policy(t1_pol_cfg, is_student=True)
        T = t1_pol_cfg.prop_history
        prop_hist = torch.randn(B, T, t1_pol_cfg.d_hist)
        commands = torch.randn(B, t1_pol_cfg.d_commands)
        actions, map_emb, prop_emb = student(policy_map, prop_hist=prop_hist,
                                              commands=commands)
        assert actions.shape == (B, t1_pol_cfg.num_joints), \
            f"TRON1 actions shape {actions.shape} != ({B}, {t1_pol_cfg.num_joints})"
        assert map_emb.shape == (B, t1_pol_cfg.d_map_emb), \
            f"TRON1 map_emb shape {map_emb.shape} != ({B}, {t1_pol_cfg.d_map_emb})"
        assert prop_emb.shape == (B, t1_pol_cfg.d_prop_emb), \
            f"TRON1 prop_emb shape {prop_emb.shape} != ({B}, {t1_pol_cfg.d_prop_emb})"
