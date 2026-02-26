"""
AME-2 RSL-RL Agent Configurations
===================================

Three training phases for ANYmal-D:

  Phase 0 (optional): MappingNet pretraining — standalone, not via RSL-RL
  Phase 1: Teacher PPO — AME2TeacherPPORunnerCfg (80k iter, Sec.IV-E)
  Phase 2: Student distillation — AME2StudentDistillationRunnerCfg (5k pure + 35k PPO)

PPO parameters from AME-2 Table VI (all [stated] unless noted):
  clip_param=0.2, value_loss_coef=1.0, entropy 0.004→0.001 [stated],
  num_learning_epochs=4 [stated], num_mini_batches=3 [stated],
  sim_steps=24 [stated], N_envs=4800 [stated],
  gamma=0.99, lam=0.95, desired_kl=0.01

RSL-RL reference: isaaclab_rl.rsl_rl (isaaclab-rl package)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# =============================================================================
# Phase 1 — Teacher PPO
# =============================================================================

@configclass
class AME2TeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training for the AME-2 teacher policy.

    Teacher receives privileged observations:
      - prop (48D): base vel (6) + gravity (3) + joint pos/vel (24) + cmd (3) + actions (12)
      - height_scan (1581D): ground-truth elevation map from RayCaster
      - contact_forces (12D): true foot contact forces

    Training: 80000 iterations [stated] (AME-2 Sec.IV-E).
    Perception noise curriculum: first 20% (16000 iter) linearly increase
    mapping noise from 0 to max, so the teacher gradually adapts.
    """

    num_steps_per_env: int = 24          # [stated] simulation steps per iteration
    max_iterations: int = 80_000         # [stated] Sec.IV-E teacher training length
    save_interval: int = 500
    experiment_name: str = "ame2_teacher"
    empirical_normalization: bool = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # [stated] Table VI
        use_clipped_value_loss=True,
        clip_param=0.2,                # [stated]
        entropy_coef=0.004,            # [stated] initial value; decays linearly to 0.001
                                       # via runner.alg.entropy_coef patch in train_ame2.py
        num_learning_epochs=4,         # [stated] Table VI
        num_mini_batches=3,            # [stated] Table VI
        learning_rate=1.0e-3,
        schedule="adaptive",           # [stated] "Adaptive"
        gamma=0.99,                    # [stated]
        lam=0.95,                      # [stated] GAE λ
        desired_kl=0.01,               # [stated]
        max_grad_norm=1.0,
    )


# =============================================================================
# Phase 2 — Student Distillation + PPO
# =============================================================================

@configclass
class AME2StudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Student training: first 5000 iter pure distillation, then 35000 iter PPO+distillation.

    From AME-2 Table VI [stated]:
      - Action Distillation Loss Coefficient (λ_dist): 0.02
      - Representation Loss Coefficient (λ_repr): 0.2
      - LR when surrogate loss disabled: 0.001
      - Total student training: 40000 iter (5000 pure + 35000 PPO)

    obs_groups maps RSL-RL observation groups to Isaac Lab observation group names:
      "policy"      → student policy observations (prop only, no privileged info)
      "teacher"     → teacher privileged observations (prop + height_scan + contact)
      "teacher_map" → GT policy map from height_scanner_policy (B, 1512 flat)
                      Training loop reshapes to (B, 3, 14, 36) and passes to
                      WTAMapManager.get_policy_maps(gt_map_flat=...).
    """

    num_steps_per_env: int = 24
    max_iterations: int = 40_000         # [stated] 5000 pure + 35000 PPO
    save_interval: int = 500
    experiment_name: str = "ame2_student"
    empirical_normalization: bool = False

    obs_groups: dict = {
        "policy": ["policy"],
        "teacher": ["teacher_privileged"],
        "teacher_map": ["teacher_map"],
    }

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=1.0e-3,
        gradient_length=24,
    )


# =============================================================================
# Convenience aliases
# =============================================================================

TEACHER_RUNNER_CFG = AME2TeacherPPORunnerCfg()
STUDENT_RUNNER_CFG = AME2StudentDistillationRunnerCfg()
