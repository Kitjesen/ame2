"""
AME-2 Training Launch Script
==============================

Three-phase training pipeline for ANYmal-D:

  Phase 0  MappingNet pretraining (standalone, ~30 min)
  Phase 1  Teacher PPO (80k iter, ~12 hr on 4800 envs)
  Phase 2  Student distillation + PPO (40k iter, ~12 hr)

Usage:
------
  # Phase 0: Pretrain MappingNet
  python train_ame2.py --phase 0

  # Phase 1: Train teacher
  python train_ame2.py --phase 1 --num_envs 4800

  # Phase 2: Train student (requires teacher checkpoint)
  python train_ame2.py --phase 2 --teacher_ckpt logs/ame2_teacher/model_80000.pt

  # Resume training
  python train_ame2.py --phase 1 --resume logs/ame2_teacher/

Requirements:
-------------
  - NVIDIA Isaac Sim + Isaac Lab (isaaclab package)
  - isaaclab_rl (RSL-RL wrapper)
  - isaaclab_assets (ANYmal-D USD + actuator nets)
  - robot_lab installed: pip install -e source/robot_lab

References:
-----------
  Zhang et al., "AME-2", arXiv:2601.08485
  Li et al., "LSIO", arXiv:2401.16889
  RSL-RL: https://github.com/leggedrobotics/rsl_rl
  robot_lab: https://github.com/fan-ziqi/robot_lab
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# 0. Parse arguments BEFORE importing Isaac Sim (AppLauncher reads sys.argv)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="AME-2 Training Launch")
parser.add_argument("--phase", type=int, choices=[0, 1, 2], required=True,
                    help="Training phase: 0=MappingNet, 1=Teacher PPO, 2=Student")
parser.add_argument("--num_envs", type=int, default=4800,
                    help="Number of parallel environments [stated: 4800]")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override max training iterations")
parser.add_argument("--teacher_ckpt", type=str, default=None,
                    help="Path to teacher checkpoint (required for phase 2)")
parser.add_argument("--resume", type=str, default=None,
                    help="Experiment log directory to resume from")
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory for saving logs and checkpoints")
parser.add_argument("--device", type=str, default="cuda",
                    help="Training device")
parser.add_argument("--headless", action="store_true", default=True,
                    help="Run without viewer (default: True for training)")
parser.add_argument("--seed", type=int, default=42)
args, unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Phase 0: MappingNet pretraining — no Isaac Sim needed
# ---------------------------------------------------------------------------
if args.phase == 0:
    print("=" * 60)
    print("Phase 0: MappingNet Pretraining")
    print("=" * 60)
    import subprocess
    os.makedirs(args.log_dir, exist_ok=True)
    save_path = os.path.join(args.log_dir, "mapping_net.pt")
    script = os.path.join(os.path.dirname(__file__), "train_mapping.py")
    subprocess.run(
        [sys.executable, script, "--save_path", save_path],
        check=True,
    )
    print(f"[Phase 0] MappingNet saved to: {save_path}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Phase 1 / 2: Isaac Sim must be launched first via AppLauncher
# ---------------------------------------------------------------------------
try:
    from isaaclab.app import AppLauncher
except ImportError:
    print("[ERROR] isaaclab not found. Install Isaac Lab first.")
    print("  See: https://isaac-sim.github.io/IsaacLab/")
    sys.exit(1)

if args.headless and "--headless" not in sys.argv:
    sys.argv.append("--headless")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# All Isaac Lab imports MUST be after AppLauncher
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl.runners import OnPolicyRunner, DistillationRunner

# robot_lab import triggers gym.register for "RobotLab-Isaac-AME2-Rough-ANYmal-D-v0"
import ame2  # noqa: F401  — registers "RobotLab-Isaac-AME2-Rough-ANYmal-D-v0"

from ame2 import AME2AnymalEnvCfg
from ame2.networks import (
    anymal_d_policy_cfg,
    anymal_d_mapping_cfg,
    MappingNet,
    ANYMAL_D_WTA_KWARGS,
    AME2ActorCritic,
    AME2StudentActorCritic,
    WTAMapManager,
    AME2MapEnvWrapper,
)
from ame2.agents.rsl_rl_cfg import (
    AME2TeacherPPORunnerCfg,
    AME2StudentDistillationRunnerCfg,
)

ENV_ID = "RobotLab-Isaac-AME2-Rough-ANYmal-D-v0"


# ---------------------------------------------------------------------------
# Helper: build logging directory
# ---------------------------------------------------------------------------
def make_log_dir(phase: int, base: str = "logs") -> str:
    import time
    name = {1: "ame2_teacher", 2: "ame2_student"}[phase]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"{name}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Phase 1: Teacher PPO
# ---------------------------------------------------------------------------
def run_teacher_training():
    print("=" * 60)
    print("Phase 1: AME-2 Teacher Policy — PPO Training")
    print(f"  Environments : {args.num_envs}")
    print(f"  Device       : {args.device}")
    print("=" * 60)

    env_cfg = AME2AnymalEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    policy_cfg = anymal_d_policy_cfg()

    # MappingNet
    mapping_net = MappingNet(anymal_d_mapping_cfg()).to(args.device)
    mapping_ckpt = os.path.join(args.log_dir, "mapping_net.pt")
    if os.path.exists(mapping_ckpt):
        mapping_net.load_state_dict(torch.load(mapping_ckpt, map_location=args.device))
        print(f"[Teacher] Loaded MappingNet weights from {mapping_ckpt}")
    else:
        print("[Teacher] No MappingNet checkpoint found — using random init weights.")

    wta_manager = WTAMapManager(
        num_envs=args.num_envs,
        wta_kwargs=ANYMAL_D_WTA_KWARGS,
        device=args.device,
    )

    gym_env = gym.make(ENV_ID, cfg=env_cfg)
    env = AME2MapEnvWrapper(
        gym_env, mapping_net, wta_manager, policy_cfg,
        is_student=False,
        device=args.device,
    )

    runner_cfg = AME2TeacherPPORunnerCfg()
    runner_cfg.device = args.device
    if args.max_iterations:
        runner_cfg.max_iterations = args.max_iterations

    log_dir = make_log_dir(1, args.log_dir)

    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=log_dir,
        device=args.device,
    )

    # Post-hoc actor-critic replacement: RSL-RL's OnPolicyRunner constructs a
    # default ActorCritic at __init__ time (via the "policy" key in runner_cfg).
    # AME2ActorCritic uses a TensorDict interface incompatible with the default
    # RSL-RL MLP policy, so we replace it immediately after runner construction
    # before any training step has used the default.  The optimizer is rebuilt
    # on the new parameters.  runner.load() (resume) is called AFTER replacement.
    ame2_net = AME2ActorCritic(
        obs=None,        # obs=None is safe: AME2ActorCritic.__init__ does not inspect obs
        obs_groups={},
        num_actions=policy_cfg.num_joints,
        ame2_cfg=policy_cfg,
        is_student=False,
    ).to(args.device)
    runner.alg.actor_critic = ame2_net
    runner.alg.optimizer = torch.optim.Adam(ame2_net.parameters(), lr=1e-3)

    if args.resume:
        runner.load(args.resume)

    # ------------------------------------------------------------------
    # PPO training curricula (Sec. IV-D3, IV-E):
    #   1. Entropy coefficient: linear decay 0.004 → 0.001 [stated]
    #   2. Perception noise:    linear 0 → max in first 20 % iter [stated]
    #   3. Initial heading:     linear 0 → full ±π in first 20 % iter [stated]
    # ------------------------------------------------------------------
    _ENTROPY_START = 0.004          # [stated] Table VI
    _ENTROPY_END   = 0.001          # [stated] Table VI
    _NOISE_RAMP_FRAC = 0.20         # [stated] first 20 % of training
    _HEADING_RAMP_FRAC = 0.20       # [stated] first 20 % of training

    # Start heading curriculum fully constrained (frac=0 → face goal at reset)
    env.set_heading_curriculum(0.0)

    _orig_teacher_update = runner.alg.update

    def _teacher_update_with_curricula():
        result = _orig_teacher_update()
        it    = runner.current_learning_iteration
        total = runner_cfg.max_iterations

        # 1. Entropy linear decay
        frac_total = it / total
        new_entropy = _ENTROPY_START + (_ENTROPY_END - _ENTROPY_START) * min(frac_total, 1.0)
        runner.alg.entropy_coef = new_entropy

        # 2. Perception noise curriculum (ramp noise in first 20 % of iterations)
        noise_scale = min(it / (_NOISE_RAMP_FRAC * total), 1.0)
        env.set_scan_noise_scale(noise_scale)

        # 3. Heading curriculum (expand heading range in first 20 % of iterations)
        heading_frac = min(it / (_HEADING_RAMP_FRAC * total), 1.0)
        env.set_heading_curriculum(heading_frac)

        return result

    runner.alg.update = _teacher_update_with_curricula

    print(f"[Teacher] Logging to: {log_dir}")
    print(f"[Teacher] Entropy decay: {_ENTROPY_START} → {_ENTROPY_END} over {runner_cfg.max_iterations} iter")
    print(f"[Teacher] Perception noise ramp: 0 → max in first "
          f"{int(_NOISE_RAMP_FRAC * runner_cfg.max_iterations)} iter")
    print(f"[Teacher] Heading curriculum: constrained → full in first "
          f"{int(_HEADING_RAMP_FRAC * runner_cfg.max_iterations)} iter")
    print(f"[Teacher] Starting training for {runner_cfg.max_iterations} iterations...")

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"[Teacher] Training complete. Checkpoint: {log_dir}")
    return log_dir


# ---------------------------------------------------------------------------
# Phase 2: Student Distillation + PPO
# ---------------------------------------------------------------------------
def run_student_training(teacher_ckpt: str):
    print("=" * 60)
    print("Phase 2: AME-2 Student Policy — Distillation + PPO")
    print(f"  Teacher ckpt : {teacher_ckpt}")
    print(f"  Environments : {args.num_envs}")
    print("=" * 60)

    if not teacher_ckpt or not os.path.exists(teacher_ckpt):
        raise ValueError(
            f"Teacher checkpoint not found: {teacher_ckpt}\n"
            "Run Phase 1 first or pass --teacher_ckpt <path>"
        )

    env_cfg = AME2AnymalEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    policy_cfg = anymal_d_policy_cfg()

    mapping_net = MappingNet(anymal_d_mapping_cfg()).to(args.device)
    mapping_ckpt = os.path.join(args.log_dir, "mapping_net.pt")
    if os.path.exists(mapping_ckpt):
        mapping_net.load_state_dict(torch.load(mapping_ckpt, map_location=args.device))
        print(f"[Student] Loaded MappingNet weights from {mapping_ckpt}")

    wta_manager = WTAMapManager(
        num_envs=args.num_envs,
        wta_kwargs=ANYMAL_D_WTA_KWARGS,
        device=args.device,
    )

    gym_env = gym.make(ENV_ID, cfg=env_cfg)
    env = AME2MapEnvWrapper(
        gym_env, mapping_net, wta_manager, policy_cfg,
        is_student=True,
        device=args.device,
    )

    runner_cfg = AME2StudentDistillationRunnerCfg()
    runner_cfg.device = args.device
    if args.max_iterations:
        runner_cfg.max_iterations = args.max_iterations

    log_dir = make_log_dir(2, args.log_dir)

    runner = DistillationRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=log_dir,
        device=args.device,
    )

    # Same post-hoc replacement pattern as teacher phase — see comment there.
    student_net = AME2StudentActorCritic(
        obs=None,        # obs=None is safe: AME2StudentActorCritic.__init__ does not inspect obs
        obs_groups={},
        num_actions=policy_cfg.num_joints,
        ame2_cfg=policy_cfg,
    ).to(args.device)
    student_net.load_state_dict(
        torch.load(teacher_ckpt, map_location=args.device), strict=False
    )
    optimizer = torch.optim.Adam(
        student_net.parameters(), lr=AME2StudentActorCritic.PHASE1_LR
    )
    runner.alg.actor_critic = student_net
    runner.alg.optimizer = optimizer

    # Inject phase management: RSL-RL has no hook, so we patch alg.update()
    # to call step_iteration() after every gradient update.
    _orig_update = runner.alg.update

    def _phase_aware_update():
        result = _orig_update()
        student_net.step_iteration(optimizer, runner.current_learning_iteration)
        return result

    runner.alg.update = _phase_aware_update

    print(f"[Student] Logging to: {log_dir}")
    print(f"[Student] Phase 1 (pure distillation): "
          f"iter 0-{AME2StudentActorCritic.PHASE1_ITERS}  "
          f"lr={AME2StudentActorCritic.PHASE1_LR}")
    print(f"[Student] Phase 2 (PPO+distillation) : "
          f"iter {AME2StudentActorCritic.PHASE1_ITERS}-{runner_cfg.max_iterations}  "
          f"lr={AME2StudentActorCritic.PHASE2_LR}")

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"[Student] Training complete. Checkpoint: {log_dir}")
    return log_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if args.phase == 1:
        run_teacher_training()
    elif args.phase == 2:
        ckpt = args.teacher_ckpt
        if ckpt is None:
            teacher_logs = [
                d for d in os.listdir(args.log_dir)
                if d.startswith("ame2_teacher")
            ] if os.path.isdir(args.log_dir) else []
            if teacher_logs:
                latest = sorted(teacher_logs)[-1]
                ckpt = os.path.join(args.log_dir, latest)
                print(f"[Auto] Using teacher checkpoint: {ckpt}")
            else:
                parser.error("Phase 2 requires --teacher_ckpt or a completed Phase 1 run.")
        run_student_training(ckpt)

    simulation_app.close()


if __name__ == "__main__":
    main()
