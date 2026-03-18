"""AME-2 Teacher Training — V43 Paper-Faithful.

Matches AME-2 paper exactly (Table I + Table VI + Sec.IV-D).
No simplifications, no bootstrap, no custom curriculum.

Only paper curricula kept:
  - Heading: face-goal → random yaw over first 20% iters (Sec.IV-D.3)
  - Perception noise: 0 → max over first 20% iters (Sec.IV-D.3)
  - Terrain: automatic via IsaacLab (success → harder, fail → easier)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_ame2_direct.py \
        --num_envs 2048 --seed 42 --log_dir logs_v43/gpu0 --headless
"""

from __future__ import annotations

import argparse
import sys

# ── Isaac Sim MUST be imported before anything else ──────────────────────────
import isaacsim  # noqa: F401

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AME-2 V43 Teacher Training (Paper-Faithful)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs",         type=int,   default=2048,
                    help="Number of parallel environments (paper: 4800)")
parser.add_argument("--max_iterations",   type=int,   default=80_000,
                    help="Total PPO training iterations")
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--log_dir",          type=str,   default="logs_v43")
parser.add_argument("--resume",           type=str,   default=None,
                    help="Checkpoint .pt path to resume from")
# ── Reward weight overrides (all w_* from config, auto-registered) ──
for _wn in [
    "w_position_tracking", "w_arrival", "w_heading_tracking",
    "w_moving_to_goal", "w_standing_at_goal",
    "w_bias_goal", "w_anti_stall", "w_upward",
    "w_goal_coarse", "w_goal_fine",
    "w_vel_toward_goal", "w_position_approach",
    "w_base_height", "w_feet_air_time", "w_anti_stagnation", "w_lin_vel_z_l2",
    "w_early_termination", "w_undesired_contacts",
    "w_ang_vel_xy_l2", "w_joint_reg_l2", "w_action_rate_l2",
    "w_link_contact_forces", "w_link_acceleration",
    "w_joint_pos_limits", "w_joint_vel_limits", "w_joint_torque_limits",
]:
    parser.add_argument(f"--{_wn}", type=float, default=None)
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Episode length in seconds (default: 20s from env config)")
parser.add_argument("--terminal_reward_window_s", type=float, default=None,
                    help="V59: pos_tracking only in last N seconds (0=always on, default)")
parser.add_argument("--anti_stagnation_decay_iter", type=int, default=150,
                    help="Zero out anti_stagnation after this many iters (RSL paper: 150)")
args_cli, _ = parser.parse_known_args()

if not hasattr(args_cli, "headless"):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import os
import time
import torch

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.runners import OnPolicyRunner

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ame2_direct import AME2DirectEnv, AME2DirectEnvCfg, AME2DirectWrapper  # noqa: E402

try:
    from ame2.networks.rslrl_wrapper import AME2ActorCritic
    from ame2.networks.ame2_model import anymal_d_policy_cfg
except ImportError:
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.rslrl_wrapper import AME2ActorCritic
    from robot_lab.tasks.manager_based.locomotion.ame2.networks.ame2_model import anymal_d_policy_cfg


# ── PPO config (Paper Table VI) ──────────────────────────────────────────────

def make_runner_cfg(seed: int, log_dir: str, device: str) -> dict:
    """PPO runner config — Paper Appendix C, Table VI."""
    cfg = RslRlOnPolicyRunnerCfg(
        seed=seed,
        device=device,
        num_steps_per_env=24,            # Paper Table VI (V52b: restored, use mini_batches=6)
        max_iterations=args_cli.max_iterations,
        save_interval=999999,
        experiment_name="ame2_v46",
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 256, 128],
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=2.0,             # V52: 1→2, strengthen critic training
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.004,          # Paper Table VI: 0.004→0.001 (decay handled by update_curricula)
            num_learning_epochs=4,       # Paper Table VI (was 2)
            num_mini_batches=6,          # V52b: batch=8192 (same VRAM as steps=16/mb=4, but 50% more data)
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,                  # Paper Table VI
            lam=0.95,                    # Paper Table VI
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "policy": ["prop"],
            "critic": ["prop"],
        },
    )
    return cfg.to_dict()


# ── Paper Curricula (Sec.IV-D.3) ────────────────────────────────────────────
# Only two curricula from paper: heading and perception noise, both 20% ramp.
# Terrain curriculum is automatic in IsaacLab (no custom code needed).

_HEADING_RAMP_FRAC = 0.2        # Paper: "first 20% iterations"
_NOISE_RAMP_FRAC   = 0.2        # Paper: "first 20% iterations"

# Entropy decay: 0.004 → 0.001 over training (Paper Table VI)
_ENTROPY_START = 0.004
_ENTROPY_END   = 0.001


def update_curricula(env_direct: AME2DirectWrapper, runner: OnPolicyRunner, it: int) -> None:
    """Paper curricula only — heading, noise, entropy decay."""
    max_it = args_cli.max_iterations

    # ── 1. Heading: face-goal → random yaw over first 20% ──
    heading_frac = min(1.0, it / max(1, int(_HEADING_RAMP_FRAC * max_it)))
    env_direct.set_heading_curriculum(heading_frac)

    # ── 2. Perception noise: 0 → max over first 20% ──
    noise_scale = min(1.0, it / max(1, int(_NOISE_RAMP_FRAC * max_it)))
    env_direct.set_scan_noise_scale(noise_scale)

    # ── 3. Entropy decay: 0.004 → 0.001 linearly (Paper Table VI) ──
    frac = min(1.0, it / max(1, max_it))
    entropy = _ENTROPY_START + (_ENTROPY_END - _ENTROPY_START) * frac
    runner.alg.entropy_coef = entropy

    # ── 4. Anti_stagnation decay: zero out after N iters (RSL paper: first 150 only) ──
    if it == args_cli.anti_stagnation_decay_iter:
        env_direct.env.cfg.w_anti_stagnation = 0.0
        print(f"[Curriculum] it={it}: w_anti_stagnation zeroed (RSL: kickstart only)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Env ──
    policy_cfg = anymal_d_policy_cfg()
    policy_cfg.d_prop_critic = 55
    env_cfg = AME2DirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # ── Apply CLI overrides (before env __init__) ──
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
        print(f"  [Override] episode_length_s = {args_cli.episode_length_s}")
    if args_cli.terminal_reward_window_s is not None:
        env_cfg.terminal_reward_window_s = args_cli.terminal_reward_window_s
        print(f"  [Override] terminal_reward_window_s = {args_cli.terminal_reward_window_s}")
    for attr_name in dir(env_cfg):
        if attr_name.startswith('w_') and isinstance(getattr(env_cfg, attr_name), (int, float)):
            val = getattr(args_cli, attr_name, None)
            if val is not None:
                setattr(env_cfg, attr_name, val)
                print(f"  [Override] {attr_name} = {val}")
    env = AME2DirectEnv(cfg=env_cfg)

    rsl_env = AME2DirectWrapper(env, device=device)

    # ── AME-2 network ──
    ame2_net = AME2ActorCritic(
        obs=None,
        obs_groups={},
        num_actions=12,
        ame2_cfg=policy_cfg,
        is_student=False,
        init_noise_std=0.5,              # V52d: 0.5, capped at 0.5 in wrapper (enough explore, stable critic)
    ).to(device)

    # ── Runner ──
    os.makedirs(args_cli.log_dir, exist_ok=True)
    runner_cfg = make_runner_cfg(args_cli.seed, args_cli.log_dir, device)
    runner = OnPolicyRunner(rsl_env, runner_cfg, args_cli.log_dir, device=device)
    runner.alg.policy = ame2_net
    runner.alg.optimizer = torch.optim.Adam(ame2_net.parameters(), lr=1e-3)

    # ── Resume ──
    it_start = 0
    if args_cli.resume:
        print(f"[Resume] {args_cli.resume}")
        ckpt = torch.load(args_cli.resume, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        ame2_net.load_state_dict(state, strict=False)
        with torch.no_grad():
            ame2_net.std.fill_(0.5)
        import re
        m = re.search(r"model_(\d+)\.pt", args_cli.resume)
        if m:
            it_start = int(m.group(1)) + 1
            print(f"[Resume] from iteration {it_start}")

    # ── Print config ──
    print(f"\n[AME-2 V43 Paper-Faithful]")
    print(f"  envs={args_cli.num_envs}  iters={args_cli.max_iterations}  "
          f"seed={args_cli.seed}  device={device}")
    tw = env_cfg.terminal_reward_window_s
    print(f"  episode={env_cfg.episode_length_s}s  "
          f"terminal_window={tw}s({'always-on' if tw == 0 else f'last {tw}s'})  "
          f"goal=[{env_cfg.goal_pos_range_min}, {env_cfg.goal_pos_range_max}]m  "
          f"v_min={env_cfg.moving_to_goal_v_min}")
    print(f"  PPO: epochs=4  mini_batches=6  steps=24  vloss_coef=2.0  entropy=0.004→0.001")
    print(f"  Rewards (pre-dt): pos={env_cfg.w_position_tracking:.0f} arrival={env_cfg.w_arrival:.0f} "
          f"vel={env_cfg.w_vel_toward_goal:.0f} move={env_cfg.w_moving_to_goal:.0f} "
          f"appr={env_cfg.w_position_approach:.0f} undes={env_cfg.w_undesired_contacts:.0f}")
    print(f"  Log: {args_cli.log_dir}\n")

    t0 = time.time()

    for it in range(it_start, args_cli.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        update_curricula(rsl_env, runner, it)

        # Save checkpoint every 50 iters
        if it % 50 == 0:
            ckpt_path = os.path.join(args_cli.log_dir, f"model_{it}.pt")
            torch.save({
                "model_state_dict": ame2_net.state_dict(),
                "iteration": it,
            }, ckpt_path)

        # Logging every 50 iters
        if it % 50 == 0:
            elapsed = time.time() - t0
            it_time = elapsed / max(it - it_start + 1, 1)
            eta_h   = it_time * (args_cli.max_iterations - it) / 3600

            ep_log  = runner.env.extras.get("log", {})
            pos_t   = float(ep_log.get("Episode_Reward/position_tracking", 0.0))
            move    = float(ep_log.get("Episode_Reward/moving_to_goal", 0.0))
            stand   = float(ep_log.get("Episode_Reward/standing_at_goal", 0.0))
            head    = float(ep_log.get("Episode_Reward/heading_tracking", 0.0))
            appr    = float(ep_log.get("Episode_Reward/position_approach", 0.0))
            vtg     = float(ep_log.get("Episode_Reward/vel_toward_goal", 0.0))
            dxy     = float(ep_log.get("Episode_Goal/terminal_dxy_mean", 0.0))
            succ05  = float(ep_log.get("Episode_Success/pos_0.50m", 0.0))
            succ100 = float(ep_log.get("Episode_Success/pos_1.00m", 0.0))

            print(
                f"[it {it:6d}] "
                f"terrain={rsl_env.get_terrain_level():.2f}  "
                f"dxy={dxy:.2f}m  succ@0.5={succ05:.2f}  succ@1.0={succ100:.2f}  "
                f"pos_t={pos_t:.3f}  move={move:.3f}  stand={stand:.3f}  head={head:.3f}  "
                f"appr={appr:.3f}  vtg={vtg:.3f}  "
                f"iter={it_time:.1f}s  ETA={eta_h:.1f}h"
            )

    print("[AME-2 V43] Done!")
    simulation_app.close()


if __name__ == "__main__":
    main()
