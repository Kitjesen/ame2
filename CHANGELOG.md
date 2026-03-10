# AME-2 Direct Training Changelog

所有版本均基于 `ame2_direct/` Direct Workflow（Isaac Lab DirectRLEnv + RSL-RL PPO）。
服务器：BSRL 8×RTX 3090 (`fe91fae6a6756695.natapp.cc:12346`)

---

## V43l — 2026-03-11 (current)

**Paper-Faithful + Anti-Crawl**

- `w_base_height = 0.0` — removed, causes "stand still" exploit
- `w_undesired_contacts = -5.0` — 5× paper value to penalize knee crawling
- Resume from V43j model_800
- Episode length crashed to 40 steps (policy restructuring from crawl→walk), recovering toward 100+

## V43k — 2026-03-11 (failed)

- Added `w_base_height=5.0` to encourage standing → created "stand still" exploit
- Robot discovered standing still gives net positive reward (+base_height - stagnation)

## V43j — 2026-03-10

**Environment Stabilization** — 环境终于稳定运行

Key fixes from V43a-V43j debug:

1. **replicate_physics=True** — prevents robot collision across envs (ROUGH_TERRAINS_CFG has 200 tiles but 2048 envs)
2. **terrain_oob removed** — not in paper, 200 tiles < 2048 envs causes all OOB
3. **base_collision removed** — terrain_origin_z unreliable as height reference on rough terrain
4. **bad_orientation simplified** — `projected_gravity_z > -0.5` (>60° tilt), grace 20 steps
5. **thigh_acc threshold 500** — normal walking jitter ~50 m/s², crash >500 m/s²
6. **mini_batches=16** — RTX 3090 OOM with paper's 3
7. **init_at_random_ep_len=False** — prevents stagnation false trigger at step 0

## V43 — 2026-03-10

**Match Paper Exactly** (Table I + Table VI + Sec.IV-D)

- Removed all non-paper rewards (bias_goal, anti_stall, etc.)
- 20s episodes, [2m, 6m] goal distance, 4 PPO epochs, entropy decay 0.004→0.001
- link_contact_forces threshold: 490N (body weight), not 1N

## V42 — 2026-03-10 (deprecated)

- "Stand still" exploit: 8s episode + 0.8m goal = no need to walk
- Custom rewards (bias_goal, anti_stall) didn't help

## V41 — 2026-03-10 (deprecated)

- Disabled all terminations except timeout → robot exploits freely
- 50% fallen start ratio for recovery training

## V40 — 2026-03-09

**根因修复：奖励权重缺少 dt 乘法**

- env.py `__init__`: 所有 `w_*` 权重 ×step_dt=0.02（对齐 HIM/IsaacLab 惯例）
- 之前所有版本的权重实际上是等价值的 50倍

## V39 — 2026-03-09

- 对齐 robot_lab 奖励命名（upright_bonus→upward, etc.）
- upward 改为 robot_lab 正奖励 `(1-g_z)²`
- feet_air_time 禁用（公式在短步态返回负值）

## V33-V38 — 2026-03-07 to 2026-03-09

- V33: 机器人可移动但无步态（6个步态惩罚全 disabled）
- V34: 恢复论文 Table I 奖励平衡
- V35: 禁用 link_contact_forces（淹没目标信号）
- V36: upright_bonus 10x, 加 lin_vel_z_l2
- V37-V38: 调试 upright 和 anti_stagnation

## V25-V32 — 2026-02-xx to 2026-03-06

- 首次用 AME2ActorCritic + RSL-RL，关键经验：`runner.alg.policy=ame2_net`（非 actor_critic）
- Baseline 稳定化

---

## Lessons Learned

- **replicate_physics=True 是必须的**：ROUGH_TERRAINS_CFG 200 tiles < 2048 envs → 机器人碰撞
- **termination 阈值需谨慎调整**：太激进 → 误触发 → episode 太短 → 学不到东西
- **terrain_origin_z 不是可靠高度参考**：rough terrain ±0.3m+ 局部变化
- **ANYmal-D HFE/KFE 关节无实际限制**（±540°）：只有 HAA 有真实限制（±35-45°）
- **不要简化论文配置**：episode length, goal distance, PPO params 都不能随便改
- **正奖励有 exploit 风险**：`w_base_height > 0` → 站着不动比走路更赚
- **RTX 3090 最多 2048 envs**（4096 PhysX OOM）
- **carb Mutex crash**：Omniverse 每 20-45 分钟必崩，靠 checkpoint resume
- **PYTHONUNBUFFERED=1**：nohup 时必须加
