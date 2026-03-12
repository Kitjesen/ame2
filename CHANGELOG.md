# AME-2 Direct Training Changelog

所有版本均基于 `ame2_direct/` Direct Workflow（Isaac Lab DirectRLEnv + RSL-RL PPO）。
服务器：BSRL 8×RTX 3090 (`fe91fae6a6756695.natapp.cc:12346`)

---

## V49 — 2026-03-13 (current)

**修复"站着不动"exploit — 真正的 approach reward + 移除 _t_mask**

### 根因诊断（多 agent CTO 审计）

V48 300 迭代后 dxy 停滞在 ~4m，0% 成功率。三个独立 agent 审查发现：

1. **`_t_mask(4.0)` 使 position_tracking 前 800/1000 步 = 0** — episode 80% 时间完全没有位置奖励
2. **goal_coarse 梯度太弱** — d=4m 处每步只有 ~0.004 奖励差，不够覆盖走路惩罚
3. **旧 position_approach=exp(-0.5d) 是静态距离奖励，不是真正的 approach reward**
4. **step() 执行顺序是 dones→rewards→reset→obs**（code-reviewer 发现），影响 buffer 设计

### V49 修改（2 个核心 + 1 个配置）

1. **移除 position_tracking 的 `_t_mask(4.0)`** — 全程激活，999 步都有位置奖励
   - 原论文设计：只在最后 4s 评估位置。但对 2048 envs 的随机策略，80% 无奖励 = 学不到走路
2. **真正的 approach reward**: `clamp(d_prev - d_curr, 0, 0.1)` — 奖励每步靠近目标
   - 使用 `_reward_d_prev` 独立 buffer（在 `_get_rewards()` 内更新，避免执行顺序问题）
   - 在 `_reset_idx()` 中同步重置
   - clamp(0, 0.1)：只奖励靠近，不惩罚远离；上限 0.1m = 5m/s 远超 ANYmal 最大速度
3. **`w_position_approach = 50`**（×dt=1.0）: 走 0.5m/s → +10/ep，走 1.0m/s → +20/ep，站着 → 0

### V49 vs V48 奖励结构对比

| 时间段 | V48 | V49 |
|--------|-----|-----|
| 步 0-800 | goal_coarse(极弱) | **position_tracking + approach(强)** |
| 步 801-999 | pos_tracking + goal_coarse | pos_tracking + approach + goal_coarse |
| 走路净收益 | ~0 (penalties ≈ reward gain) | **+8.7~+18.7/ep >> penalties** |

---

## V48 — 2026-03-12

**修复 stagnation 和 base_collision 导致 episode 过早终止 — 10轮审计 Round 6**

### Round 6：Episode 生存时间修复（3 个关键修正）

**根因诊断**：V47 所有 episode 在 250 步（5s）被 stagnation 杀死。max_episode_length=1000（20s），
但 stagnation 5s 窗口（250步）无 grace period → 第 250 步立即终止。_t_mask(4.0) 在 step>800 才激活，
position_tracking 永远为 0。机器人只有负奖励，理性选择不动。

1. **stagnation 添加 grace period 800 步（16s）** — 保证 episode 活到 _t_mask(4.0) 激活
   V47 全部 100% stagnation 在 5 秒 → V48 0% stagnation，episode 跑满 999 步（20s）
2. **base_collision 移除（第三次）** — contact sensor 在 rough terrain 上 100% 误触发（和 V43h 高度代理一样）
   bad_orientation 已覆盖摔倒检测，base_collision 冗余且不可靠
3. **启用 goal_coarse = 5.0** — `1-tanh(d/2)` 提供从第 1 步就有的正梯度
   V47 前 16s 完全没有正 task reward → V48 goal_coarse 提供持续引导信号

### V48 vs V47 初始指标对比

| 指标 | V47 | V48 |
|------|-----|-----|
| Mean reward | -22.88 | **+51.54** |
| Episode length | 250 (5s) | **999 (20s)** |
| position_tracking | 0.0000 | **4.23** |
| goal_coarse | 0.0000 | **0.25** |
| stagnation | 100% | **0%** |
| base_collision | 0% (V47) / 100% (V48r1) | **0% (disabled)** |

### 实验配置
- **V48a (GPU 0)**: move=20, jpos=-50, seed=480, goal_coarse=5 → logs_v48/v48a_r2.log
- **V48b (GPU 2)**: move=10, jpos=-50, seed=481, goal_coarse=5 → logs_v48/v48b_r2.log
- **V48c (GPU 3)**: move=5(论文), jpos=-50, seed=482, goal_coarse=5 → logs_v48/v48c_r2.log
- ETA: ~350h each。首次有意义检查：~1000 迭代（~4.2h）

---

## V47 — 2026-03-12

**终止条件 + 奖励公式 + 训练超参回归论文 — 10轮审计 Round 2-5**

### 终止条件修复（4 个偏差修正）
1. **bad_orientation 3轴检查** — |g_x|>0.985(roll>80°) | |g_y|>0.7(pitch>45°) | g_z>0.0(倒转>90°)
   之前只检查 g_z>-0.5(>60° tilt)，缺少 roll/pitch 独立检测
2. **base_collision 重新启用** — 用 contact sensor 力 > 体重(490N)，替代不可靠的 terrain_origin_z 高度代理
3. **thigh_acc 阈值 500→100** — 论文 60 m/s²，但数值微分噪声导致正常步态峰值~50，折中取 100
4. **stagnation 距离 0.5m→1.0m** — 论文值，只有距目标>1m 且 5s 内移动<0.5m 才终止

### 奖励公式修复（5 个偏差修正）
5. **_t_mask 改为二值指示器** — 论文 Eq.(1) I(t_goal≤T) 是 0/1，之前返回 1/T=0.25 导致 position_tracking 有效权重仅 25
6. **entropy 改为论文值** — 0.004→0.001（之前 0.01→0.003，偏离论文 Table VI）
7. **self-collision 累加 cap 到 1** — 之前 6 对 shank 可累加 0-6，现在 any() 后二值 0/1
8. **moving_to_goal 改为论文连续公式** — `max(v·cos(θ)-v_min, 0) * I(d>0.5)`
   之前是二值 0/1（cos>0.5 且 vel_nrm>v_min），缺乏连续梯度。
   且 `(d_xy<0.5)|` 条件反了：到达目标时不该给 moving 奖励
9. **standing_at_goal 改为论文 Eq.(5)** — `exp(-4·||v||²) * I(d<0.5) * I(d_yaw<0.5) * I(t_left<2s)`
   之前用了脚部接触+重力+关节偏差的复杂公式，与论文完全不同，且缺少 _t_mask(2)

### Bug 修复（1 个 runtime error）
10. **cos_t/vel_nrm NameError** — moving_to_goal 重写后 bias_goal 引用了已删除变量，导致运行崩溃
    虽然 w_bias_goal=0 理论上不会触发，但 IsaacLab DirectRLEnv 会执行所有奖励计算

实验（V47 Round4fix，BSRL GPU 0/2/3）：
- V47a (GPU 0): move=20, jpos=-50, seed=470
- V47b (GPU 2): move=10, jpos=-50, seed=471
- V47c (GPU 3): move=5(论文值), jpos=-50, seed=472

预期：_t_mask 二值化使 position_tracking 有效权重 4x 提升 (25→100)，
moving_to_goal 连续化消除梯度断崖，standing_at_goal+_t_mask(2) 正确激励减速。
ETA: ~350h per experiment (15s/iter × 80000 iters)。每 1000 迭代检查进展。

## V46 — 2026-03-12

**全面回归论文设计 — 代码审查修复 + 奖励策略对齐**

### 代码修复（8 个 bug）
1. **MappingNet scanner 前移 x=1.0m** — 论文 Sec.V-B，覆盖 [0, 2.0]m 前方
2. **Policy map scanner 前移 x=0.6m** — 论文 Sec.IV-E，覆盖 [-0.12, 1.12]m 前方
3. **link_acceleration 改用真正加速度** — 帧间速度差/dt
4. **terminal_d_xy 在 robot.reset() 之前计算** — 确保 terrain curriculum 正确
5. **ang_vel_xy_l2 加入 pitch** — roll + pitch
6. **obs noise 覆盖全部 48D** — 补 prev_actions(12D) + actor_cmd(3D)
7. **_prev_d_xy 改为 in-place 更新** — 避免 full-tensor 替换问题
8. **num_envs 默认值 4800→2048** — RTX 3090 OOM 保护

### 奖励策略回归论文（4 个偏差修正）
1. **position_tracking 改回论文 Eq.(1)** — `1/(1+0.25d²) * t_mask(4)` 最后 4s 激活
   之前 V44 改成 `clamp(d_prev-d_curr)` approach reward 是错的：V43n 去掉 t_mask 导致
   exploit，应该直接回退而非换公式。论文设计：episode-level 到达，不约束路径。
2. **移除 arrival bonus** — w_arrival=0，论文 Table I 无此项
3. **移除 vel_toward_goal** — w_vel_toward_goal=0，论文 Table I 无此项
4. **joint_reg 改为论文公式** — `||q̇||² + 0.01||τ||² + 0.001||q̈||²`
   之前用 `(q-q_default)²` 是位置偏差，论文是速度+力矩+加速度

实验（论文策略 + 必要硬件适配）：
- V46a (GPU 0): 论文原版, jpos=-50, move=20, seed=460
- V46b (GPU 2): 论文原版, jpos=-50, move=10, seed=461
- V46c (GPU 3): 论文原版, jpos=-50, move=5(论文值), seed=462

jpos=-50 和 move>5 是 2048 envs 的硬件适配（论文 4800 envs 有更多随机探索）。

## V45 — 2026-03-12 (killed, scanner offset wrong)

**AME2 12种地形长期训练 — 修复 map encoder 不学习问题**

关键诊断：V44 的 map encoder 权重在 2750 迭代后几乎没变化（norm 变化 <2%），
而 prop encoder (+35%) 和 decoder (+42%) 在正常学习。
根因：训练量严重不足（2750/80000 = 3.4%）+ ROUGH 地形太简单 + mini_batches=16 梯度弱。
论文：80000 迭代, 4800 envs, mini_batches=3, 12种复杂地形, ~60 RTX-4090-days。

切换到 AME2_TERRAINS_CFG（12种地形, 12×12m tiles），目标 80000 迭代。

实验：
- V45a (GPU 0): AME2, jpos=-50, move=20, seed=450
- V45b (GPU 2): AME2, jpos=-50, move=20+vel=20, seed=451 — 双重梯度
- V45c (GPU 3): AME2, jpos=-50, vel=20, seed=452

ETA: ~350h (14.6 天) per experiment。每 5000 步检查 map encoder 权重变化。

## V44 — 2026-03-12

**代码审阅修复 — 5 个关键 bug**

全面审阅后发现：undesired_contacts 惩罚 > 全部正奖励 → 走路比站着亏 → 80% stagnation

修复内容：
1. **实现 arrival bonus** — w_arrival=500 有定义但 _get_rewards 里没有代码。现在 d<0.5m 时给持续奖励
2. **position_tracking 改为 approach reward** — `clamp(d_prev - d_curr, 0, 0.5)` 替代 `1/(1+d²)*_t_mask(4s)`。只奖励"靠近了多少"，站着不动=0，不再有 exploit
3. **slippage 阈值 0.1→0.5 m/s** — 旧阈值在正常行走时每步都触发
4. **stumbling 只检查非脚部** — 旧代码检查全部 link（包括脚），脚蹬地时水平力>垂直力是正常的
5. **w_undesired_contacts -5→-1** — 回到论文值，配合 bug 修复后不再需要 5x

新增 CLI 参数用于 A/B 测试：`--w_vel_toward_goal --w_moving_to_goal --w_undesired_contacts --w_position_tracking --w_arrival`

新增 CLI 参数：`--w_joint_pos_limits`

A/B 测试：
- V44a (GPU 0): baseline fixes, resume → bad_orientation 100% (value mismatch), killed
- V44b (GPU 3): fixes + vel=20, resume → same issue, killed
- V44c (GPU 2): from scratch, baseline → stagnation 100% at ep=244, walking net-negative
- V44d (GPU 0): from scratch, vel=20 → crashed (carb Mutex)
- V44e (GPU 3): from scratch, vel=20 → bad_orientation 100%, killed
- **V44f (GPU 0)**: `jpos=-100`, baseline — 降低关节限制惩罚让走路盈利 ⏳
- **V44g (GPU 2)**: `jpos=-100, move=20` — 同上 + 更强走路激励 ⏳
- **V44h (GPU 3)**: `jpos=-50, vel=20` — 极低关节限制 + 强方向梯度 ⏳

关键发现：w_joint_pos_limits=-1000(×dt=-20) 使走路每 ep 惩罚 -12.4，
走路总奖励(+12.5-12.4-9-10=-18.9) < 站着(-10)，机器人理性选择站着不动

## V43q — 2026-03-11

**启用 vel_toward_goal 连续方向梯度**

- `w_vel_toward_goal = 10`（新启用）— 速度投影到目标方向，连续梯度替代 binary 信号
- `w_moving_to_goal = 5`（从 20 降低）— 保留作为阈值激励，不再是主力
- 12 种地形回退到 ROUGH_TERRAINS_CFG（新地形导致 bad_orientation 100%）
- env.py 回退到 git 版本（V43p 的 goal resampling 有 bug: `_goal_lifetime=0` 导致每步都 resample）
- Resume from V43o model_4200，GPU 0

## V43p — 2026-03-11 (failed)

**goal 指标 + terrain curriculum — _goal_lifetime=0 bug**

- 添加 goals_reached/goals_attempted/arrival_rate 日志
- terrain curriculum 改为论文方案
- **失败**：`_goal_lifetime` 初始化为 0 → `goal_timeout = elapsed >= 0` 永远 True → 每步都 resample 目标
- bad_orientation 100%，37 分钟无恢复

## V43o — 2026-03-11

**距离变化奖励 — 修复 position_tracking exploit**

- position_tracking 从 `1/(1+0.25*d²)`（绝对距离，站着也拿分）改为 `clamp(d_prev - d_curr, 0, 0.5)`（只奖励靠近）
- 新增 arrival bonus（w=500 → 10/次），到达目标（d<0.5m）时一次性奖励
- Resume from V43m model_3450
- V43n 失败记录：always-on `1/(1+0.25d²)` 在 d=4m 处每步白拿 0.2，450 步净赚 +180/ep，机器人学会站着不动

## V43n — 2026-03-11 (failed)

**连续目标刷新 + always-on position_tracking — exploit**

- 添加 mid-episode goal resampling（论文 resampling_time_range=10-20s）
- position_tracking 改为 always-on（移除 _t_mask）
- 问题：`r = 1/(1+0.25*d²)` 在 d=4m 给 0.2/step，机器人站着白拿 → dxy=5.17，成功率 0%
- _t_mask 从 episode-based 改 goal-relative 导致策略崩溃（760→159 步），最终全部移除

## V43m — 2026-03-11

**提升走路激励 — 机器人学会走路**

- `w_moving_to_goal = 20`（论文 5，4x 提升）
- Resume from V43l model_1350
- 效果：episode length 135→760 步（2.7s→15.2s），机器人学会平衡和行走
- 偶尔到达目标（dxy=0.51m, succ@1.0=1.00），position_tracking 首次激活
- 新瓶颈：stagnation 终止（走一段就停），binary 奖励的断崖效应

## V43l — 2026-03-11

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

### 奖励设计
- **正负奖励必须平衡**：倒地惩罚 -10 vs 走路奖励 +0.5（w=5）时，机器人理性选择不冒险。w=20 让走路收益接近倒地代价，机器人才愿意走
- **论文参数不能直接抄**：论文 4800 envs 随机探索够覆盖大奖（position_tracking +400），2048 envs 不够，需更强的中间引导
- **大奖有 bootstrap 问题**：position_tracking 只在最后 4s 激活，机器人要先活 16s 才拿到。没足够 moving_to_goal 引导，永远发现不了
- **binary 奖励有断崖效应**：moving_to_goal 要求 >0.3m/s，低于阈值就归零，导致机器人走一段就停。需要连续奖励（vel_toward_goal）补充梯度
- **正奖励有 exploit 风险**：`w_base_height > 0` → 站着不动比走路更赚；`1/(1+0.25d²)` always-on → 站在 4m 处白拿 +180/ep
- **绝对距离奖励 vs 距离变化奖励**：`1/(1+d²)` 类绝对距离奖励在任意位置都给正值，机器人不需移动就赚钱。改为 `clamp(d_prev-d_curr, 0)` 只奖励"靠近了多少"，站着不动=0
- **到达目标需要显式奖励**：连续目标刷新下，到达 → 目标跳走 → 距离变大，approach reward 为 0。需要独立的 arrival bonus 奖励到达行为
- **论文的 _t_mask（最后 N 秒窗口）不适合连续目标刷新**：episode-based 倒计时改成 goal-relative 会破坏已训练好的 value function

### 环境配置
- **replicate_physics=True 是必须的**：ROUGH_TERRAINS_CFG 200 tiles < 2048 envs → 机器人碰撞
- **termination 阈值需谨慎调整**：太激进 → episode 太短 → 学不到东西
- **terrain_origin_z 不是可靠高度参考**：rough terrain ±0.3m+ 局部变化
- **ANYmal-D HFE/KFE 关节无实际限制**（±540°）：只有 HAA 有真实限制（±35-45°）

### 运维
- **RTX 3090 最多 2048 envs**（4096 PhysX OOM）
- **carb Mutex crash**：Omniverse 每 20-45 分钟必崩，靠 checkpoint resume
- **PYTHONUNBUFFERED=1**：nohup 时必须加
- **从坏 checkpoint 恢复有过渡期**：膝盖爬行→强制站立→episode 崩到 40 步→慢慢恢复
