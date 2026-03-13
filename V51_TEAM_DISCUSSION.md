# V51 团队讨论记录

日期: 2026-03-14
参与者: PPO专家、奖励专家、观测架构专家、训练策略专家

---

## 一、核心诊断：50 个版本失败的三大根因

### 根因 1：走路是净亏损的（奖励专家）

精确计算证明，在当前奖励配置下，**走路比站着不动少拿分**：

```
站着不动 (d=3m):  +0.625/step  →  1000步 = +625
朝目标走 0.5m/s:  +0.590/step  →  1000步 = +590  (比站着少35!)
反方向走 0.5m/s:  +0.560/step  →  1000步 = +560
```

原因拆解：
- position_tracking 1/(1+0.25*d²) 在 d=3m 时白拿 0.615/step — 不需要任何动作
- 走路额外正奖励: approach(0.01) + moving(0.02) = +0.030/step
- 走路额外惩罚: undesired_contacts(-0.060) + 其他(-0.005) = -0.065/step
- **走路净收益 = +0.030 - 0.065 = -0.035/step**

undesired_contacts 是最大杀手：7个子项中3-4个在正常走路时触发（leaping、slippage、non-foot contact）。

### 根因 2：梯度信号弱 12.5 倍（PPO专家）

| 参数 | 论文 | 我们 | 差距 |
|------|------|------|------|
| mini_batch size | 38,400 | 3,072 | **12.5x** |
| 梯度方差 | 1x | 12.5x | — |

mini_batches=16 导致：
- Value function 拟合不稳定 → advantage 估计噪声大
- Policy gradient 方向漂移 → 学习低效
- Adaptive lr 过度反应 → lr 被压低

**RTX 3090 24GB 可以跑 mini_batches=4**（VRAM 预计从 14GB→16GB）。

### 根因 3：Actor 在 d>2m 时几乎没有方向信息（观测专家）

Actor cmd = [clip(d_xy, 2.0), sin(yaw+noise), cos(yaw+noise)]：
- d>2m 时距离信息 = 0（clip 到 2.0）
- 方向信息被 N(0,1) 噪声污染，信号效率只有 20%
- Map encoder 只有地形高度，**不包含目标位置**
- 方向学习完全依赖 critic→actor 的间接传递，在弱梯度下失效

---

## 二、四位专家的共识方案

### 共识 1：mini_batches 16 → 4

所有专家一致认为这是**最高优先级修改**。
- 梯度质量提升 4 倍
- VRAM 预计增加 1-2GB，不会 OOM
- 如果 OOM，退到 mini_batches=8

### 共识 2：奖励大幅简化

**删除 position_tracking（w=0）**— 消除站着白拿分的 exploit
**approach 改对称**：`clamp(d_prev - d_curr, -0.1, 0.1)` — 远离也惩罚
**删除或大幅弱化 undesired_contacts**（w=0 或 -0.1）— 正常走路不应被惩罚

修改后的数值验证：
```
站着不动:      0/step
朝目标走:     +0.079/step  ← 明确正收益！
反方向走:     -0.041/step  ← 明确负收益！
```

### 共识 3：修复 Actor 观测

**移除 V50 的 yaw 噪声** — 同一物理状态产生不同观测，PPO 会困惑。
**扩大距离 clamp** — d_xy clamp 从 2.0 → 5.0。
论文的 heading 随机性应通过 heading curriculum（reset 时初始朝向）控制，不是在 obs 加噪声。

### 共识 4：先简化架构再加复杂度（训练策略专家提议）

**Phase 0**: flat terrain + 标准 MLP + 精确方向 + 简化奖励 → 3h 内验证
**Phase 1**: rough terrain + resume → 加正则化
**Phase 2**: heading + standing → 完整功能

---

## 三、V51 实验方案

### V51a: 最小改动方案（保持 AME2 网络，改奖励+PPO+obs）

改动：
1. mini_batches: 16 → 4
2. position_tracking: 100 → 0
3. approach 改对称: clamp(-0.1, 0.1)
4. undesired_contacts: -1 → 0
5. actor cmd: 移除噪声，d_xy clamp 5.0
6. position_approach: 200
7. vel_toward_goal: 20
8. moving_to_goal: 20

### V51b: 简化架构方案（标准 MLP + flat terrain）

改动：
1. 用 rsl_rl 默认 ActorCritic（去掉 AME2Encoder, MoE, cross-attention）
2. Flat terrain
3. 同 V51a 的奖励和 PPO 配置
4. Goal distance: [1.5, 4.0]m
5. Episode: 10s（缩短）

### 预期结果

- V51a: 如果 mini_batches=4 不 OOM + 奖励修正 → 1000 iter 内应该有方向收敛信号
- V51b: 3h 内（5000 iter）dxy < 1.0m, succ@1.0 > 30%

---

## 四、风险和 Fallback

- mini_batches=4 OOM → 退到 8
- 标准 MLP 方案太简单学不了 rough terrain → Phase 1 再引入 map
- 对称 approach 导致震荡 → 改成不对称但有弱惩罚: clamp(-0.05, 0.1)

---

## 五、关键教训（50 个版本总结）

1. **不要在弱梯度条件下调奖励函数** — 12.5x 弱的梯度使任何静态正奖励都会变成 exploit
2. **先跑通最简版本** — 50 个版本都在同时对抗复杂网络+小batch+困难地形，应该分而治之
3. **数值计算比直觉更可靠** — "走路应该有正收益" 这个假设从来没被精确验证过，直到今天才发现走路其实是亏的
4. **观测噪声不等于鲁棒性** — V50 的 yaw 噪声破坏了 PPO 的一致性，应该用 curriculum 而非噪声
