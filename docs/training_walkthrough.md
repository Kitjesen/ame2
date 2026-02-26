# AME-2 Training Walkthrough

End-to-end guide: Phase 0 → Phase 1 → Phase 2, parameter explanations, and monitoring tips.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | |
| PyTorch | ≥ 2.0 | CUDA 12.x recommended |
| **Isaac Sim** | 4.x | Phase 1 / 2 only |
| **Isaac Lab** | latest | `pip install -e source/isaaclab` |
| isaaclab_assets | — | ANYmal-D USD + actuator nets |
| isaaclab_rl (RSL-RL) | — | PPO + distillation runners |
| robot_lab | — | `pip install -e source/robot_lab` |

**Phase 0 (MappingNet pretraining) requires only PyTorch — no Isaac Sim.**

### Install the AME-2 package

```bash
pip install -e .            # installs ame2 package (torch + tensordict only)
pip install -e ".[dev]"     # + pytest, matplotlib, numpy
```

---

## Directory Layout After Full Training

```
logs/
├── mapping_net.pt                   ← Phase 0 output
├── mapping_vis/                     ← MappingNet training visualisations
│   ├── mapping_0500.png
│   └── mapping_loss.png
├── ame2_teacher_YYYYMMDD_HHMMSS/    ← Phase 1 output
│   ├── model_10000.pt
│   ├── model_20000.pt
│   └── model_80000.pt               ← final teacher checkpoint
└── ame2_student_YYYYMMDD_HHMMSS/    ← Phase 2 output
    ├── model_10000.pt
    └── model_40000.pt               ← final student checkpoint
```

---

## One-Command Full Pipeline

```bash
# Full training (paper scale — requires 8× RTX-4090 and Isaac Sim):
bash scripts/train_all.sh

# Single-GPU debug run (reduced envs):
bash scripts/train_all.sh --num_envs 512 --mapping_steps 200

# Skip Phase 0 if mapping_net.pt already exists:
bash scripts/train_all.sh --phase_start 1

# Start from Phase 2 with a known teacher checkpoint:
bash scripts/train_all.sh --phase_start 2 \
    --teacher_ckpt logs/ame2_teacher_20260301_120000/model_80000.pt
```

---

## Phase 0 — MappingNet Pretraining

**What it does:** Trains a lightweight U-Net (9 475 params) to predict
elevation mean + log-variance from noisy depth scans using the β-NLL loss
(Eq. 9) with TV reweighting (Eq. 10).  No Isaac Sim required.

### Command

```bash
# Quick demo (100 steps, ~30 seconds on CPU):
python scripts/train_mapping.py

# Production run (~1 hr on GPU):
python scripts/train_mapping.py \
    --num_steps  50000 \
    --batch_size 64 \
    --lr         1e-3 \
    --save_path  logs/mapping_net.pt \
    --output_dir logs/mapping_vis \
    --vis_interval 500
```

Or via the unified launcher (automatically saves to `logs/mapping_net.pt`):

```bash
python scripts/train_ame2.py --phase 0
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num_steps` | 100 | Gradient steps. Paper trains until β-NLL plateaus; ~50k is sufficient |
| `--batch_size` | 32 | Samples per step. 64 is faster on GPU |
| `--lr` | 1e-3 | Adam learning rate |
| `--save_path` | `logs/mapping_net.pt` | Output weights file |
| `--vis_interval` | 20 | Save visualisation every N steps (0 = off) |

### What to look for

- Loss should drop from ~0.5 → ~0.05–0.10 within a few thousand steps.
- Visualisations show: **Noisy Input** · **Predicted Elevation** · **Uncertainty** · **GT**
- Uncertainty (variance) should be high over steep edges and occluded cells.

---

## Phase 1 — Teacher PPO

**What it does:** Trains the teacher policy with full access to ground-truth
elevation maps and full proprioception (including base linear velocity).
Uses asymmetric actor-critic with MoE critic.

**Duration:** ~80 000 PPO iterations, ≈60 RTX-4090-days at 4 800 envs.

### Command

```bash
python scripts/train_ame2.py \
    --phase       1 \
    --num_envs    4800 \
    --log_dir     logs \
    --device      cuda \
    --seed        42 \
    --headless
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num_envs` | 4800 | Parallel environments. Paper: 4 800 (8 GPUs). Single GPU: 512–1024 |
| `--log_dir` | `logs` | Base directory for checkpoints + TensorBoard logs |
| `--max_iterations` | 80 000 | Override iteration count for debugging |
| `--resume` | — | Path to previous log directory to continue training |
| `--seed` | 42 | Random seed |

### Training Curricula (auto-managed, Sec. IV-D.3 / IV-E)

| Curriculum | Schedule | Range |
|------------|----------|-------|
| Entropy coefficient | Linear decay over all 80k iter | 0.004 → 0.001 |
| Perception noise (scan σ) | Linear ramp in first 20% (16k iter) | 0 → `SCAN_NOISE_STD_MAX` |
| Heading constraint | Linear ramp in first 20% (16k iter) | face-goal → ±π random |
| Terrain level (EMA) | Per-episode, EMA α=0.1 | promote if EMA>0.5; demote if EMA≤0.5 & d>4m |

### Resume Training

```bash
python scripts/train_ame2.py --phase 1 \
    --resume logs/ame2_teacher_20260301_120000/
```

---

## Phase 2 — Student Distillation + PPO

**What it does:** Distills the teacher into a student that replaces the
GT elevation map with the neural map (4-channel MappingNet output) and
replaces the plain MLP proprioception encoder with the LSIO module.

**Duration:** ~40 000 iterations, ≈30 RTX-4090-days.

### Command

```bash
python scripts/train_ame2.py \
    --phase        2 \
    --teacher_ckpt logs/ame2_teacher_20260301_120000/model_80000.pt \
    --num_envs     4800 \
    --log_dir      logs \
    --device       cuda \
    --headless
```

If `--teacher_ckpt` is omitted, the script auto-detects the latest
`logs/ame2_teacher_*/` directory.

### Two-Phase Learning Rate Schedule (auto-managed)

| Sub-phase | Iterations | LR | PPO enabled |
|-----------|------------|-----|-------------|
| Pure distillation | 0 – 5 000 | 1e-3 | No |
| PPO + distillation | 5 000 – 40 000 | 1e-4 | Yes |

**Loss:** `L = L_PPO + 0.02·L_distill + 0.2·L_repr`

### Domain Randomisation (auto-applied in Phase 2)

Configured inside `AME2MapEnvWrapper`:

| Randomisation | Default | Effect |
|---------------|---------|--------|
| Scan dropout | 10% | Depth points zeroed (missing returns) |
| Scan artifact | σ=0.5 m, ~5% pixels | Gaussian noise spikes |
| Partial map | 50% of envs | Local-only map (no global WTA history) |
| Map cell corruption | 20% per step | Random elevation + INF variance sentinel |
| Pose drift | ±3 cells (±24 cm) | Random map-frame drift |

---

## GPU Scaling Guide

| Setup | `--num_envs` | Expected speed |
|-------|-------------|----------------|
| 1× RTX-4090 (24 GB) | 512–1 024 | ~8× slower than paper |
| 4× RTX-4090 | 2 400 | ~2× slower than paper |
| 8× RTX-4090 (paper) | 4 800 | baseline |
| A100 80 GB | 2 400–4 800 | comparable |

For multi-GPU, Isaac Lab handles the parallelism automatically via Isaac Sim's
GPU pipeline — no code changes needed.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir logs/ --port 6006
```

### Key Metrics to Watch

| Metric | Expected behaviour |
|--------|--------------------|
| `ep_rew_mean` | Increases monotonically in Phase 1 |
| `d_xy` (distance to goal) | Decreases as terrain curriculum advances |
| `terrain_level` | Slowly climbs as EMA success rate improves |
| `beta_nll_loss` (Phase 0) | Drops from ~0.5 to ~0.05–0.10 |
| `distill_loss` (Phase 2) | Drops sharply in first 5k iter |
| `entropy` | Decays from 0.004 → 0.001 in Phase 1 |

---

## Unit Tests (No Isaac Sim Required)

Before running full training, verify your installation with the 19 included tests:

```bash
pytest scripts/test_ame2.py -v
```

All 19 tests should pass. They cover: MappingNet, WTAMapFusion, LSIO,
AME2Encoder, StudentLoss, AsymmetricCritic, and L-R symmetry augmentation.

---

## Common Issues

### `ImportError: No module named 'isaaclab'`

Phase 1 / 2 require Isaac Sim. Phase 0 and unit tests do not.
Install Isaac Lab following [official instructions](https://isaac-sim.github.io/IsaacLab/).

### `FileNotFoundError: logs/mapping_net.pt`

Phase 0 was not run or saved to a different path.  Run:
```bash
python scripts/train_ame2.py --phase 0
```

### `RuntimeError: CUDA out of memory`

Reduce `--num_envs`. Each environment uses ~3–5 MB GPU memory.
A single RTX-4090 can handle ~512–1 024 envs comfortably.

### Phase 2 not improving

- Ensure teacher checkpoint is from iteration 80 000 (fully converged).
- Check that `logs/mapping_net.pt` exists and is a valid Phase 0 output.
- The first 5 000 iterations are pure distillation — PPO loss will be 0 initially.

---

## Citation

```bibtex
@article{zhang2025ame2,
  title   = {{AME-2}: Agile and Generalized Legged Locomotion via
              Attention-Based Neural Map Encoding},
  author  = {Zhang, Chong and Klemm, Victor and Yang, Fan and Hutter, Marco},
  year    = {2025},
  url     = {https://arxiv.org/abs/2601.08485}
}
```
