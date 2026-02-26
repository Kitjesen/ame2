# AME-2: Attention-Based Neural Map Encoding for Legged Locomotion

Standalone implementation of **AME-2** — a three-phase training pipeline for agile
legged locomotion on ANYmal-D, reproducing the paper:

> Zhang et al., "AME-2: Agile and Generalized Legged Locomotion via Attention-Based
> Neural Map Encoding", arXiv:2601.08485

## Repository Structure

```
ame2/                    # Installable Python package
├── __init__.py          # Gymnasium env registration
├── ame2_env_cfg.py      # Isaac Lab ManagerBasedRLEnv config
├── rewards.py           # AME-2 reward functions (Table I, Eq.1-5)
├── terrains.py          # 12 procedural terrain types (Sec.V-B)
├── networks/
│   ├── ame2_model.py    # AME2Policy, MappingNet, WTAMapFusion (pure PyTorch)
│   └── rslrl_wrapper.py # RSL-RL integration + domain randomization
└── agents/
    └── rsl_rl_cfg.py    # PPO / distillation runner configs

scripts/
├── train_ame2.py        # Three-phase training launcher
├── train_mapping.py     # Phase 0: standalone MappingNet pretraining
└── test_ame2.py         # Unit tests (no Isaac Sim needed)

paper/
├── AME2-Attention-Neural-Map.pdf
└── AME2/                # MinerU-parsed paper (markdown + figures)
```

## Key Architecture (from the paper)

- **Teacher Policy**: asymmetric actor-critic with privileged height scan + contact forces
- **Student Policy**: distilled from teacher; uses only depth camera + WTA global map
- **MappingNet**: lightweight U-Net (9,475 params), gated output, β-NLL loss, TV-weighted training
- **WTA Map Fusion**: deterministic Winner-Takes-All per-cell (min variance wins), 400×400 @ 8 cm

## Quick Start

### 1. Install

```bash
# Requires Isaac Sim + Isaac Lab + robot_lab
pip install -e .
```

### 2. Phase 0 — Pretrain MappingNet (no Isaac Sim)

```bash
cd scripts
python train_ame2.py --phase 0
# Saves logs/mapping_net.pt (~30 min on GPU)
```

### 3. Phase 1 — Teacher PPO (needs Isaac Sim)

```bash
python scripts/train_ame2.py --phase 1 --num_envs 4800
# 80,000 iterations, ~12 hr on 4×A100
```

### 4. Phase 2 — Student Distillation + PPO

```bash
python scripts/train_ame2.py --phase 2 \
    --teacher_ckpt logs/ame2_teacher_<timestamp>/model_80000.pt
# 40,000 iterations (5k pure distillation + 35k PPO)
```

### 5. Run unit tests (no Isaac Sim)

```bash
python scripts/test_ame2.py
```

## Domain Randomization (Sec. IV-D.3)

Configure on the `AME2MapEnvWrapper`:

```python
env.set_student_scan_degradation(dropout_rate=0.05, artifact_std=0.3)
env.set_map_randomization(
    partial_fraction=0.3,    # 30% envs get local-only map
    drop_fraction=0.05,      # 5% map cells corrupted
    drift_max_cells=3,       # ±3 cells pose drift
)
```

## Dependencies

- NVIDIA Isaac Sim 4.x + Isaac Lab
- `robot_lab` (https://github.com/fan-ziqi/robot_lab)
- PyTorch ≥ 2.0, tensordict ≥ 0.3
- RSL-RL (`isaaclab_rl.rsl_rl`)

## Reference

```bibtex
@article{zhang2025ame2,
  title  = {AME-2: Agile and Generalized Legged Locomotion via
             Attention-Based Neural Map Encoding},
  author = {Zhang et al.},
  year   = {2025},
  url    = {https://arxiv.org/abs/2601.08485}
}
```
