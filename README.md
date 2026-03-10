# AME-2: Unofficial Reimplementation

Unofficial reimplementation of **AME-2** — goal-conditioned navigation for legged robots on rough terrains.

> Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter (ETH Zurich RSL)
> *"AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding"*
> [arXiv:2601.08485](https://arxiv.org/abs/2601.08485)

**Robot:** ANYmal-D (12-DoF quadruped)
**Sim:** Isaac Lab 0.46.x (Direct Workflow) + RSL-RL (PPO)
**Status:** Phase 1 Teacher training in progress (V43l, iteration ~1150)

> This is an independent reimplementation for research/learning, not official code.

---

## Current Results

**V43l** (2048 envs, RTX 3090, training from iteration 801):

| Metric | Value | Trend |
|--------|-------|-------|
| moving_to_goal | 0.32 | rising |
| episode length | ~100 steps (2s) | recovering (bottom was 40) |
| terminal_dxy | ~3.8m | slowly decreasing |
| success@0.5m | 0% | not yet reaching goals |

Robot has learned to move toward goals and avoid knee crawling. Still falling early (~2s episodes). Training ongoing — expect improvement over next few thousand iterations as episode length grows.

---

## How It Works

```
Phase 0 ── Pretrain MappingNet (no sim, ~1hr GPU)
Phase 1 ── Teacher PPO (80K iters, Isaac Sim)     ← current
Phase 2 ── Student Distillation + PPO (40K iters)
```

**Teacher architecture:**
```
Height Map (31×51@4cm) → MappingNet (UNet) → Policy Map (14×36@8cm)
                                                      ↓
Proprioception (48D) → PropEncoder (128D) → Cross-Attention (16 heads)
                                                      ↓
                                           map_emb(192D) + prop_emb(128D)
                                                      ↓
                                              MLP → 12 joint targets
```

---

## Quick Start

### Prerequisites

- NVIDIA GPU (RTX 3090+ recommended, 24GB VRAM)
- Isaac Sim 5.0 + Isaac Lab 0.46.x
- Python 3.10, PyTorch 2.x

### Install

```bash
# 1. Install Isaac Lab (follow official docs)
# https://isaac-sim.github.io/IsaacLab/

# 2. Clone this repo
git clone https://github.com/Kitjesen/ame2.git
cd ame2

# 3. Install network package
pip install -e .

# 4. Verify (no Isaac Sim needed)
pytest scripts/test_ame2.py -v   # 19 tests
```

### Train

```bash
# Single GPU training (RTX 3090, ~2048 envs max)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs/gpu0 --headless

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/train_ame2_direct.py \
    --num_envs 2048 --seed 42 --log_dir logs/gpu0 \
    --resume logs/gpu0/model_800.pt --headless

# Multi-GPU: run separate processes on different GPUs
CUDA_VISIBLE_DEVICES=0 python scripts/train_ame2_direct.py --seed 42 --log_dir logs/gpu0 --headless &
CUDA_VISIBLE_DEVICES=1 python scripts/train_ame2_direct.py --seed 43 --log_dir logs/gpu1 --headless &
```

Checkpoints saved every 50 iterations to `log_dir/model_*.pt`.

### Record Video

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play_record.py \
    --checkpoint logs/gpu0/model_1000.pt \
    --num_envs 4 --num_steps 500 --headless --output record.mp4
```

### Deploy to Remote Server

```bash
# Upload code (server has no git)
scp -r ame2_standalone/ user@server:/path/to/

# On server: install Isaac Lab, then
cd /path/to/ame2_standalone
pip install -e .

# Start training with nohup (PYTHONUNBUFFERED=1 required for log output)
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_ame2_direct.py \
    --num_envs 2048 --headless --log_dir logs/gpu0' > train.log 2>&1 &
```

---

## Project Structure

```
ame2/                       # Network package (pip install -e .)
├── networks/
│   ├── ame2_model.py       # MappingNet, AME2Encoder, AME2Policy, Critic, LSIO
│   └── rslrl_wrapper.py    # RSL-RL wrapper: AME2ActorCritic

ame2_direct/                # Direct Workflow environment
├── config.py               # All hyperparameters (rewards, terminations, PPO)
├── env.py                  # DirectRLEnv (~960 lines)
└── wrapper.py              # RSL-RL compatible obs wrapper

scripts/
├── train_ame2_direct.py    # Training script (Phase 1)
├── play_record.py          # Video recording
├── train_mapping.py        # Phase 0: MappingNet pretraining
└── test_ame2.py            # Unit tests
```

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

## License

Apache-2.0
