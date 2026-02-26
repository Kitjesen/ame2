"""
MappingNet Training Pipeline
=============================
Trains the MappingNet (from ame2/networks/ame2_model.py) on procedurally
generated synthetic terrain with simulated noisy depth scans.

Components:
    HeightFieldGenerator  - Procedural terrain generation
    DepthScanSimulator    - Noisy depth scan simulation
    MappingTrainer        - Training loop with beta-NLL + TV weighting
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ame2.networks import MappingNet, MappingConfig


# ---------------------------------------------------------------------------
# 1. HeightFieldGenerator
# ---------------------------------------------------------------------------

class HeightFieldGenerator:
    """
    Procedurally generates synthetic terrain height fields.

    Three generation methods are randomly mixed per batch:
        1. Superimposed sine waves (random frequency / amplitude)
        2. Random box stacking (0.05 - 0.4m tall)
        3. Gaussian noise field (smoothed)
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device

    def _sine_terrain(self, B: int, H: int, W: int) -> torch.Tensor:
        hf = torch.zeros(B, 1, H, W, device=self.device)
        xs = torch.linspace(0, 1, W, device=self.device).view(1, 1, 1, W)
        ys = torch.linspace(0, 1, H, device=self.device).view(1, 1, H, 1)
        for _ in range(5):
            freq_x = torch.rand(B, 1, 1, 1, device=self.device) * 8.0 + 1.0
            freq_y = torch.rand(B, 1, 1, 1, device=self.device) * 8.0 + 1.0
            amp = torch.rand(B, 1, 1, 1, device=self.device) * 0.15 + 0.02
            phase_x = torch.rand(B, 1, 1, 1, device=self.device) * 2 * np.pi
            phase_y = torch.rand(B, 1, 1, 1, device=self.device) * 2 * np.pi
            hf += amp * torch.sin(freq_x * xs * 2 * np.pi + phase_x) \
                      * torch.cos(freq_y * ys * 2 * np.pi + phase_y)
        return hf

    def _box_terrain(self, B: int, H: int, W: int) -> torch.Tensor:
        hf = torch.zeros(B, 1, H, W, device=self.device)
        for _ in range(8):
            for b in range(B):
                bh = np.random.randint(3, max(4, H // 3))
                bw = np.random.randint(3, max(4, W // 3))
                y0 = np.random.randint(0, H - bh + 1)
                x0 = np.random.randint(0, W - bw + 1)
                height = np.random.uniform(0.05, 0.4)
                hf[b, 0, y0:y0 + bh, x0:x0 + bw] += height
        return hf

    def _gaussian_terrain(self, B: int, H: int, W: int) -> torch.Tensor:
        noise = torch.randn(B, 1, H, W, device=self.device) * 0.3
        k, pad = 5, 2
        kernel = torch.ones(1, 1, k, k, device=self.device) / (k * k)
        smoothed = nn.functional.conv2d(noise, kernel, padding=pad)
        return nn.functional.conv2d(smoothed, kernel, padding=pad)

    def generate(self, B: int, H: int = 31, W: int = 51) -> torch.Tensor:
        n1, n2, n3 = B // 3, B // 3, B - 2 * (B // 3)
        parts = []
        if n1 > 0:
            parts.append(self._sine_terrain(n1, H, W))
        if n2 > 0:
            parts.append(self._box_terrain(n2, H, W))
        if n3 > 0:
            parts.append(self._gaussian_terrain(n3, H, W))
        hf = torch.cat(parts, dim=0)
        perm = torch.randperm(B, device=self.device)
        return hf[perm]


# ---------------------------------------------------------------------------
# 2. DepthScanSimulator
# ---------------------------------------------------------------------------

class DepthScanSimulator:
    """
    Simulates noisy depth scans from ground-truth height fields.

    Noise model:
        - Range truncation: cells with abs(elevation) > 2.0m are set to 0
        - Gaussian noise:   sigma = 0.02m
        - Random dropout:   20% of cells set to 0
    """

    def __init__(self, range_max: float = 2.0, noise_std: float = 0.02,
                 dropout_rate: float = 0.20,
                 device: torch.device = torch.device("cpu")):
        self.range_max = range_max
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.device = device

    def simulate(self, gt: torch.Tensor) -> torch.Tensor:
        scan = gt.clone()
        scan = scan + torch.randn_like(scan) * self.noise_std
        scan[scan.abs() > self.range_max] = 0.0
        scan[torch.rand_like(scan) < self.dropout_rate] = 0.0
        return scan


# ---------------------------------------------------------------------------
# 3. MappingTrainer
# ---------------------------------------------------------------------------

class MappingTrainer:
    """Training loop for MappingNet."""

    def __init__(self, device: torch.device = torch.device("cpu"), lr: float = 1e-3):
        self.device = device
        self.model = MappingNet(MappingConfig()).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.terrain_gen = HeightFieldGenerator(device=device)
        self.scan_sim = DepthScanSimulator(device=device)
        self.loss_history = []

    def train_step(self, B: int = 32) -> float:
        self.model.train()
        gt = self.terrain_gen.generate(B)
        noisy_scan = self.scan_sim.simulate(gt)
        pred_elev, log_var = self.model(noisy_scan)
        tv_weights = MappingNet.total_variation_weight(gt)
        loss = MappingNet.beta_nll_loss(pred_elev, log_var, gt,
                                        beta=0.5, tv_weights=tv_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    @torch.no_grad()
    def visualize(self, step: int, output_dir: str = "outputs"):
        self.model.eval()
        gt = self.terrain_gen.generate(1)
        noisy_scan = self.scan_sim.simulate(gt)
        pred_elev, log_var = self.model(noisy_scan)

        noisy_np = noisy_scan[0, 0].cpu().numpy()
        pred_np = pred_elev[0, 0].cpu().numpy()
        unc_np = torch.exp(log_var[0, 0]).cpu().numpy()
        gt_np = gt[0, 0].cpu().numpy()

        vmin = min(gt_np.min(), pred_np.min(), noisy_np.min())
        vmax = max(gt_np.max(), pred_np.max(), noisy_np.max())

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, data, title, cmap, v in zip(
            axes,
            [noisy_np, pred_np, unc_np, gt_np],
            ["Noisy Input", "Predicted Elevation", "Uncertainty (variance)", "Ground Truth"],
            ["terrain", "terrain", "hot", "terrain"],
            [(vmin, vmax), (vmin, vmax), (None, None), (vmin, vmax)],
        ):
            im = ax.imshow(data, cmap=cmap,
                           vmin=v[0], vmax=v[1],
                           aspect="auto", origin="lower")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"MappingNet - Step {step}", fontsize=14)
        plt.tight_layout()
        path = os.path.join(output_dir, f"mapping_{step:04d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved visualization: {path}")

    def plot_loss(self, output_dir: str = "outputs"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.loss_history, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (beta-NLL + TV)")
        ax.set_title("MappingNet Training Loss")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "mapping_loss.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved loss curve: {path}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = "outputs/mapping"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trainer = MappingTrainer(device=device, lr=1e-3)

    n_steps = 100
    vis_interval = 20

    print(f"Training MappingNet for {n_steps} steps (batch_size=32)...")
    print("-" * 50)

    for step in range(1, n_steps + 1):
        loss = trainer.train_step(B=32)
        if step % 10 == 0 or step == 1:
            print(f"Step {step:4d}/{n_steps}  loss={loss:.5f}")
        if step % vis_interval == 0:
            trainer.visualize(step, output_dir)

    print("-" * 50)
    print("Training complete.")
    trainer.plot_loss(output_dir)
    print(f"Outputs saved to: {output_dir}")
