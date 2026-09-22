"""Render the geometric blind area and controlled mapping ablations."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.diagnose_mid360_mapping import rotation_rpy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/mid360_diagnosis")
    p.add_argument("--output", default="docs/assets/mid360_diagnosis.png")
    args = p.parse_args()
    root = Path(args.input)
    result = json.loads((root / "result.json").read_text())
    steps = result["training"]["current_sparse"][-1]["steps"]
    maps = np.load(root / "maps.npz")
    cfg = json.loads(Path("configs/thunder_v4_mid360.json").read_text())
    colors = ["#b84336", "#2674a8", "#28865b"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    distances = np.linspace(.45, 2.1, 300)
    origin = np.array(cfg["sensor_translation_m"])
    origin[2] += cfg["fixture_base_height_m"]
    rays = np.stack((distances, distances*0., distances*0.), -1) - origin
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    ax = axes[0, 0]
    ax.axhspan(-7, 52, color="#e0eee4", label="MID-360 specified vertical FOV")
    for rpy, label, color in [(cfg["sensor_rpy_rad"], "Current front mount", colors[0]),
                              (result["counterfactual_rpy_rad"], "Diagnostic mount only", colors[1])]:
        local = rays @ rotation_rpy(rpy)
        elevations = np.degrees(np.arcsin(np.clip(local[:, 2], -1, 1)))
        ax.plot(distances, elevations, color=color, lw=2, label=label)
    ax.set(xlim=(.45, 2.1), ylim=(-10, 95), xlabel="Ground position forward of base x (m)",
           ylabel="Required elevation in sensor frame (deg)", title="A  Forward ground is outside the current FOV")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    sweep = result["pitch_sweep"]
    ax.plot([v["pitch_deg"] for v in sweep], [100*v["ideal_flat_strip_fov_fraction"] for v in sweep],
            color=colors[1], marker="o")
    ax.scatter([-45], [0], color=colors[0], s=80, zorder=3, label="Current configuration")
    ax.set(xlabel="Diagnostic pitch (deg), roll fixed at -180 deg", ylabel="Inside specified FOV (%)", ylim=(-3, 105),
           title="B  Geometric coverage bound, before occlusion")
    ax.text(.03, .94, "Flat ground: x = 0.5–2 m, |y| ≤ 0.2 m\nNo body mesh; not a mounting recommendation",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    x = -.4 + (np.arange(48)+.5)*.05
    row = 20
    ax.step(x, maps["truth"][2, 0, row], where="mid", color="#252b33", lw=2, label="Downstairs truth")
    for case, label, color in zip(("current_sparse", "counterfactual_sparse", "dense_oracle"),
                                   ("Current scan", "Diagnostic scan", "Dense oracle (control)"), colors):
        ax.plot(x, maps[case+"_prediction"][2, 0, row], color=color, lw=1.7, label=label,
                linestyle="--" if case == "dense_oracle" else "-")
    ax.set(xlabel="Forward x (m), y = 0.025 m", ylabel="Height relative to base (m)",
           title=f"C  Downstairs cross-section after {steps:,} steps", xlim=(-.4, 2.))
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    cases = ["current_sparse", "counterfactual_sparse", "dense_oracle"]
    names = ["Current scan", "Diagnostic scan", "Dense oracle"]
    locations = np.arange(3)
    for offset, metric, label, color in [(-.18, "mae_m", "Whole-map MAE", "#617287"),
                                        (.18, "edge_mae_m", "Edge-cell MAE", "#c17b35")]:
        values = [100*result["training"][case][-1][metric] for case in cases]
        bars = ax.bar(locations+offset, values, width=.35, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set(xticks=locations, xticklabels=names, ylabel="Height error (cm)", title="D  Equal training budget, same terrain fixtures")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(100*result["training"][case][-1]["edge_mae_m"] for case in cases)*1.25)
    for ax in axes.flat:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Thunder V4 front MID-360: diagnosis, not a validated locomotion map\n"
                 "Fixed base height 0.5 m | No sensor noise or articulated body | Scan-window holdout only", fontsize=13)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
