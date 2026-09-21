"""Plot saved fixture evidence without launching Isaac Sim."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/mid360_validation/maps.npz")
    parser.add_argument("--output", default="docs/assets/mid360_validation.png")
    args = parser.parse_args()
    data = np.load(args.input)
    fig, axes = plt.subplots(4, 4, figsize=(13, 11), layout="constrained")
    height_cmap = plt.get_cmap("viridis").copy()
    height_cmap.set_bad("#d4d8dc")
    names = ["Flat", "Ascending stairs", "Descending stairs", "Occluded step"]
    columns = ["Actual MID-360 projection", "Terrain supervision", "MappingNet: 300 steps", "Predicted sigma (m)"]
    for row, name in enumerate(names):
        raw = np.ma.masked_where(~data["observed"][row, 0], data["raw"][row, 0])
        for col, values in enumerate((raw, data["truth"][row, 0], data["predicted"][row, 0], data["sigma"][row, 0])):
            ax = axes[row, col]
            im = ax.imshow(values, origin="lower", extent=(-.4, 2., -1., 1.),
                           interpolation="nearest", aspect="equal",
                           cmap=height_cmap if col < 3 else "magma",
                           vmin=-1.1 if col < 3 else 0., vmax=.4 if col < 3 else .5)
            if row == 0:
                ax.set_title(columns[col], fontsize=10)
            if col == 0:
                coverage = data["observed"][row, 0].mean() * 100
                ax.set_ylabel(f"{name}\nLateral y (m)")
                ax.text(.03, .95, f"{coverage:.1f}% observed", transform=ax.transAxes,
                        va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
            if row == 3:
                ax.set_xlabel("Forward x (m)")
            ax.set_xticks([0., 1., 2.])
            ax.set_yticks([-1., 0., 1.])
            if col == 2:
                height_image = im
            if col == 3:
                sigma_image = im
    fig.colorbar(height_image, ax=axes[:, :3], shrink=.7, label="Height relative to base (m)")
    fig.colorbar(sigma_image, ax=axes[:, 3], shrink=.7, label="Sigma (m); display clipped at 0.5")
    fig.suptitle("Thunder V4 extrinsics + MID-360 pattern | Stationary fixtures, no robot body mesh\n"
                 "Gray = no return. Same-terrain scan holdout; learned completion is not validated for locomotion.", fontsize=12)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
