"""Analytic flat-ground FOV audit, independent of raycasting and neural networks."""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation_rpy(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thunder_v4_mid360.json")
    parser.add_argument("--output", default="artifacts/mid360_fov")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rotation = rotation_rpy(cfg["sensor_rpy_rad"])
    origin = np.array(cfg["sensor_translation_m"])
    origin[2] += cfg["fixture_base_height_m"]
    targets = np.array([[.5, 0, 0], [1., 0, 0], [2., 0, 0]])
    local = (targets - origin) @ rotation
    elevation = np.degrees(np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1)))
    record = {"assumption": "CAD link axes equal Livox measurement axes; not independently calibrated",
              "base_height_m": cfg["fixture_base_height_m"], "sensor_world_position_m": origin.tolist(),
              "sensor_positive_z_in_base": rotation[:, 2].tolist(), "upper_blind_cone_half_angle_deg": 38,
              "targets": [{"x_from_base_m": float(x), "required_sensor_elevation_deg": float(e),
                           "inside_nominal_fov": bool(-7 <= e <= 52)} for x, e in zip(targets[:, 0], elevation)],
              "excludes": ["robot occlusion", "range noise", "sparse scan pattern", "neural network"]}
    (output / "result.json").write_text(json.dumps(record, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    phi = np.radians(np.linspace(-7, 52, 18))
    for theta in (0., np.pi):
        rays = np.stack((np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)), -1) @ rotation.T
        for ray in rays:
            distance = min(8., -origin[2]/ray[2]) if ray[2] < 0 else 1.
            end = origin + distance * ray
            ax.plot([origin[0], end[0]], [origin[2], end[2]], color="#178f89", alpha=.3)
    axis = rotation[:, 2]
    intercept = origin - origin[2]/axis[2] * axis
    ax.plot([origin[0], intercept[0]], [origin[2], 0], "--", color="#c62828", lw=2,
            label="+Z sensor axis (not a laser ray)")
    for target, e in zip(targets, elevation):
        ax.plot([origin[0], target[0]], [origin[2], 0], ":", color="#c62828", alpha=.6)
        ax.scatter(target[0], 0, color="#c62828")
        ax.text(target[0], -.06, f"x={target[0]:g} m\nneeds {e:.1f} deg", ha="center", va="top", fontsize=9)
    ax.scatter(origin[0], origin[2], color="black", s=50)
    ax.axhline(0, color="black", lw=1)
    ax.text(.65, .78, "Forward-down axis points into\nthe top blind cone", color="#c62828", fontsize=10)
    ax.set(xlim=(-.4, 2.4), ylim=(-.24, 1.), xlabel="Forward from base (m)", ylabel="Height (m)",
           title="Current rotation: roll -180 deg, pitch -45 deg")
    ax.legend(loc="upper right", fontsize=8)
    x = np.linspace(-.4, 2., 480)
    y = np.linspace(-1., 1., 400)
    xx, yy = np.meshgrid(x, y)
    ground = np.stack((xx, yy, np.zeros_like(xx)), -1)
    local = (ground - origin) @ rotation
    el = np.degrees(np.arctan2(local[..., 2], np.linalg.norm(local[..., :2], axis=-1)))
    visible = (el >= -7) & (el <= 52)
    from matplotlib.colors import ListedColormap
    axes[1].imshow(visible, origin="lower", extent=(-.4, 2., -1., 1.),
                   cmap=ListedColormap(["#f5c5be", "#b4dfd7"]), interpolation="nearest")
    axes[1].contour(xx, yy, el, levels=[52], colors=["#c62828"], linewidths=1)
    axes[1].plot([.5, 2, 2, .5, .5], [-.2, -.2, .2, .2, -.2], "k--", lw=1)
    axes[1].text(.9, .02, "Outside FOV", ha="center", color="#912820", fontsize=12)
    axes[1].text(0, .8, "Inside FOV", ha="center", color="#17645e", fontsize=10)
    axes[1].set(xlabel="Forward from base (m)", ylabel="Left (m)", title="Flat ground: FOV only, before body occlusion")
    fig.suptitle("Why pointing the housing downward does not guarantee forward-ground coverage", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "fov.png", dpi=150)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
