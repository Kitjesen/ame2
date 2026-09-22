"""Paired mount evaluation and previews from actual Isaac sensor/mesh exports."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ame2.lidar_mapping import GridSpec
from ame2.networks.ame2_model import MappingConfig
from ame2.networks.lidar_mapping_model import LidarContextMappingNet
from train_mid360_mapping import metrics, predict


def model_prediction(path, dataset):
    cfg = dataset["config"]
    grid = GridSpec(**cfg["grid"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["config"] != cfg or saved.get("model_kind") != "lidar-context":
        raise ValueError("Comparison checkpoint must match its acquisition and model")
    model = LidarContextMappingNet(MappingConfig(map_h=grid.height, map_w=grid.width),
                                  missing_height=grid.missing_height)
    model.load_state_dict(saved["model"])
    model.eval()
    return predict(model, dataset["test"]["raw"], dataset["test"]["observed"])


def mesh_color(name):
    if name == "lidar1_emitter":
        return "#efa733"
    return "#2c9be5" if "camera1" in name.lower() else "#57616e"


def robot_cloud_plot(before, after, grid, output):
    fig = plt.figure(figsize=(14, 6))
    assert before["preview"]["sample_index"] == after["preview"]["sample_index"]
    z_low = min(data["preview"]["truth"].min().item() for data in (before, after)) - .1
    for i, data in enumerate((before, after)):
        preview = data["preview"]
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        xy = grid.xy("cpu").numpy()
        ax.plot_surface(xy[...,0], xy[...,1], preview["truth"][0].numpy(),
                        color="#d2ddd3", alpha=.25, rstride=2, cstride=2, linewidth=0)
        for mesh in preview["meshes"]:
            vertices, faces = mesh["vertices"].numpy(), mesh["faces"].numpy()
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor=mesh_color(mesh["name"]),
                                                edgecolor="none", alpha=1.))
        points = preview["points"].numpy()
        ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap="viridis",
                   vmin=z_low, vmax=0., s=.6, alpha=.75)
        ax.set(xlim=(-.85,2.1), ylim=(-1.05,1.05), zlim=(z_low,.4),
               xlabel="Forward x (m)", ylabel="Left y (m)", zlabel="Height from base (m)")
        ax.set_box_aspect((2.95,2.1,1.95))
        ax.view_init(elev=22, azim=-60)
        ax.set_title("BEFORE: existing mount" if i==0 else "AFTER: proposed mount + optical origin", pad=15)
    fig.suptitle("Thunder V4: actual visual meshes and raw MID-360 returns\nOrange: MID-360 | Blue: front camera | Static sensor test, not a locomotion rollout", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "robot_clouds.png", dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    from matplotlib.collections import PolyCollection
    for i, data in enumerate((before, after)):
        for mesh in data["preview"]["meshes"]:
            if mesh["name"] != "lidar1_emitter" and "camera1" not in mesh["name"].lower():
                continue
            vertices, faces = mesh["vertices"].numpy(), mesh["faces"].numpy()
            faces = faces[np.argsort(vertices[faces, 1].mean(-1))]
            axes[i].add_collection(PolyCollection(vertices[faces][:,:,[0,2]],
                                                   facecolor=mesh_color(mesh["name"]), edgecolor="none"))
        axes[i].set(xlim=(.30,.53), ylim=(-.015,.20), aspect="equal", xlabel="Forward x (m)", ylabel="Height from base (m)")
        axes[i].set_title("Existing housing pose" if i==0 else "Proposed housing pose")
        axes[i].grid(alpha=.15)
    fig.suptitle("Sensor geometry close-up: housing and rays rotate together\nThis is a mount proposal; bracket fit/strength have not been qualified.", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "mount_closeup.png", dpi=155)
    plt.close(fig)


def maps_plot(before, after, old_prediction, prediction, grid, output):
    old, new = before["test"], after["test"]
    truth = new["truth"]
    xy = grid.xy("cpu")
    slope = truth[:,0,xy[...,0]>1.4].mean(-1) - truth[:,0,xy[...,0]<0].mean(-1)
    ids = [int(truth.flatten(1).std(-1).argmin()), int(slope.argmax()), int(slope.argmin())]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for row, idx in enumerate(ids):
        target = truth[idx,0].numpy()
        previous = old["raw"][idx,0].numpy().copy()
        current = new["raw"][idx,0].numpy().copy()
        previous[~old["observed"][idx,0].numpy()] = np.nan
        current[~new["observed"][idx,0].numpy()] = np.nan
        estimate = prediction[idx,0].numpy()
        arrays = [previous, current, old_prediction[idx,0].numpy(), estimate, target, abs(estimate-target)]
        titles = ["Before: raw returns", "After: raw returns", "Before: reconstructed",
                  "After: reconstructed", "Terrain truth", "After: abs error (m)"]
        for col, (array, title) in enumerate(zip(arrays, titles)):
            cmap = plt.get_cmap("viridis" if col<5 else "magma").copy()
            cmap.set_bad("#b7bcc1")
            low, high = (float(target.min()), float(target.max())) if col<5 else (0., .15)
            if high-low < .1: low, high = low-.05, high+.05
            im = axes[row,col].imshow(array, origin="lower", extent=(grid.x_min,grid.x_max,grid.y_min,grid.y_max),
                                      cmap=cmap, vmin=low, vmax=high)
            axes[row,col].set_title(title, fontsize=10)
            axes[row,col].set_xlabel("Forward x (m)")
            fig.colorbar(im, ax=axes[row,col], fraction=.045)
        axes[row,0].set_ylabel(["Flattest test crop", "Largest rise", "Largest descent"][row]+"\nLeft y (m)")
    fig.suptitle("Paired held-out scenes | Grey = no observed return | Same poses, pattern windows and noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "map_comparison.png", dpi=145)
    plt.close(fig)


def coverage_plot(datasets, grid, output):
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    titles = ("Before: existing mount", "Optical-origin correction only",
              "Candidate A: pitch +67.5 deg", "Candidate B: roll -120 deg")
    for ax, data, title in zip(axes, datasets.values(), titles):
        coverage = data["test"]["observed"].float().mean(0)[0].numpy()
        im = ax.imshow(coverage*100, origin="lower", vmin=0, vmax=100, cmap="viridis",
                       extent=(grid.x_min,grid.x_max,grid.y_min,grid.y_max))
        ax.add_patch(Rectangle((.5,-.2),1.5,.4,fill=False,edgecolor="white",linewidth=1.5))
        ax.set(title=title, xlabel="Forward x (m)", ylabel="Left y (m)")
    fig.colorbar(im, ax=axes, label="Frames with an observed return in this cell (%)", shrink=.8)
    fig.suptitle("512 paired static scenes | White box: forward 0.5-2 m, +/-0.2 m | Raw returns before completion")
    fig.savefig(output / "coverage.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mount_comparison")
    parser.add_argument("--proposal", choices=("ground", "side"), default="side")
    args = parser.parse_args()
    root = Path(args.root)
    torch.set_num_threads(4)
    datasets = {key: torch.load(root/key/"dataset.pt", weights_only=True)
                for key in ("legacy","origin","ground","side")}
    baseline = datasets["legacy"]
    for name, data in datasets.items():
        if not data.get("complete"):
            raise ValueError(f"Incomplete acquisition: {name}")
        for split in ("train", "validation", "test"):
            for key in ("pose", "tile", "pattern_start"):
                torch.testing.assert_close(data[split][key], baseline[split][key], atol=0., rtol=0.)
            torch.testing.assert_close(data[split]["truth"], baseline[split]["truth"], atol=2e-4, rtol=0.)
        assert data["preview"]["mesh_rotation_matches_sensor"]
    grid = GridSpec(**baseline["config"]["grid"])
    xy = grid.xy("cpu")
    roi = (xy[...,0]>=.5) & (xy[...,1].abs()<=.2)
    result = {"paired_poses_pattern_and_truth_verified": True,
              "scan_directions":baseline["config"]["samples_per_scan"],
              "test_frames": len(baseline["test"]["raw"]), "mounts":{},
              "preview_proposal": args.proposal,
              "count_columns": ["terrain_returns", "self_occluded_rays", "nonself_returns",
                                "noisy_map_returns", "observed_cells"],
              "limits":["New mount requires physical repositioning; not an extrinsic-only repair",
                        "Static synthetic returns; real calibration/fit/strength not qualified"]}
    predictions = {}
    for name, data in datasets.items():
        test = data["test"]
        record = {"observed_fraction":test["observed"].float().mean().item(),
                  "forward_observed_fraction":test["observed"][:,:,roi].float().mean().item(),
                  "raw_observed_mae_m":(test["raw"]-test["truth"]).abs()[test["observed"]].mean().item(),
                  "counts_mean":test["counts"].float().mean(0).tolist(), "config":data["config"],
                  "paired_native_label_boundary_cells":data.get("paired_native_label_boundary_cells"),
                  "paired_native_label_max_difference_m":data.get("paired_native_label_max_difference_m"),
                  "near_observed_fraction":test["observed"][:,:,xy[...,0]<.5].float().mean().item()}
        if name != "origin":
            mean, lv = model_prediction(root/name/"run/mapping_best.pt", data)
            record["reconstruction"] = metrics(mean,test["truth"],test["observed"],grid,lv)
            predictions[name] = mean
            training = json.loads((root/name/"run/result.json").read_text())
            record["training"] = {key:training[key] for key in
                                  ("status", "steps", "best_step", "train_frames", "validation_frames",
                                   "test_frames", "checkpoint_reload_verified", "test_clean_returns",
                                   "test_clean_observed_mae_m", "nearest_observed_baseline_test")}
        result["mounts"][name] = record
    maps_plot(baseline, datasets[args.proposal], predictions["legacy"], predictions[args.proposal], grid, root)
    robot_cloud_plot(baseline, datasets[args.proposal], grid, root)
    coverage_plot(datasets, grid, root)
    (root/"comparison.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
