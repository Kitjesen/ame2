"""Evaluate the warm start and two training recipes on identical saved scans."""
import argparse
import json
from pathlib import Path

import torch

from ame2.lidar_mapping import GridSpec
from ame2.networks.ame2_model import MappingConfig
from ame2.networks.lidar_mapping_model import LidarContextMappingNet
from train_mid360_mapping import edges, load_initial_weights, metrics, predict


def plot_results(root, test, predictions, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = edges(test["truth"]).flatten(1).sum(-1).argsort(descending=True)
    ids = order[torch.tensor([0, len(order)//3, 2*len(order)//3])]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    for row, index in enumerate(ids):
        target = test["truth"][index, 0].numpy()
        raw = test["raw"][index, 0].numpy().copy()
        raw[~test["observed"][index, 0].numpy()] = float("nan")
        maps = [raw, target] + [predictions[name][index, 0].numpy()
                                for name in ("initial", "control", "edge_mix")]
        maps.append(abs(maps[-1] - target))
        titles = ["Raw scan (grey = unknown)", "Terrain truth", "Previous checkpoint",
                  "Longer training", "Mixed scans + edge loss", "New height error (m)"]
        for col, values in enumerate(maps):
            cmap = plt.get_cmap("viridis" if col < 5 else "magma").copy()
            cmap.set_bad("#b7bcc1")
            low, high = (float(target.min()), float(target.max())) if col < 5 else (0., .15)
            if high-low < .1:
                low, high = low-.05, high+.05
            im = axes[row,col].imshow(values, origin="lower", cmap=cmap, vmin=low, vmax=high,
                                      extent=(grid.x_min,grid.x_max,grid.y_min,grid.y_max))
            axes[row,col].set(title=titles[col], xlabel="Forward x (m)")
            axes[row,col].title.set_fontsize(10)
            fig.colorbar(im, ax=axes[row,col], fraction=.045)
        axes[row,0].set_ylabel("Left y (m)")
    fig.suptitle("Thunder V4 MID-360 | Same test scans | Single-frame mapping, before history fusion")
    fig.tight_layout()
    fig.savefig(root/"comparison.png", dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mapping_round2")
    parser.add_argument("--initial-checkpoint", default="artifacts/mount_comparison/side/run/mapping_best.pt")
    args = parser.parse_args()
    root = Path(args.root)
    torch.set_num_threads(4)
    data = torch.load(root/"dataset.pt", map_location="cpu", weights_only=True)
    if not data["complete"]:
        raise ValueError("Acquisition is incomplete")
    grid = GridSpec(**data["config"]["grid"])
    test = data["test"]
    result = {"config":data["config"], "frames":{s:len(data[s]["raw"]) for s in ("train","validation","test")},
              "tile_splits":data["tile_splits"], "models":{},
              "limits":["Static synthetic scans, no real sensor calibration or locomotion evaluation",
                        "Original held-out tile boundary preserved; these tiles were used in the prior mount study",
                        "Each model evaluated on the same noisy and clean scans"]}
    predictions = {}
    for name in ("initial", "control", "edge_mix"):
        path = Path(args.initial_checkpoint) if name == "initial" else root/name/"mapping_best.pt"
        model = LidarContextMappingNet(MappingConfig(map_h=grid.height,map_w=grid.width),
                                      missing_height=grid.missing_height)
        step = load_initial_weights(model, path, data["config"], "lidar-context", "cpu")
        model.eval()
        mean, lv = predict(model, test["raw"], test["observed"])
        clean_mean, clean_lv = predict(model, test["clean_raw"], test["clean_observed"])
        predictions[name] = mean
        result["models"][name] = {"checkpoint":str(path), "selected_step":step,
                                   "noisy":metrics(mean,test["truth"],test["observed"],grid,lv),
                                   "clean":metrics(clean_mean,test["truth"],test["clean_observed"],grid,clean_lv)}
        if name != "initial":
            training = json.loads((root/name/"result.json").read_text())
            result["models"][name]["training"] = {key:training[key] for key in
                ("status","steps","best_step","initial_checkpoint_step","clean_frame_fraction",
                 "edge_loss_weight","validation_objective","checkpoint_reload_verified","training_seconds")}
    plot_results(root, test, predictions, grid)
    (root/"comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
