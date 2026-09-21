"""Train only AME-2 MappingNet on saved, terrain-disjoint LiDAR acquisitions."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ame2.lidar_mapping import GridSpec
from ame2.networks.ame2_model import MappingConfig, MappingNet
from ame2.networks.lidar_mapping_model import LidarContextMappingNet


def edges(height, threshold=.05):
    mask = torch.zeros_like(height, dtype=torch.bool)
    dx = (height[..., 1:] - height[..., :-1]).abs() > threshold
    dy = (height[..., 1:, :] - height[..., :-1, :]).abs() > threshold
    mask[..., 1:] |= dx
    mask[..., :-1] |= dx
    mask[..., 1:, :] |= dy
    mask[..., :-1, :] |= dy
    return mask


def sample_weights(truth, uniform_fraction=.1):
    """Keep flat examples trainable while emphasizing rough terrain as in Eq.10.

    Exact paper TV assigns flat maps zero weight, including an all-flat batch.
    This LiDAR adaptation mixes 10% uniform loss with 90% normalized TV loss.
    """
    tv = MappingNet.total_variation_weight(truth)
    tv = torch.where(tv.sum() > 0, tv / tv.sum().clamp_min(1e-8), torch.ones_like(tv) / len(tv))
    return (1 - uniform_fraction) * tv + uniform_fraction / len(tv)


def forward_map(model, inputs, observed):
    return model(inputs, observed) if isinstance(model, LidarContextMappingNet) else model(inputs)


@torch.no_grad()
def predict(model, inputs, observed):
    outputs = [forward_map(model, x, mask) for x, mask in zip(inputs.split(64), observed.split(64))]
    return torch.cat([x[0] for x in outputs]), torch.cat([x[1] for x in outputs])


@torch.no_grad()
def nearest_fill(raw, observed, grid):
    xy = grid.xy(raw.device).reshape(-1, 2)
    distance = torch.cdist(xy, xy)
    outputs = []
    for x, mask in zip(raw.split(8), observed.split(8)):
        index = distance[None].masked_fill(~mask.flatten(1)[:, None], torch.inf).argmin(-1)
        result = x.flatten(1).gather(1, index).reshape_as(x)
        result[~mask.flatten(1).any(-1)] = grid.missing_height
        outputs.append(result)
    return torch.cat(outputs)


@torch.no_grad()
def metrics(mean, truth, observed, grid, log_var=None):
    error = (mean - truth).abs()
    boundary = edges(truth)
    xy = grid.xy(truth.device)
    roi = ((xy[..., 0] >= .5) & (xy[..., 1].abs() <= .2))[None, None].expand_as(error)
    masked_mean = lambda x, mask: x[mask].mean().item() if mask.any() else None
    result = {"mae_m": error.mean().item(), "rmse_m": error.square().mean().sqrt().item(),
              "observed_mae_m": masked_mean(error, observed),
              "unknown_mae_m": masked_mean(error, ~observed),
              "edge_mae_m": masked_mean(error, boundary),
              "forward_strip_mae_m": masked_mean(error, roi),
              "forward_strip_observed_fraction": masked_mean(observed.float(), roi),
              "observed_fraction": observed.float().mean().item()}
    predicted_edge = edges(mean)
    truth_near = F.max_pool2d(boundary.float(), 3, 1, 1).bool()
    pred_near = F.max_pool2d(predicted_edge.float(), 3, 1, 1).bool()
    result["edge_precision_within_one_cell"] = masked_mean(truth_near.float(), predicted_edge)
    result["edge_recall_within_one_cell"] = masked_mean(pred_near.float(), boundary)
    if log_var is not None:
        sigma = (log_var * .5).exp()
        result.update({"mean_sigma_m": sigma.mean().item(),
                       "observed_mean_sigma_m": masked_mean(sigma, observed),
                       "unknown_mean_sigma_m": masked_mean(sigma, ~observed),
                       "coverage_1sigma": (error <= sigma).float().mean().item(),
                       "coverage_2sigma": (error <= 2 * sigma).float().mean().item(),
                       "low_sigma_fraction": (sigma < .05).float().mean().item(),
                       "low_sigma_mae_m": masked_mean(error, sigma < .05)})
        ordered = sigma.flatten().argsort()
        result["uncertainty_quartiles"] = []
        for ids in ordered.chunk(4):
            result["uncertainty_quartiles"].append({
                "mean_sigma_m": sigma.flatten()[ids].mean().item(),
                "rmse_m": error.flatten()[ids].square().mean().sqrt().item()})
    return result


def make_plot(output, raw, truth, observed, mean, log_var, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Select edge-rich unseen examples deterministically, without selecting by error.
    ids = edges(truth).flatten(1).sum(-1).argsort(descending=True)[::max(1, len(raw) // 3)][:3]
    fig, axes = plt.subplots(len(ids), 5, figsize=(15, 3 * len(ids)), squeeze=False)
    for row, idx in enumerate(ids):
        target = truth[idx, 0].cpu().numpy()
        scan = raw[idx, 0].cpu().numpy().copy()
        scan[~observed[idx, 0].cpu().numpy()] = float("nan")
        prediction = mean[idx, 0].cpu().numpy()
        data = [scan, target, prediction, abs(prediction - target), (.5 * log_var[idx, 0]).exp().cpu().numpy()]
        titles = ["MID-360 (grey = no return)", "Terrain truth", "MappingNet height", "Absolute error (m)", "Predicted sigma (m)"]
        for col, (values, title) in enumerate(zip(data, titles)):
            cmap = plt.get_cmap("viridis" if col < 3 else "magma").copy()
            cmap.set_bad("#bbbbbb")
            lim = dict(vmin=float(target.min()), vmax=float(target.max()) + 1e-5) if col < 3 else dict(vmin=0, vmax=.3)
            im = axes[row, col].imshow(values, origin="lower", cmap=cmap,
                extent=(grid.x_min, grid.x_max, grid.y_min, grid.y_max), **lim)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xlabel("Forward x (m)")
            fig.colorbar(im, ax=axes[row, col], fraction=.046)
        axes[row, 0].set_ylabel("Left y (m)")
    fig.suptitle("Thunder V4 / unseen terrain tiles / single-frame reconstruction", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "reconstruction.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/mapping_training/dataset.pt")
    parser.add_argument("--output", default="artifacts/mapping_training/run")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=922)
    parser.add_argument("--model", choices=("paper", "lidar-context"), default="paper")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    if not dataset.get("complete"):
        raise ValueError("Acquisition did not complete all three splits")
    splits = [{tuple(tile) for tile in dataset["tile_splits"][key]} for key in ("train", "validation", "test")]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        raise ValueError("Terrain tiles leak across dataset splits")
    grid = GridSpec(**dataset["config"]["grid"])
    train, val, test = [{k: v.to(args.device) for k, v in dataset[s].items()}
                        for s in ("train", "validation", "test")]
    cfg = MappingConfig(map_h=grid.height, map_w=grid.width)
    model_class = MappingNet if args.model == "paper" else LidarContextMappingNet
    model_kwargs = {"missing_height": grid.missing_height} if args.model == "lidar-context" else {}
    model = model_class(cfg, **model_kwargs).to(args.device)
    baseline = None
    if args.baseline_checkpoint:
        old = MappingNet(cfg).to(args.device)
        checkpoint = torch.load(args.baseline_checkpoint, map_location=args.device, weights_only=True)
        for key in ("grid", "sensor_translation_m", "sensor_rpy_rad", "max_distance_m", "samples_per_scan", "scan_hz"):
            if checkpoint["config"][key] != dataset["config"][key]:
                raise ValueError(f"Baseline and dataset sensor configurations differ: {key}")
        old.load_state_dict(checkpoint["model"])
        mu, lv = predict(old, test["raw"], test["observed"])
        baseline = metrics(mu, test["truth"], test["observed"], grid, lv)
        del old
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=.0001)
    result = {"status": "training", "steps": args.steps, "seed": args.seed,
              "model": args.model,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "train_frames": len(train["raw"]), "validation_frames": len(val["raw"]), "test_frames": len(test["raw"]),
              "config": dataset["config"], "acquisition": dataset["acquisition"], "noise": dataset["noise"],
              "tile_splits": dataset["tile_splits"], "loss": "beta-NLL beta=.5; 90% TV + 10% uniform sample weights",
              "old_300step_checkpoint_test": baseline, "curve": [],
              "limits": ["No real sensor recording or calibrated noise distribution",
                         "Saved angle sequence lacks timestamps; 20000 directions at nominal 10 Hz",
                         "Static robot poses, default joints; no rolling-scan motion or odometry errors",
                         "Single-frame network evaluated before history fusion",
                         "Held-out terrain tiles, not an independent real-world test"]}
    best = float("inf")
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(train["raw"]), (args.batch_size,), device=args.device)
        mean, lv = forward_map(model, train["raw"][ids], train["observed"][ids])
        loss = model.beta_nll_loss(mean, lv, train["truth"][ids], tv_weights=sample_weights(train["truth"][ids]))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite mapping loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            mu, log_var = predict(model, val["raw"], val["observed"])
            record = metrics(mu, val["truth"], val["observed"], grid, log_var)
            record.update(step=step, training_loss=loss.item(), grad_norm=grad_norm.item())
            result["curve"].append(record)
            print(json.dumps({"step": step, "validation_mae_m": record["mae_m"],
                              "unknown_mae_m": record["unknown_mae_m"], "edge_mae_m": record["edge_mae_m"]}), flush=True)
            if record["mae_m"] < best:
                best = record["mae_m"]
                result["best_step"] = step
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "model_kind": args.model,
                            "config": dataset["config"], "steps": step,
                            "status": "synthetic held-out terrain prototype; not deployment weights"}, output / "mapping_best.pt")
            (output / "result.json").write_text(json.dumps(result, indent=2))
    result["training_seconds"] = time.monotonic() - started
    torch.save({"model": model.state_dict(), "model_kind": args.model, "config": dataset["config"], "steps": args.steps}, output / "mapping_last.pt")
    saved = torch.load(output / "mapping_best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(saved["model"])
    mean, lv = predict(model, test["raw"], test["observed"])
    result["test"] = metrics(mean, test["truth"], test["observed"], grid, lv)
    clean_mean, clean_lv = predict(model, test["clean_raw"], test["clean_observed"])
    result["test_clean_returns"] = metrics(clean_mean, test["truth"], test["clean_observed"], grid, clean_lv)
    nearest = nearest_fill(test["raw"], test["observed"], grid)
    result["nearest_observed_baseline_test"] = metrics(nearest, test["truth"], test["observed"], grid)
    result["test_raw_observed_mae_m"] = (test["raw"] - test["truth"]).abs()[test["observed"]].mean().item()
    result["test_clean_observed_mae_m"] = (test["clean_raw"] - test["truth"]).abs()[test["clean_observed"]].mean().item()
    result["count_columns"] = ["terrain_returns", "self_occluded_rays", "nonself_returns", "noisy_map_returns", "observed_cells"]
    result["test_mean_counts"] = test["counts"].float().mean(0).tolist()
    make_plot(output, test["raw"], test["truth"], test["observed"], mean, lv, grid)
    # Reload the deployment-shaped checkpoint and compare inference on identical inputs.
    restored = model_class(cfg, **model_kwargs).to(args.device)
    restored.load_state_dict(saved["model"])
    torch.testing.assert_close(forward_map(restored, test["raw"][:4], test["observed"][:4])[0],
                               forward_map(model, test["raw"][:4], test["observed"][:4])[0])
    result["status"] = "completed"
    result["checkpoint_reload_verified"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("TRAINING_COMPLETE=" + json.dumps(result["test"]), flush=True)


if __name__ == "__main__":
    main()
