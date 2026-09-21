"""Controlled geometry and mapper ablations using the saved MID-360 sequence.

The alternate mounting is a diagnostic counterfactual, not a calibration edit.
An independent analytic box raycast is compared against saved Isaac fixtures.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ame2.lidar_mapping import GridSpec, project_scan
from ame2.networks.ame2_model import MappingConfig, MappingNet


FIXTURES = [[], [(.75, 1.15, .18), (1.15, 1.55, .36), (1.55, 3., .54)],
            [(-4., .75, 0.), (.75, 1.15, -.18), (1.15, 1.55, -.36)],
            [(1.2, 3., .18), (.7, .8, .8)]]
FLOORS = [0., 0., -.54, 0.]


def rotation_rpy(rpy):
    r, p, y = rpy
    rx = np.array([[1., 0., 0.], [0., np.cos(r), -np.sin(r)], [0., np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0., np.sin(p)], [0., 1., 0.], [-np.sin(p), 0., np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0.], [np.sin(y), np.cos(y), 0.], [0., 0., 1.]])
    return (rz @ ry @ rx).astype(np.float32)


def raycast_boxes(directions, origin, boxes, max_distance):
    """First surface hit, including vertical risers, using the slab method."""
    low = np.array([[left, -3., -1.] for left, right, top in boxes], dtype=np.float32)
    high = np.array([[right, 3., top] for left, right, top in boxes], dtype=np.float32)
    parallel = np.abs(directions[:, None]) < 1e-8
    safe_direction = np.where(parallel, 1., directions[:, None])
    a, b = (low - origin) / safe_direction, (high - origin) / safe_direction
    lower = np.where(parallel, -np.inf, np.minimum(a, b)).max(-1)
    upper = np.where(parallel, np.inf, np.maximum(a, b)).min(-1)
    outside = (parallel & ((origin < low) | (origin > high))).any(-1)
    distance = np.where((upper >= lower) & (lower > 0.) & ~outside, lower, np.inf).min(-1)
    valid = (distance >= .1) & (distance <= max_distance)
    points = origin + directions * np.where(valid, distance, 0.)[:, None]
    return points, valid


def fixture_scans(directions, cfg, rpy, grid, windows=40):
    rotation = rotation_rpy(rpy)
    origin = np.array(cfg["sensor_translation_m"], dtype=np.float32)
    origin[2] += cfg["fixture_base_height_m"]
    poses = torch.zeros(4, 4)
    poses[:, 2] = cfg["fixture_base_height_m"]
    identity = torch.eye(4)[None].repeat(4, 1, 1)
    x = grid.xy("cpu")[..., 0]
    truth = torch.empty(4, 1, grid.height, grid.width)
    for i, objects in enumerate(FIXTURES):
        surface = torch.full_like(x, FLOORS[i])
        for left, right, top in objects:
            surface = torch.where((x >= left) & (x < right), torch.maximum(surface, torch.tensor(top)), surface)
        truth[i, 0] = surface - cfg["fixture_base_height_m"]
    raws, masks = [], []
    samples = cfg["samples_per_scan"]
    for frame in range(windows):
        ids = (frame*samples + np.arange(samples)) % len(directions)
        rays_world = directions[ids] @ rotation.T
        points, valid = [], []
        for i, objects in enumerate(FIXTURES):
            p, v = raycast_boxes(rays_world, origin, [(-4., 4., FLOORS[i])] + objects, cfg["max_distance_m"])
            points.append(p)
            valid.append(v)
        projected = project_scan(torch.tensor(np.stack(points)), torch.tensor(np.stack(valid)), identity, poses, grid)
        raws.append(projected.height)
        masks.append(projected.observed)
    return torch.stack(raws), torch.stack(masks), truth


def receptive_field(grid, cfg):
    """Structural support upper bound: average pool covers every max-pool input."""
    net = MappingNet(cfg)
    net.pool = nn.AvgPool2d(cfg.pool_kernel, cfg.pool_stride)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.fill_(.01)
                module.bias.fill_(.01)
    x = torch.ones(1, 1, grid.height, grid.width, requires_grad=True)
    row = grid.height // 2
    col = round((1.2 - grid.x_min) / grid.resolution - .5)
    net(x)[0][0, 0, row, col].backward()
    support = x.grad[0, 0] != 0
    rows, cols = support.nonzero(as_tuple=True)
    return support, {"query_cell_y_x": [row, col], "support_y_x": [int(rows.max()-rows.min()+1), int(cols.max()-cols.min()+1)],
                     "support_min_y_x": [int(rows.min()), int(cols.min())],
                     "support_max_y_x": [int(rows.max()), int(cols.max())]}


def evaluate(model, x, y, mask, edge, roi):
    with torch.no_grad():
        mean, lv = model(x)
        error = (mean-y).abs()
        edge = edge.expand_as(error)
        roi = roi[None, None].expand_as(error)
        return {"mae_m": error.mean().item(), "observed_mae_m": error[mask].mean().item(),
                "unobserved_mae_m": error[~mask].mean().item() if (~mask).any() else None,
                "edge_mae_m": error[edge].mean().item(), "forward_strip_mae_m": error[roi].mean().item(),
                "mean_sigma_m": (.5*lv).exp().mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--config", default="configs/thunder_v4_mid360.json")
    parser.add_argument("--isaac-maps", default="artifacts/mid360_validation/maps.npz")
    parser.add_argument("--output", default="artifacts/mid360_diagnosis")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads(Path(args.config).read_text())
    grid = GridSpec(**cfg["grid"])
    mc = MappingConfig(map_h=grid.height, map_w=grid.width)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    angles = np.load(args.pattern)
    az, el = angles.T
    directions = np.stack((np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)), -1)
    current, masks, truth = fixture_scans(directions, cfg, cfg["sensor_rpy_rad"], grid)
    alternate_rpy = [-math.pi, math.pi/8, 0.]
    alternate, alternate_masks, _ = fixture_scans(directions, cfg, alternate_rpy, grid)
    xy = grid.xy("cpu")
    roi = (xy[..., 0] >= .5) & (xy[..., 1].abs() <= .2)
    ground = np.concatenate((xy.numpy(), np.zeros((grid.height, grid.width, 1), dtype=np.float32)), -1)
    origin = np.array(cfg["sensor_translation_m"])
    origin[2] += cfg["fixture_base_height_m"]
    ground_rays = ground - origin
    ground_rays /= np.linalg.norm(ground_rays, axis=-1, keepdims=True)
    pitch_sweep = []
    for pitch in (-45, -30, -15, 0, 15, 22.5, 30, 45):
        sensor_rays = ground_rays @ rotation_rpy([-math.pi, math.radians(pitch), 0.])
        elevation = np.degrees(np.arcsin(np.clip(sensor_rays[..., 2], -1, 1)))
        visible = (elevation >= -7.) & (elevation <= 52.)
        pitch_sweep.append({"pitch_deg": pitch, "ideal_flat_strip_fov_fraction": float(visible[roi.numpy()].mean())})
    # Cells bordering a ground-truth discontinuity, on either side.
    edge = torch.zeros_like(truth, dtype=torch.bool)
    delta = (truth[..., 1:] - truth[..., :-1]).abs() > .05
    edge[..., 1:] |= delta
    edge[..., :-1] |= delta
    saved = np.load(args.isaac_maps)
    saved_mask = torch.tensor(saved["observed"])
    saved_raw = torch.tensor(saved["raw"])
    intersection = (masks & saved_mask).sum((1, 2, 3, 4))
    union = (masks | saved_mask).sum((1, 2, 3, 4))
    iou = intersection / union
    match = int(iou.argmax())
    both = masks[match] & saved_mask
    raw_match_error = (current[match]-saved_raw).abs()[both]
    # Riser hits lie exactly on grid boundaries; float32 raycasters may assign
    # them to opposite neighbouring cells. Compare interiors separately.
    interior = both & ~edge
    interior_error = (current[match]-saved_raw).abs()[interior]
    print("FIXTURE_MATCH=" + json.dumps({"window": match, "iou": iou[match].item(),
          "raw_mae_m": raw_match_error.mean().item(),
          "interior_mae_m": interior_error.mean().item(),
          "per_scene_iou": ((masks[match] & saved_mask).sum((1, 2, 3)) / (masks[match] | saved_mask).sum((1, 2, 3))).tolist()}), flush=True)
    assert iou[match] > .98 and interior_error.max() < .002, "Analytic fixture interiors differ from Isaac evidence"
    raw_error = (current-truth).abs()
    raw_metrics = []
    for i in range(4):
        observed = masks[:, i]
        boundary = edge[i].expand_as(observed)
        values = raw_error[:, i]
        raw_metrics.append({"coverage": observed.float().mean().item(),
                            "union_coverage": observed.any(0).float().mean().item(),
                            "forward_strip_coverage": observed[:, 0, roi].float().mean().item(),
                            "raw_observed_mae_m": values[observed].mean().item(),
                            "raw_edge_mae_m": values[observed & boundary].mean().item() if (observed & boundary).any() else None,
                            "raw_away_from_edge_mae_m": values[observed & ~boundary].mean().item()})
    support, rf = receptive_field(grid, mc)
    row, col = rf["query_cell_y_x"]
    # Flat vs descending stairs: same local input support, different truth.
    patch_difference = (current[:, 0, 0, support] - current[:, 2, 0, support]).abs().max().item()
    rf.update({"flat_downstairs_support_max_input_difference_m": patch_difference,
               "flat_downstairs_query_truth_difference_m": abs(truth[0, 0, row, col]-truth[2, 0, row, col]).item()})
    record = {"pattern_elevation_min_deg": np.degrees(angles[:, 1]).min().item(),
              "pattern_elevation_max_deg": np.degrees(angles[:, 1]).max().item(),
              "analytic_isaac_match_window": match, "analytic_isaac_mask_iou": iou[match].item(),
              "analytic_isaac_raw_mae_m": raw_match_error.mean().item(),
              "analytic_isaac_interior_max_error_m": interior_error.max().item(),
              "original_mount_metrics": raw_metrics,
              "counterfactual_rpy_rad": alternate_rpy,
              "counterfactual_coverage": alternate_masks.float().mean((0, 2, 3, 4)).tolist(),
              "counterfactual_forward_strip_coverage": alternate_masks[:, :, 0, roi].float().mean((0, 2)).tolist(),
              "counterfactual_union_forward_strip_coverage": alternate_masks.any(0)[:, 0, roi].float().mean(1).tolist(),
              "pitch_sweep": pitch_sweep,
              "receptive_field": rf, "training": {},
              "limits": ["Diagnostic pitch change is not physical calibration", "Stationary fixtures without robot self-occlusion",
                         "Validation windows share training terrains", "Exact geometric returns without real sensor noise"]}
    print("GEOMETRY=" + json.dumps(record), flush=True)
    arrays = {"current_raw": current[-1].numpy(), "current_mask": masks[-1].numpy(),
              "alternate_raw": alternate[-1].numpy(), "alternate_mask": alternate_masks[-1].numpy(),
              "truth": truth.numpy(), "rf_support": support.numpy()}
    cases = [("current_sparse", current, masks), ("counterfactual_sparse", alternate, alternate_masks),
             ("dense_oracle", truth[None].repeat(len(current), 1, 1, 1, 1), torch.ones_like(masks))]
    split = 28
    for name, inputs, observed in cases:
        torch.manual_seed(22)
        model = MappingNet(mc).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        train_x = inputs[:split].flatten(0, 1).to(args.device)
        train_y = truth.repeat(split, 1, 1, 1).to(args.device)
        val_x = inputs[split:].flatten(0, 1).to(args.device)
        val_y = truth.repeat(len(inputs)-split, 1, 1, 1).to(args.device)
        val_mask = observed[split:].flatten(0, 1).to(args.device)
        val_edge = edge.repeat(len(inputs)-split, 1, 1, 1).to(args.device)
        curve = []
        for step in range(args.steps):
            ids = torch.randint(len(train_x), (16,), device=args.device)
            mu, lv = model(train_x[ids])
            loss = model.beta_nll_loss(mu, lv, train_y[ids])
            assert torch.isfinite(loss), (name, step, "nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), (name, step, "nonfinite gradient")
            optimizer.step()
            if step+1 in (300, args.steps):
                metrics = evaluate(model, val_x, val_y, val_mask, val_edge, roi.to(args.device))
                metrics["steps"] = step+1
                curve.append(metrics)
                print(name + "=" + json.dumps(metrics), flush=True)
        record["training"][name] = curve
        with torch.no_grad():
            mu, lv = model(inputs[-1].to(args.device))
            arrays[name + "_prediction"] = mu.cpu().numpy()
            arrays[name + "_sigma"] = (.5*lv).exp().cpu().numpy()
            if name == "current_sparse":
                rf["flat_downstairs_predicted_difference_m"] = abs(mu[0, 0, row, col]-mu[2, 0, row, col]).item()
        (output / "result.json").write_text(json.dumps(record, indent=2))
    np.savez_compressed(output / "maps.npz", **arrays)
    print("DIAGNOSIS_COMPLETE=" + str(output), flush=True)


if __name__ == "__main__":
    main()
