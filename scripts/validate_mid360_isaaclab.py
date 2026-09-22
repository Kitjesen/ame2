"""Bounded real-pattern raycast -> map -> learned mapper -> AME encoder check.

Use Isaac Lab Python with this checkout and the fixed OmniPerception on PYTHONPATH.
Fixtures use Thunder V4 mounting extrinsics, not an articulated robot asset.
"""
import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/thunder_v4_mid360.json")
parser.add_argument("--output", default="artifacts/mid360_validation")
parser.add_argument("--scans", type=int, default=30)
parser.add_argument("--train-steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat
from isaacsim.core.utils.prims import create_prim
from LidarSensor.example.isaaclab.isaaclab.sensors import LidarSensor, LidarSensorCfg, LivoxPatternCfg
from ame2.lidar_mapping import GridSpec, LidarElevationMap, omni_points_metres, project_scan
from ame2.networks.ame2_model import MappingConfig, MappingNet, PolicyConfig, AME2Policy


def run():
    torch.manual_seed(22)
    torch.set_num_threads(4)
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = args.device
    grid = GridSpec(**cfg["grid"])
    b = 4
    # Flat, ascending stairs, descending stairs, and a step behind an occluder.
    fixtures = [[], [(.75, 1.15, .18), (1.15, 1.55, .36), (1.55, 3., .54)],
                [(-4., .75, 0.), (.75, 1.15, -.18), (1.15, 1.55, -.36)],
                [(1.2, 3., .18), (.7, .8, .8)]]
    floors = [0., 0., -.54, 0.]
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01, device=device))
    for i in range(b):
        root = f"/World/envs/env_{i}"
        create_prim(root, "Xform")
        create_prim(root + "/Terrain", "Xform")
        create_prim(root + "/Sensor", "Xform", translation=(0., i*10., cfg["fixture_base_height_m"]))
        surfaces = [(-4., 4., floors[i])] + fixtures[i]
        for j, (left, right, top) in enumerate(surfaces):
            bottom = -1.
            block = sim_utils.CuboidCfg(size=(right-left, 6., top-bottom),
                collision_props=sim_utils.CollisionPropertiesCfg())
            block.func(root + f"/Terrain/Box{j}", block, translation=((left+right)/2, i*10., (top+bottom)/2))

    rpy = [torch.tensor([a], device=device) for a in cfg["sensor_rpy_rad"]]
    rotation = quat_from_euler_xyz(*rpy)[0]
    sensor = LidarSensor(LidarSensorCfg(
        prim_path="/World/envs/env_.*/Sensor", mesh_prim_paths=[],
        dynamic_env_mesh_prim_paths=["{ENV_REGEX_NS}/Terrain"],
        offset=LidarSensorCfg.OffsetCfg(pos=tuple(cfg["sensor_translation_m"]), rot=tuple(rotation.tolist())),
        max_distance=cfg["max_distance_m"], update_period=1/cfg["scan_hz"],
        pattern_cfg=LivoxPatternCfg(sensor_type="mid360", samples=cfg["samples_per_scan"], downsample=1)))
    sim.reset()
    poses = torch.zeros(b, 4, device=device)
    poses[:, 1] = torch.arange(b, device=device)*10.
    poses[:, 2] = cfg["fixture_base_height_m"]
    tf = torch.eye(4, device=device)[None].repeat(b, 1, 1)
    tf[:, :3, :3] = matrix_from_quat(rotation[None])
    tf[:, :3, 3] = poses[:, :3] + torch.tensor(cfg["sensor_translation_m"], device=device)
    x = grid.xy(device)[..., 0]
    truth = torch.empty(b, 1, grid.height, grid.width, device=device)
    for i in range(b):
        surface = torch.full_like(x, floors[i])
        for left, right, top in fixtures[i]:
            surface = torch.where((x >= left) & (x < right), torch.maximum(surface, torch.tensor(top, device=device)), surface)
        truth[i, 0] = surface - poses[i, 2]

    raw, masks, ray_counts, sensor_ms = [], [], [], []
    history = LidarElevationMap(b, grid).to(device)
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.scans):
        sim.step(render=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        sensor.update(.1, force_recompute=True)
        obs = sensor.get_observation()
        pts, valid = omni_points_metres(obs, cfg["max_distance_m"])
        scan = project_scan(pts, valid, tf, poses, grid)
        torch.cuda.synchronize()
        sensor_ms.append((time.perf_counter()-start)*1000)
        assert obs.shape == (b, cfg["samples_per_scan"], 4)
        assert torch.isfinite(scan.height).all()
        raw.append(scan.height.clone())
        masks.append(scan.observed.clone())
        ray_counts.append(valid.sum(1))
        # Geometry-only baseline: illustrative 2 cm measurement sigma, not calibration.
        history.update(scan.height, torch.full_like(scan.height, math.log(.02**2)), scan.observed,
                       poses, torch.full((b,), step*.1, device=device))
    raw = torch.stack(raw)
    masks = torch.stack(masks)
    counts = torch.stack(ray_counts)
    flat_error = (raw[:, 0] - truth[0]).abs()[masks[:, 0]]
    assert flat_error.numel() and flat_error.max() < .002, "Flat-plane geometry or mounting transform is wrong"
    final_time = torch.full((b,), (args.scans-1)*.1, device=device)
    geometry_map, geometry_seen, _ = history.crop(poses, final_time)

    # Hold out scan windows, not terrains: this is an integration smoke experiment.
    split = max(1, int(args.scans*.7))
    train_x = raw[:split].flatten(0, 1)
    train_y = truth.repeat(split, 1, 1, 1)
    val_x = raw[split:].flatten(0, 1)
    val_y = truth.repeat(args.scans-split, 1, 1, 1)
    val_mask = masks[split:].flatten(0, 1)
    model = MappingNet(MappingConfig(map_h=grid.height, map_w=grid.width)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    def metrics():
        with torch.no_grad():
            mu, lv = model(val_x)
            err = (mu-val_y).abs()
            return {"height_mae_m": err.mean().item(),
                    "observed_mae_m": err[val_mask].mean().item(),
                    "unobserved_mae_m": err[~val_mask].mean().item(),
                    "mean_sigma_m": (.5*lv).exp().mean().item(),
                    "within_2sigma_fraction": (err <= 2*(.5*lv).exp()).float().mean().item()}

    before = metrics()
    training_start = time.perf_counter()
    losses = []
    for step in range(args.train_steps):
        indices = torch.randint(len(train_x), (16,), device=device)
        mu, lv = model(train_x[indices])
        loss = model.beta_nll_loss(mu, lv, train_y[indices])
        assert torch.isfinite(loss), "Non-finite mapping loss"
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(loss.item())
        if (step+1) % 100 == 0:
            print(f"MAPPING_STEP={step+1} LOSS={loss.item():.6f}", flush=True)
    torch.cuda.synchronize()
    after = metrics()
    assert after["height_mae_m"] < before["height_mae_m"], "Bounded mapper training did not improve validation error"
    checkpoint = output / "mapping_smoke.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
                "steps": args.train_steps, "status": "fixture smoke only, not deployment weights"}, checkpoint)
    restored = MappingNet(MappingConfig(map_h=grid.height, map_w=grid.width)).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    with torch.no_grad():
        prediction, logvar = model(raw[-1])
        torch.testing.assert_close(restored(raw[-1])[0], prediction)
        learned_history = LidarElevationMap(b, grid).to(device)
        for step in range(args.scans):
            mu, lv = model(raw[step])
            learned_history.update(mu, lv, masks[step], poses,
                                   torch.full((b,), step*.1, device=device))
        policy_map, measured, age = learned_history.crop(poses, final_time)
    policy_cfg = PolicyConfig(map_h=grid.height, map_w=grid.width)
    student = AME2Policy(policy_cfg, is_student=True).to(device)
    actions, _, _ = student(policy_map, prop_hist=torch.zeros(b, policy_cfg.prop_history, policy_cfg.d_hist, device=device),
                            commands=torch.zeros(b, 3, device=device))
    assert torch.isfinite(actions).all()
    actions.square().mean().backward()
    assert student.map_encoder.local_cnn[0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
    policy_map[:, 2].sum().item()

    result = {"status": "passed", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "num_envs": b, "scans": args.scans, "rays_per_env_scan": cfg["samples_per_scan"],
              "grid_shape_y_x": [grid.height, grid.width], "policy_shape": list(policy_map.shape),
              "channels": ["x_m", "y_m", "z_m", "height_variance_m2"],
              "flat_max_error_m": flat_error.max().item(),
              "valid_rays_mean_per_fixture": counts.float().mean(0).tolist(),
              "single_scan_coverage_mean": masks.float().mean((0, 2, 3, 4)).tolist(),
              "history_measured_coverage": geometry_seen.float().mean((1, 2, 3)).tolist(),
              "sensor_and_projection_ms_median": float(np.median(sensor_ms[3:])),
              "mapping_steps": args.train_steps, "mapping_before": before, "mapping_after": after,
              "mapping_training_seconds": time.perf_counter()-training_start,
              "torch_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "checkpoint_reload_equal": True, "student_forward_backward_finite": True,
              "limitations": ["Mounting-extrinsic fixtures; no articulated Thunder mesh or hardware evidence",
                              "Single pose per scan; no simulated within-scan motion distortion",
                              "Held-out scan windows share terrain fixtures; not terrain generalization",
                              "Variance calibration and a locomotion PPO rollout remain unverified",
                              "Torch memory excludes Warp and Isaac Sim allocations"]}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(output / "maps.npz", raw=raw[-1].cpu().numpy(), observed=masks[-1].cpu().numpy(),
                        truth=truth.cpu().numpy(), predicted=prediction.cpu().numpy(), sigma=(.5*logvar).exp().cpu().numpy(),
                        policy=policy_map.cpu().numpy(), measured=measured.cpu().numpy(),
                        geometry=geometry_map.cpu().numpy(), losses=np.array(losses))
    print("MID360_MAP_RESULT=" + json.dumps(result), flush=True)


try:
    run()
finally:
    app.close()
