"""Collect static MID-360 reconstruction pairs on disjoint Isaac terrain tiles.

Run after installing the existing robot_lab and fixed OmniPerception environment.
This is supervised sensor acquisition, not a locomotion rollout. No PPO runs.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/thunder_v4_mid360.json")
parser.add_argument("--output", default="artifacts/mapping_training/dataset.pt")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--train-batches", type=int, default=128)
parser.add_argument("--eval-batches", type=int, default=32)
parser.add_argument("--seed", type=int, default=922)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import robot_lab.tasks  # Register the existing Thunder asset/task.
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_apply, matrix_from_quat
from ame2.lidar_mapping import GridSpec, omni_points_metres, project_scan
from ame2.thunder_lidar_env import configure_lidar_scene


@torch.no_grad()
def collect():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    grid = GridSpec(**config["grid"])
    cfg = parse_env_cfg("PoseRoughV4", device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    terrain_cfg = cfg.scene.terrain.terrain_generator
    terrain_cfg.num_rows, terrain_cfg.num_cols = 8, 10
    terrain_cfg.seed = args.seed
    terrain_cfg.use_cache = False
    cfg.scene.terrain.max_init_terrain_level = 1
    configure_lidar_scene(cfg, config, output.parent / "robot_usd")
    # Acquisition drives static poses and explicitly refreshes each scan.
    cfg.scene.mid360.update_period = 0.
    cfg.scene.mid360_self.update_period = 0.
    env = gym.make("PoseRoughV4", cfg=cfg).unwrapped
    env.reset()
    scene, sim = env.scene, env.sim
    robot, sensor, body = scene["robot"], scene["mid360"], scene["mid360_self"]
    teacher = scene["ame2_truth"]
    origins = scene.terrain.terrain_origins
    # Split within each terrain-type column, including multiple difficulty rows.
    splits = {"train": [], "validation": [], "test": []}
    for col in range(terrain_cfg.num_cols):
        rows = torch.randperm(terrain_cfg.num_rows).tolist()
        splits["train"] += [[r, col] for r in rows[:5]]
        splits["validation"] += [[r, col] for r in rows[5:6]]
        splits["test"] += [[r, col] for r in rows[6:]]
    tile_sets = [{tuple(t) for t in tiles} for tiles in splits.values()]
    assert not any(tile_sets[i] & tile_sets[j] for i in range(3) for j in range(i))
    dataset = {"config": config, "tile_splits": splits, "seed": args.seed,
               "acquisition": "static articulated Thunder V4 visuals; terrain first hit; no motion distortion",
               "noise": {"range_sigma_m": .02, "return_dropout": .05,
                         "calibrated_to_hardware": False},
               "pattern_timestamps_available": False}
    xy = grid.xy(args.device)
    center = (xy[..., 0].abs() < .1) & (xy[..., 1].abs() < .1)
    b = args.num_envs
    offset = torch.tensor(config["sensor_translation_m"], device=args.device).expand(b, -1)
    mount_quat = torch.tensor(sensor.cfg.offset.rot, device=args.device).expand(b, -1)
    for split, tiles in splits.items():
        values = {k: [] for k in ("raw", "clean_raw", "observed", "clean_observed", "truth", "tile", "counts")}
        tile_tensor = torch.tensor(tiles, device=args.device)
        batches = args.train_batches if split == "train" else args.eval_batches
        for batch in range(batches):
            tile = tile_tensor[torch.randint(len(tiles), (b,), device=args.device)]
            root = robot.data.default_root_state.clone()
            root[:, :3] = origins[tile[:, 0], tile[:, 1]]
            # This keeps the 2.4 x 2 m target crop inside its own 8 x 8 m tile.
            root[:, :2] += torch.rand(b, 2, device=args.device) * 2.8 - 1.4
            roll, pitch = (torch.rand(2, b, device=args.device) - .5) * .28
            yaw = (torch.rand(b, device=args.device) - .5) * (2 * torch.pi)
            root[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
            root[:, 7:] = 0.
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            sim.forward()
            scene.update(.1)
            # Set clearance from local terrain rather than a fixed world height.
            ground = teacher.data.ray_hits_w[..., 2].reshape(b, grid.height, grid.width)
            assert torch.isfinite(ground).all(), "Ground-truth rays missed the assigned terrain"
            root[:, 2] = ground[:, center].max(-1).values + .40 + torch.rand(b, device=args.device) * .25
            robot.write_root_state_to_sim(root)
            sim.forward()
            scene.update(.1)
            environment, self_hit = sensor.get_observation(), body.get_observation()
            # Verify both raycasters query the same pattern window.
            torch.testing.assert_close(sensor._sensor_directions, body._sensor_directions)
            blocked = (self_hit[..., 3] > .5) & ((environment[..., 3] < .5) |
                (self_hit[..., :3].norm(dim=-1) <= environment[..., :3].norm(dim=-1) + 1e-5))
            data = sensor.data
            tf = torch.eye(4, device=args.device)[None].repeat(b, 1, 1)
            tf[:, :3, :3] = matrix_from_quat(quat_mul(data.quat_w, mount_quat))
            tf[:, :3, 3] = data.pos_w + quat_apply(data.quat_w, offset)
            torch.testing.assert_close(data.pos_w, root[:, :3], atol=1e-4, rtol=0.)
            poses = torch.cat((root[:, :3], yaw[:, None]), -1)
            points, valid = omni_points_metres(environment, config["max_distance_m"])
            clean = project_scan(points, valid, tf, poses, grid, exclude=blocked)
            # Deliberate training augmentation in range space, before max-z binning.
            # 2 cm is a test setting, not a fitted near-field sensor noise model.
            ranges = points.norm(dim=-1)
            noisy_range = ranges + torch.randn_like(ranges) * .02
            noisy_points = points * (noisy_range / ranges.clamp_min(1e-6))[..., None]
            noisy_valid = valid & (noisy_range >= .2) & (noisy_range <= config["max_distance_m"])
            noisy_valid &= torch.rand_like(ranges) >= .05
            scan = project_scan(noisy_points, noisy_valid, tf, poses, grid, exclude=blocked)
            truth = teacher.data.ray_hits_w[..., 2] - root[:, 2:3]
            truth = truth.reshape(b, 1, grid.height, grid.width)
            assert torch.isfinite(truth).all() and torch.isfinite(scan.height).all()
            counts = torch.stack((valid.sum(-1), blocked.sum(-1), (valid & ~blocked).sum(-1),
                                  scan.count.sum((1, 2, 3)), scan.observed.sum((1, 2, 3))), -1)
            for key, value in (("raw", scan.height), ("observed", scan.observed),
                               ("clean_raw", clean.height), ("clean_observed", clean.observed),
                               ("truth", truth), ("tile", tile), ("counts", counts)):
                values[key].append(value.cpu())
            if (batch + 1) % 16 == 0 or batch + 1 == batches:
                print(json.dumps({"split": split, "batch": batch + 1, "of": batches,
                                  "observed_cells": counts[:, -1].float().mean().item()}), flush=True)
        dataset[split] = {k: torch.cat(v) for k, v in values.items()}
        assert dataset[split]["observed"].any(), "No accepted terrain returns"
        torch.save(dataset, output)
    dataset["complete"] = True
    torch.save(dataset, output)
    print("DATASET_COMPLETE=" + str(output), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        collect()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
