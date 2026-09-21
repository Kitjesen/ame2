"""Thunder MID-360 -> frozen mapper -> AME-2 student -> asymmetric PPO.

The default two iterations verify integration, not locomotion performance.
Requires the existing robot_lab, fixed OmniPerception and Isaac Lab environment.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="PoseRoughV4")
parser.add_argument("--config", default="configs/thunder_v4_mid360.json")
parser.add_argument("--mapping-checkpoint", required=True)
parser.add_argument("--output", default="artifacts/thunder_mid360")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--iterations", type=int, default=2)
parser.add_argument("--terrain", choices=("plane", "rough"), default="rough")
parser.add_argument("--resume")
parser.add_argument("--seed", type=int, default=22)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import robot_lab.tasks  # Registers the user's existing Thunder tasks.
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from ame2.lidar_mapping import GridSpec
from ame2.networks.ame2_model import MappingConfig, MappingNet
from ame2.thunder_lidar_env import configure_lidar_scene, ThunderLidarEnv


def run():
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps({"status": "running"}))
    sensor_cfg = json.loads(Path(args.config).read_text())
    grid = GridSpec(**sensor_cfg["grid"])
    checkpoint = torch.load(args.mapping_checkpoint, map_location=args.device, weights_only=True)
    for key in ("grid", "sensor_translation_m", "sensor_rpy_rad", "max_distance_m", "samples_per_scan", "scan_hz"):
        if checkpoint["config"][key] != sensor_cfg[key]:
            raise ValueError(f"Mapper checkpoint and sensor configuration disagree: {key}")
    mapper = MappingNet(MappingConfig(map_h=grid.height, map_w=grid.width)).to(args.device)
    mapper.load_state_dict(checkpoint["model"])
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.terrain == "plane":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.curriculum.terrain_levels = None
    else:
        cfg.scene.terrain.terrain_generator.num_rows = 3
        cfg.scene.terrain.terrain_generator.num_cols = 4
        cfg.scene.terrain.max_init_terrain_level = 1
    configure_lidar_scene(cfg, sensor_cfg, output / "robot_usd")
    env = ThunderLidarEnv(gym.make(args.task, cfg=cfg), mapper, sensor_cfg)
    env.get_observations()
    print("INITIAL_POINT_COUNTS=" + json.dumps(env.pipeline.counts.tolist()), flush=True)
    if env.pipeline.counts[:, 3].sum() == 0:
        print("SELF_MESHES=" + json.dumps(env.self_sensor._geometry.mesh_paths), flush=True)
        print("SELF_RANGES=" + str(env.self_sensor.data.distances[0, :20].tolist()), flush=True)
        print("SENSOR_PARENT=" + str(env.sensor.data.pos_w[0].tolist()), flush=True)
        print("ROBOT_ROOT=" + str(env.robot.data.root_pos_w[0].tolist()), flush=True)
        raise ValueError("No terrain returns in any initial actor map; correct sensor geometry before PPO")
    train_cfg = {
        "num_steps_per_env": 24, "save_interval": 1, "logger": "tensorboard",
        "obs_groups": {"policy": ["map", "history", "commands"],
                       "critic": ["map_teacher", "critic_prop", "contact"]},
        "policy": {"class_name": "ame2.networks.thunder_actor_critic:ThunderLidarActorCritic",
                   "init_noise_std": .2, "noise_std_type": "log"},
        "algorithm": {"class_name": "rsl_rl.algorithms:PPO", "num_learning_epochs": 2,
                      "num_mini_batches": 2, "clip_param": .2, "gamma": .99, "lam": .95,
                      "value_loss_coef": 1., "entropy_coef": .004, "learning_rate": .0003,
                      "max_grad_norm": 1., "use_clipped_value_loss": True, "schedule": "fixed",
                      "desired_kl": .01, "rnd_cfg": None, "symmetry_cfg": None},
    }
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(output / "ppo"), device=args.device)
    if args.resume:
        runner.load(args.resume)
    initial = runner.alg.policy.actor.map_encoder.local_cnn[0].weight.detach().clone()
    losses = []
    original_update = runner.alg.update

    def checked_update():
        values = original_update()
        assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
        assert all(torch.isfinite(p.grad).all() for p in runner.alg.policy.parameters() if p.grad is not None)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in values.values())
        losses.append({k: float(v) for k, v in values.items()})
        return values

    runner.alg.update = checked_update
    runner.learn(num_learning_iterations=args.iterations)
    assert not torch.equal(initial, runner.alg.policy.actor.map_encoder.local_cnn[0].weight)
    assert mapper.training is False and all(p.grad is None for p in mapper.parameters())
    current_obs = env.get_observations()
    with torch.no_grad():
        expected = runner.alg.policy.act_inference(current_obs).clone()
        # Actor must remain independent of every privileged critic input.
        poisoned = current_obs.clone()
        for key in ("map_teacher", "critic_prop", "contact"):
            poisoned[key].fill_(float("nan"))
        torch.testing.assert_close(runner.alg.policy.act_inference(poisoned), expected)
    saved = output / "ppo" / "integration.pt"
    runner.save(str(saved), infos={"sensor_config": sensor_cfg, "mapper": str(Path(args.mapping_checkpoint).resolve())})
    runner.load(str(saved))
    with torch.no_grad():
        torch.testing.assert_close(runner.alg.policy.act_inference(current_obs), expected)
    # Exercise the real auto-reset path for one environment at the control boundary.
    env.episode_length_buf[0] = env.max_episode_length - 1
    _, _, done, _ = env.step(expected)
    assert done[0] and env.pipeline.frames[0] <= 1
    assert (env.prop_history.values[0, :-1] == 0).all()
    assert env.count_sum[3] > 0, "No terrain returns reached the actor map; inspect sensor/self geometry"
    result = {
        "status": "integration_passed_not_locomotion_validated", "task": args.task, "terrain": args.terrain,
        "num_envs": env.num_envs, "control_dt_s": env.unwrapped.step_dt,
        "scan_hz": sensor_cfg["scan_hz"], "samples_per_scan": sensor_cfg["samples_per_scan"],
        "joint_order": env.joint_names, "action_dim": env.num_actions,
        "observation_shapes": {k: list(v.shape) for k, v in current_obs.items()},
        "scan_updates": env.scan_updates, "environment_resets": env.reset_events,
        "point_count_columns": ["rays", "nonself_returns", "self_occluded_rays", "map_returns", "observed_cells"],
        "mean_counts": (env.count_sum / env.count_samples).tolist(), "min_counts": env.count_min.tolist(),
        "max_counts": env.count_max.tolist(),
        "ppo_losses": losses, "map_encoder_updated": True, "mapper_frozen": True,
        "actor_privileged_independence": True, "checkpoint_reload_equal": True,
        "partial_reset_verified": True,
        "mapping_checkpoint_status": checkpoint.get("status"),
        "limitations": ["Bounded PPO integration, no trained walking skill or teacher distillation",
                        "Uses simulation poses; no within-scan motion distortion or real odometry errors",
                        "Front LiDAR optical axes are not independently calibrated on hardware",
                        "Mapper remains fixture-trained; map generalization/uncertainty not validated"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("THUNDER_MID360_RESULT=" + json.dumps(result), flush=True)
    env.close()


try:
    run()
except Exception as error:
    import traceback
    traceback.print_exc()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps({"status": "failed", "error": str(error)}))
    raise
finally:
    app.close()
