# Copyright (c) 2024-2026 Inovxio
# SPDX-License-Identifier: Apache-2.0

"""AME-2 locomotion task for ANYmal-D.

Reference: Zhang et al., "AME-2: Agile and Generalized Legged Locomotion via
Attention-Based Neural Map Encoding", arXiv:2601.08485.

Gym environments registered:
    RobotLab-Isaac-AME2-Rough-ANYmal-D-v0
"""

try:
    import gymnasium as gym
    from . import agents
    from .ame2_env_cfg import AME2AnymalEnvCfg

    ##
    # Register Gym environments (only when Isaac Lab / gymnasium is available).
    ##
    gym.register(
        id="RobotLab-Isaac-AME2-Rough-ANYmal-D-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:AME2AnymalEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_cfg:AME2TeacherPPORunnerCfg"
            ),
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_cfg:AME2StudentDistillationRunnerCfg"
            ),
        },
    )

    __all__ = ["AME2AnymalEnvCfg"]

except (ImportError, ModuleNotFoundError):
    # Running without Isaac Sim / gymnasium (e.g., pure-PyTorch testing).
    # The ame2.networks sub-package is still fully importable.
    __all__ = []
