"""AME-2 student architecture adapted to Thunder velocity commands and 16 actions."""
from .ame2_model import PolicyConfig
from .rslrl_wrapper import AME2ActorCritic


class ThunderLidarActorCritic(AME2ActorCritic):
    """Preserve Thunder action semantics; do not apply ANYmal joint symmetry."""

    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        if num_actions != 16 or obs["history"].shape[-1] != 54:
            raise ValueError("Thunder requires 12 leg positions + 4 wheel velocities and 54-D history")
        cfg = PolicyConfig(
            map_h=obs["map"].shape[-2], map_w=obs["map"].shape[-1], num_joints=16,
            d_prop_raw=60, d_prop_critic=60, d_commands_critic=3,
        )
        super().__init__(obs, obs_groups, num_actions, ame2_cfg=cfg, is_student=True,
                         critic_kwargs={"d_contact": obs["contact"].shape[-1]}, **kwargs)

    def evaluate(self, obs, **kwargs):
        # The inherited left/right transform is specifically for ANYmal's 12 joints.
        return self.critic(obs["map_teacher"], obs["critic_prop"], obs["contact"])
