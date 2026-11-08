"""Experiment 2: baseline + gait reward + gait phase observation (the 09-17-v2 gait setup).

Adds gait_phase to the policy (and critic), so the observation grows from 510 to 520 numbers.
deploy.yaml lists it, and the deploy side picks it up from there.
"""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.gait = baseline.gait_reward()
        self.observations.policy.gait_phase = baseline.gait_phase_observation()
        self.observations.critic.gait_phase = baseline.gait_phase_observation()


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
