"""Experiment 1: baseline + gait reward, without the gait phase observation (the G1 default).

feet_gait still grades the feet against its own clock (time since reset); the policy just can't
see that clock.
"""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.gait = baseline.gait_reward()


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
