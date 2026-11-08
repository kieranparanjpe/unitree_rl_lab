"""Experiment 4b: heading mode for every robot, as in 09-17-v2 (baseline: 50%)."""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_heading_envs = 1.0


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
