"""Experiment 11: baseline with alive 2.0 (baseline: 1.0).

At iteration 3000 of the first baseline run, the penalties outweighed alive and tracking while
the robot was up, so the -200 fall penalty was the main reason to stay up, and it learned to stay
up about 1500 iterations later than 09-17-v2 (which had gait and pose bonuses). This adds 1 per
second while up and changes nothing else.
"""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.alive.weight = 2.0


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
