"""Experiment 6: baseline + each command axis set to exactly 0 with probability 0.3.

For robots not in heading mode, cx, cy and cz of every newly drawn command are each zeroed
independently with probability 0.3 (`zero_axis_prob`); heading-mode robots are unchanged. In the
baseline no axis is ever exactly 0, so turning in place only comes up when both linear axes land
near 0 (about 2% of draws). Exp12's Gazebo eval turned in place poorly (right spin at 12% of the
commanded rate, half_square's 0.5 rad/s turns at 33%) while its walking turns were fine.
"""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.zero_axis_prob = 0.3


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
