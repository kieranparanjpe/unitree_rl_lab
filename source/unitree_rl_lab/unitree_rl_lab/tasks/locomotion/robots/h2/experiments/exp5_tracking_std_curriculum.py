"""Experiment 5: baseline + a tracking std curriculum.

Each tracking term's paid std shrinks from its baseline value (0.5 linear, 1.0 turn) by 0.05
(down to 0.05) whenever that term averages above 0.8 of its weight, checked once per episode
length. Linear and turn shrink separately. The speed and turn curricula keep scoring at the
baseline's fixed std (0.5 linear, 1.0 turn), so they grow exactly as in the baseline.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.track_lin_vel_xy_std = CurrTerm(
            func=mdp.tracking_std_levels, params={"reward_term_name": "track_lin_vel_xy"}
        )
        self.curriculum.track_ang_vel_z_std = CurrTerm(
            func=mdp.tracking_std_levels, params={"reward_term_name": "track_ang_vel_z"}
        )


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
