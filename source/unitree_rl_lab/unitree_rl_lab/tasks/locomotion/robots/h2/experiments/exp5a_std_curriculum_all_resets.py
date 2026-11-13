"""Experiment 5a: experiment 5, with its std curriculum judged on every robot whose episode ended
since the last check instead of only the robots resetting on the check step.

In experiment 5's run (2026-11-12) the check-step sample broke the std curriculum. Once robots
stopped falling, no robot reset on the check steps, so every curriculum froze from 6.6k to 15.9k
iterations. Then one robot that fell on a check step decided every later check, and the stds
shrank at every check (sigma_ang 0.45 to 0.25 in about 125 iterations). The speed and turn
curricula still judge only the check step, as in the baseline; experiment 5b changes those too.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import baseline, exp5_tracking_std_curriculum


@configclass
class EnvCfg(exp5_tracking_std_curriculum.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.track_lin_vel_xy_std.func = mdp.tracking_std_levels_all_resets
        self.curriculum.track_ang_vel_z_std.func = mdp.tracking_std_levels_all_resets


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
