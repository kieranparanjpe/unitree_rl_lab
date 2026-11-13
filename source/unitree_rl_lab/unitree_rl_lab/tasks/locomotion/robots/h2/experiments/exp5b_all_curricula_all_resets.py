"""Experiment 5b: experiment 5a, with the speed and turn curricula also judged on every robot whose
episode ended since the last check. If experiment 5a's speed or turn ranges stall as experiment
5's did, this shows whether the same scoring fixes them.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import baseline, exp5a_std_curriculum_all_resets


@configclass
class EnvCfg(exp5a_std_curriculum_all_resets.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.lin_vel_cmd_levels.func = mdp.lin_vel_cmd_levels_fixed_std_all_resets
        self.curriculum.ang_vel_cmd_levels.func = mdp.ang_vel_cmd_levels_fixed_std_all_resets


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
