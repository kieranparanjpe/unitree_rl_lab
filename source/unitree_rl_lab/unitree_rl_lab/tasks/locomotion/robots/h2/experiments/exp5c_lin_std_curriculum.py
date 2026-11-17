"""Experiment 5c: experiment 5a with only the linear std curriculum. Paid and scoring sigma_ang stay
at the baseline's 0.5.

In experiments 5, 5a and 5b, paid sigma_lin and paid sigma_ang both shrank, and in-place turning
failed in Gazebo (spins 0-0.3 rad/s against the baseline's 0.9). Linear tracking improved by about
30%. Both could be the cause: a tight sigma_ang stops paying for partial turning, and a tight
sigma_lin charges more for the pelvis sway that stepping in place produces under a zero linear
command. This run keeps 5a's linear tracking gain if sigma_ang was the cause, and loses turning
again if sigma_lin was.
"""

from isaaclab.utils import configclass

from . import baseline, exp5a_std_curriculum_all_resets


@configclass
class EnvCfg(exp5a_std_curriculum_all_resets.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.track_ang_vel_z_std = None


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
