"""Experiment 16a: baseline + a fine tracking term next to each coarse one.

`track_lin_vel_xy_fine` (paid sigma_lin 0.25, weight 0.5) and `track_ang_vel_z_fine` (paid
sigma_ang 0.25, weight 0.25) measure the same errors as `track_lin_vel_xy` and `track_ang_vel_z`
(paid and scoring sigma 0.5, weights 1.0 and 0.5), which stay unchanged. The curricula keep scoring
on the coarse terms at scoring sigma_lin and sigma_ang 0.5.

The coarse term keeps a gradient at large errors for the whole run; the fine term adds a payoff for
precision. The sigma curricula (exp5/5a/5b/5c) shrink the only term instead, which removes the
gradient at large errors. Taken from an H1_2 Isaac Lab setup deployed on hardware
(gaetanotorella/H1_2_Squat_Locomotion_RL), with each fine weight half its coarse weight; that
setup's coarse turn weight of 1.0 is not copied.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_fine = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.25},
        )
        self.rewards.track_ang_vel_z_fine = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.25,
            params={"command_name": "base_velocity", "std": 0.25},
        )


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
