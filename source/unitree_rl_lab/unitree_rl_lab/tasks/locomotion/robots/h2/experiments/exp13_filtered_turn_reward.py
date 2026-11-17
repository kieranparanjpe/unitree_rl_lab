"""Experiment 13: filtered turn reward.

`track_ang_vel_z` scores a smoothed turn-rate error instead of the per-step error: an exponential
moving average of cz - wz with a time constant of 0.5 s, scored as exp(-e^2 / sigma_ang^2). Weight
0.5 and paid and scoring sigma_ang 0.5 as in the baseline; the turn curriculum scores the same
smoothed error.

In Gazebo the baseline's pelvis twists back and forth with each stride: walking straight, its turn
rate has an SD of about 0.15 rad/s at about 1.6-2.8 Hz while its net turn rate is near 0. The
per-step reward charges for that sway. The smoothed error cuts sway at stride frequency to a
quarter or less and still catches slow drift. The reward depends on about the last 0.5 s, more
than the 0.1 s observation history, so the critic also observes the smoothed error. The policy's
observation is unchanged, so the deploy side is too.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_z.func = mdp.track_ang_vel_z_filtered_exp_with_fixed_std_sum
        self.rewards.track_ang_vel_z.params["tau"] = 0.5
        self.observations.critic.filtered_turn_error = ObsTerm(
            func=mdp.filtered_turn_error, params={"reward_term_name": "track_ang_vel_z"}
        )


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
