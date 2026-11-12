"""Experiment 12: baseline with sigma_ang = 1.0 for track_ang_vel_z, both paid (`std`) and scoring
(`fixed_std`) (baseline at the time: 0.5 for both).

Folded into the baseline on 2026-11-10, then replaced by exp14 on 2026-11-11 (sigma_ang back to
0.5 with a turn bar of 0.4). The turn bar is set back to 0.8 here so this task still reproduces
the exp12 run.

The turn-rate limit only widens when the yaw reward at the scoring sigma_ang averages above 0.8.
At sigma_ang = 0.5, 09-10 (same yaw term) settled at 0.60, and in the first baseline run the
turn-rate limit had not widened by iteration 4500. At 1.0, 09-10's yaw error would score about 0.85.
"""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_z.params["std"] = 1.0
        self.rewards.track_ang_vel_z.params["fixed_std"] = 1.0
        self.curriculum.ang_vel_cmd_levels.params["threshold"] = 0.8


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
