"""Experiment 14: baseline with sigma_ang back to 0.5, paid (`std`) and scoring (`fixed_std`), and
the turn curriculum's bar lowered from 0.8 to 0.4.

Folded into the baseline on 2026-11-11, so this task is now the same as the baseline.

At sigma_ang 1.0 (the baseline since exp12) a robot standing still under a 0.5 rad/s turn command
still earns 78% of the turn reward, so starting a turn from standing barely pays: in Isaac, exp6
and exp12 both stall when asked to turn in place from standing (exp12 manages only the left spin at
1.0 rad/s). At sigma_ang 0.5 standing earns 37%. The old baseline's problem at 0.5 was the
curriculum: its score never reached 0.8, so the turn range stayed at +-0.1. A score of 0.8 at
sigma_ang 1.0 and 0.41 at 0.5 both mean an error of 0.47 rad/s, and for any spread of errors a
score above 0.8 at 1.0 implies one above 0.41 at 0.5, so a bar of 0.4 widens the turn range at
least as readily as exp12 did.
"""

from isaaclab.utils import configclass

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_z.params["std"] = 0.5
        self.rewards.track_ang_vel_z.params["fixed_std"] = 0.5
        self.curriculum.ang_vel_cmd_levels.params["threshold"] = 0.4


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
