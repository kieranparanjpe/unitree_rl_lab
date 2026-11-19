"""Experiment 18b: experiment 18 with `feet_clearance` at G1's weight, 1.0 instead of H1's 20.

In exp18 (2026-11-19) the +20 term paid about 20 per second for standing still: it charges a foot
only while the foot moves sideways and is away from 0.15 m, so feet that never move are never
charged. Every other term together paid under 2.7 per second at most. The policy never lifted a foot
at any command, in Isaac or Gazebo. Scored on the baseline's walking gait, the term would have
cost 1.1-1.4 per second more than standing still, more than walking earns from tracking and the gait
reward.

Unitree's first commit (33d3ba2, 2025-06-30) had weight 20 for both H1 and G1; the same day,
ddcb2a2 set G1's to 1.0 (and its target from 0.1 to 0.08 m). H1 was never changed. Only the weight
changes here; the target stays H1's 0.15 m.
"""

from isaaclab.utils import configclass

from . import exp18_h1_setup


@configclass
class EnvCfg(exp18_h1_setup.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_clearance.weight = 1.0


@configclass
class PlayEnvCfg(exp18_h1_setup.PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_clearance.weight = 1.0
