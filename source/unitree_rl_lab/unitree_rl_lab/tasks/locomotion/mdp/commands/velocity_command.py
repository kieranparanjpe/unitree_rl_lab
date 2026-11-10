from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

# The snap and the axis zeroing below work by overriding private methods. If upstream ever
# renames one the override would simply never be called - a silent revert to the stock
# commands, which is exactly the kind of failure that costs a training run. Fail at import instead.
if not all(hasattr(UniformVelocityCommand, name) for name in ("_update_command", "_resample_command")):
    raise ImportError(
        "UniformVelocityCommand._update_command or _resample_command is gone; "
        "UniformLevelVelocityCommand's near-zero snap and axis zeroing need rewriting against the new API."
    )


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Adds a near-zero snap, and optional per-axis zeroing, to the stock uniform velocity command.

    Upstream samples each axis independently, so a command can land just above the 0.1 that
    `gait_phase`, `feet_gait`, `feet_clearance`, `feet_slide` and `pose` all gate on - asking
    the robot to creep instead of either standing or walking. Any command whose norm is at or
    below that is set to exactly zero, which makes the standing regime a clean partition rather
    than something `rel_standing_envs` reaches only by chance.

    The snap runs every step, after the heading controller has set that step's turn rate, so it
    judges the command the robot is actually given, the same way those gates do. A heading-mode
    robot with little linear command turns toward its target until its turn rate drops to the
    threshold, then stands.

    unitree_rl_mjlab snaps only when it draws a command (`unitree_rl_mjlab/.../mdp/
    velocity_command.py:77`), using a turn rate that heading mode overwrites on the next step,
    so there a heading-mode command can settle below the threshold without being zeroed.

    Because the axes are drawn independently, none is ever exactly zero: turning in place needs
    both linear axes to land near zero, about 2% of draws. With `zero_axis_prob` > 0, each axis
    of a new command is set to exactly zero with that probability, independently, so single-axis
    commands (turn in place, straight walk, strafe) and two-axis ones (walk and turn without
    strafing) come up. Only robots not in heading mode are zeroed. This happens once per draw;
    the snap still runs every step after it, so a command left with one small axis stands.
    """

    cfg: UniformLevelVelocityCommandCfg

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self.cfg.zero_axis_prob > 0.0:
            zero = torch.rand(len(env_ids), 3, device=self.device) < self.cfg.zero_axis_prob
            zero &= ~self.is_heading_env[env_ids].unsqueeze(1)
            self.vel_command_b[env_ids] *= ~zero

    def _update_command(self):
        super()._update_command()
        keep = torch.norm(self.vel_command_b, dim=1) > self.cfg.zero_command_threshold
        self.vel_command_b *= keep.unsqueeze(1)


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Ceiling the command curriculum grows `ranges` toward, and what export_deploy_cfg writes
    into deploy.yaml as the clamp policy_node applies to /cmd_vel."""

    zero_command_threshold: float = 0.0
    """Commands whose 3-norm is at or below this are set to exactly zero, checked every step.
    Set it equal to the command_threshold the gait/foot/pose reward terms gate on - 0.1 for H2.

    Defaults to 0.0 (no-op) because g1, go2 and h1 share this cfg class and none of them asked
    for the behaviour change; opt in per robot."""

    zero_axis_prob: float = 0.0
    """Probability that each axis (cx, cy, cz) of a newly drawn command is set to exactly zero,
    independently per axis. Applies only to robots not in heading mode; heading-mode robots keep
    the command as drawn. Defaults to 0.0 (no-op), for the same reason as above."""
