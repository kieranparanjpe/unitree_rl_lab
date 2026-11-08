from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

# The snap below works by overriding a private method. If upstream ever renames it the override
# would simply never be called - a silent revert to un-snapped commands, which is exactly the
# kind of failure that costs a training run. Fail at import instead.
if not hasattr(UniformVelocityCommand, "_resample_command"):
    raise ImportError(
        "UniformVelocityCommand._resample_command is gone; UniformLevelVelocityCommand's "
        "near-zero snap needs rewriting against the new API."
    )


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Adds unitree_rl_mjlab's near-zero snap to the stock uniform velocity command.

    Upstream samples each axis independently, so a draw can land just above the 0.1 that
    `gait_phase`, `feet_gait`, `feet_clearance`, `feet_slide` and `pose` all gate on - asking
    the robot to creep instead of either standing or walking. mjlab zeroes any draw whose norm
    falls at or below that (`unitree_rl_mjlab/.../mdp/velocity_command.py:77`), which makes the
    standing regime a clean partition rather than something `rel_standing_envs` reaches only by
    chance.

    Unlike mjlab, a snapped robot is also put in standing mode until its next draw. Without that,
    the heading controller rewrites a heading-mode robot's yaw on the very next step, so the snap
    only zeroes x and y and the robot is told to turn in place instead of stand.

    An unsnapped heading-mode command can still drift back across the threshold as its turn
    settles - that is true of mjlab too and is not what this fixes.
    """

    cfg: UniformLevelVelocityCommandCfg

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        keep = torch.norm(self.vel_command_b[env_ids], dim=1) > self.cfg.zero_command_threshold
        self.vel_command_b[env_ids] *= keep.unsqueeze(1)
        self.is_standing_env[env_ids] |= ~keep


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Ceiling the command curriculum grows `ranges` toward, and what export_deploy_cfg writes
    into deploy.yaml as the clamp policy_node applies to /cmd_vel."""

    zero_command_threshold: float = 0.0
    """Draws whose 3-norm is at or below this are snapped to exactly zero, and the robot stands
    until its next draw. Set it equal to the command_threshold the gait/foot/pose reward terms
    gate on - 0.1 for H2.

    Defaults to 0.0 (no-op) because g1, go2 and h1 share this cfg class and none of them asked
    for the behaviour change; opt in per robot."""
