from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

# The snap below works by overriding a private method. If upstream ever renames it the override
# would simply never be called - a silent revert to un-snapped commands, which is exactly the
# kind of failure that costs a training run. Fail at import instead.
if not hasattr(UniformVelocityCommand, "_update_command"):
    raise ImportError(
        "UniformVelocityCommand._update_command is gone; UniformLevelVelocityCommand's "
        "near-zero snap needs rewriting against the new API."
    )


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Adds a near-zero snap to the stock uniform velocity command.

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
    """

    cfg: UniformLevelVelocityCommandCfg

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
