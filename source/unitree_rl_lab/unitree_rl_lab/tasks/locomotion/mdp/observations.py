from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv, period: float, command_name: str | None = None, command_threshold: float = 0.1
) -> torch.Tensor:
    """Sin/cos of a free-running gait clock, optionally zeroed while the command is near zero.

    Without `command_name` the clock runs unconditionally, which leaves the policy with a
    periodic input at zero command and no reward term opposing a periodic output - it marches in
    place. Passing `command_name` gives the standing regime a constant observation, matching
    unitree_rl_mjlab's `phase` term.

    The clock keeps advancing while gated; only the output is zeroed. Freezing it instead would
    make the phase resume from a stale offset when the command returns.

    DEPLOY CONTRACT: the C++ `gait_phase` in deploy/include/isaaclab/envs/mdp/observations/
    observations.h must gate identically - same threshold, same 3-component norm. A checkpoint
    trained with the gate and run against an ungated term marches at zero command anyway.
    """
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        phase = torch.where((cmd_norm > command_threshold).unsqueeze(1), phase, torch.zeros_like(phase))
    return phase
