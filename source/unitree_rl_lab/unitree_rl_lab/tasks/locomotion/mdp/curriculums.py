from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _widen_lin_vel_ranges(env: ManagerBasedRLEnv) -> None:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    delta_command = torch.tensor([-0.1, 0.1], device=env.device)
    ranges.lin_vel_x = torch.clamp(
        torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
        limit_ranges.lin_vel_x[0],
        limit_ranges.lin_vel_x[1],
    ).tolist()
    ranges.lin_vel_y = torch.clamp(
        torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
        limit_ranges.lin_vel_y[0],
        limit_ranges.lin_vel_y[1],
    ).tolist()


def _widen_ang_vel_ranges(env: ManagerBasedRLEnv) -> None:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    delta_command = torch.tensor([-0.1, 0.1], device=env.device)
    ranges.ang_vel_z = torch.clamp(
        torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
        limit_ranges.ang_vel_z[0],
        limit_ranges.ang_vel_z[1],
    ).tolist()


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            _widen_lin_vel_ranges(env)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            _widen_ang_vel_ranges(env)

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def lin_vel_cmd_levels_fixed_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """`lin_vel_cmd_levels`, scored on what the tracking term would have paid at its `fixed_std`
    rather than what it paid, so `tracking_std_levels` tightening the std can't stall the speed
    ranges. The term must be `track_lin_vel_xy_yaw_frame_exp_with_fixed_std_sum`."""
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    score = torch.mean(reward_term.func.fixed_std_episode_sum[env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        # The total is unweighted, so this is the same test as lin_vel_cmd_levels' 0.8 * weight.
        if score > 0.8:
            _widen_lin_vel_ranges(env)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels_fixed_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """`ang_vel_cmd_levels`, scored at the tracking term's `fixed_std`; see
    `lin_vel_cmd_levels_fixed_std`. The term must be `track_ang_vel_z_exp_with_fixed_std_sum`."""
    ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    score = torch.mean(reward_term.func.fixed_std_episode_sum[env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if score > 0.8:
            _widen_ang_vel_ranges(env)

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def tracking_std_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
) -> torch.Tensor:
    """Shrink a tracking term's `std` by 0.05, down to 0.05, on the speed curricula's schedule,
    whenever the reward it actually paid averages above 0.8 of its weight. Never loosens.

    Pair it with the `_fixed_std` speed curricula: the plain ones score on the paid reward, which
    this makes stricter, so they would stop widening the ranges."""
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            reward_term.params["std"] = max(round(reward_term.params["std"] - 0.05, 2), 0.05)
            env.reward_manager.set_term_cfg(reward_term_name, reward_term)

    return torch.tensor(reward_term.params["std"], device=env.device)
