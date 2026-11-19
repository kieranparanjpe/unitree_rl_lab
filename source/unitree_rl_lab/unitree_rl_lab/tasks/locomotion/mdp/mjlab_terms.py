"""unitree_rl_mjlab's velocity-task terms (mjlab 1.2.0, src/tasks/velocity), ported to Isaac Lab for
the mjlab-mirror config (robots/h2/experiments/exp17_mjlab_mirror.py).

Each term keeps mjlab's definition. Where Isaac Lab forces a difference, the term says so. Names
overlap with terms in rewards.py (`stand_still`, `feet_gait`, ...) whose definitions differ, so
this module is not star-imported into `mdp`; import it as a module.

"Command speed" below is mjlab's |c_xy| + |c_z|, not the 3-component norm used elsewhere here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, RewardTermCfg


def _command_speed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


def _foot_sites(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, site_offsets: list[list[float]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """World position and velocity of a point fixed in each foot body, standing in for mjlab's foot
    sites. `site_offsets` is one [x, y, z] per body in `asset_cfg.body_ids` order, in that body's
    frame. Returns (num_envs, num_feet, 3) each."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    quat = asset.data.body_link_quat_w[:, ids]
    offsets = torch.tensor(site_offsets, device=env.device).expand(quat.shape[0], -1, -1)
    offset_w = math_utils.quat_apply(quat, offsets)
    pos = asset.data.body_link_pos_w[:, ids] + offset_w
    vel = asset.data.body_link_lin_vel_w[:, ids] + torch.cross(asset.data.body_link_ang_vel_w[:, ids], offset_w, dim=-1)
    return pos, vel


# -- rewards


def track_linear_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """exp(-(|c_xy - v_xy|^2 + 2 v_z^2) / std^2), pelvis frame. The vertical-velocity part is
    what the baseline's separate lin_vel_z_l2 penalty does."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + 2 * z_error) / std**2)


def track_angular_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """exp(-((c_z - w_z)^2 + 0.05 (w_x^2 + w_y^2)) / std^2), pelvis frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + 0.05 * xy_error) / std**2)


def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared x/y of gravity in one body's frame (mjlab applies it to H2's torso_link)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
    gravity_b = math_utils.quat_apply_inverse(quat, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(gravity_b[:, :2]), dim=1)


def body_angular_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared x/y angular velocity of one body, world frame (yaw rate not penalised)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids].squeeze(1)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


class angular_momentum_penalty(ManagerTermBase):
    """|L|^2, L the angular momentum of a set of bodies about their combined centre of mass, world
    frame.

    mjlab reads MuJoCo's `subtreeangmom` sensor, which the engine computes for a body and every body
    below it in the tree. PhysX has no such sensor, so it is summed here from each body's mass,
    inertia and velocity: L = sum m_i (p_i - p_c) x (v_i - v_c) + R_i I_i R_i^T w_i. Isaac Lab gives
    I_i about the body's centre of mass in its link frame. `asset_cfg.body_names` lists the subtree:
    mjlab's H2 attaches the sensor to torso_link, so it covers the torso, head and arms only (its G1
    uses pelvis, the whole robot). Masses and inertias are read once; nothing randomises them here.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._mass = None
        self._inertia = None

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        ids = asset_cfg.body_ids
        if self._mass is None:
            self._mass = asset.root_physx_view.get_masses()[:, ids].to(env.device)
            self._inertia = asset.root_physx_view.get_inertias()[:, ids].to(env.device).view(env.num_envs, -1, 3, 3)
        m = self._mass.unsqueeze(-1)
        pos = asset.data.body_com_pos_w[:, ids]
        vel = asset.data.body_com_lin_vel_w[:, ids]
        ang_vel = asset.data.body_link_ang_vel_w[:, ids]
        total_mass = m.sum(dim=1)
        pos_c = (m * pos).sum(dim=1) / total_mass
        vel_c = (m * vel).sum(dim=1) / total_mass
        orbital = (m * torch.cross(pos - pos_c.unsqueeze(1), vel - vel_c.unsqueeze(1), dim=-1)).sum(dim=1)
        rot = math_utils.matrix_from_quat(asset.data.body_link_quat_w[:, ids])
        inertia_w = rot @ self._inertia @ rot.transpose(-1, -2)
        spin = (inertia_w @ ang_vel.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        return torch.sum(torch.square(orbital + spin), dim=1)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Share of feet whose contact matches a free-running clock's stance/swing, 0..1, while the
    command speed is above `command_threshold`. (rewards.feet_gait sums instead of averaging, 0..2,
    and gates on the 3-component norm.)"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    is_stance = (global_phase + offsets) % 1.0 < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)
    return reward * (_command_speed(env, command_name) > command_threshold)


def feet_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    site_offsets: list[list[float]],
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """sum over feet of |site height - target| x site horizontal speed, while moving.
    (rewards.feet_clearance_penalty is the same with the foot body origin instead of the sole.)"""
    pos, vel = _foot_sites(env, asset_cfg, site_offsets)
    cost = torch.sum(torch.abs(pos[..., 2] - target_height) * torch.norm(vel[..., :2], dim=-1), dim=1)
    return cost * (_command_speed(env, command_name) > command_threshold)


def feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    site_offsets: list[list[float]],
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """sum over feet in contact of squared site horizontal speed, while moving. (Isaac Lab's
    feet_slide uses the unsquared speed of the body origin.)"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0).float()
    _, vel = _foot_sites(env, asset_cfg, site_offsets)
    cost = torch.sum(torch.sum(torch.square(vel[..., :2]), dim=-1) * in_contact, dim=1)
    return cost * (_command_speed(env, command_name) > command_threshold)


def soft_landing(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, command_threshold: float
) -> torch.Tensor:
    """sum over feet that touched down this step of their contact force, while moving."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    cost = torch.sum(force * first_contact.float(), dim=1)
    return cost * (_command_speed(env, command_name) > command_threshold)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """sum of squared joint offsets from the default pose, while the command speed is at most
    `command_threshold`. (rewards.stand_still sums absolute offsets.)"""
    asset: Articulation = env.scene[asset_cfg.name]
    offset = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(offset), dim=1) * (_command_speed(env, command_name) <= command_threshold)


def self_collision_cost(env: ManagerBasedRLEnv, sensor_names: list[str], force_threshold: float) -> torch.Tensor:
    """Number of the last physics substeps (the sensors' history length) in which any listed sensor
    saw a filtered contact above `force_threshold`.

    mjlab watches every robot-robot contact. Isaac Lab reports filtered contacts for one sensor body
    per sensor, so this watches only the feet (one sensor each, filtered against the bodies a foot
    can hit); other self-contact is left to undesired_contacts-style terms."""
    hits = None
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        force = torch.norm(sensor.data.force_matrix_w_history, dim=-1)  # (envs, history, bodies, filters)
        hit = (force > force_threshold).flatten(start_dim=2).any(dim=2)
        hits = hit if hits is None else hits | hit
    return hits.sum(dim=1).float()


# -- observations (critic)


def foot_height(env: ManagerBasedRLEnv, site_offsets: list[list[float]], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _foot_sites(env, asset_cfg, site_offsets)[0][..., 2]


def foot_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time[:, sensor_cfg.body_ids]


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0).float()


def foot_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """sign(f) log(1 + |f|) of each foot's 3D contact force, flattened."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids].flatten(start_dim=1)
    return torch.sign(force) * torch.log1p(torch.abs(force))


# -- events


class randomize_feet_friction_shared(ManagerTermBase):
    """mjlab's foot_friction: one coefficient per robot from `friction_range`, used by every
    collision shape of the foot bodies (both feet), static = dynamic, no restitution. Every other
    shape of the robot gets `other_friction`: mjlab gives non-foot geoms condim 1, which is
    frictionless contact. Body-to-shape mapping as in Isaac Lab's randomize_rigid_body_material."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        view = self.asset.root_physx_view
        counts = [self.asset._physics_sim_view.create_rigid_body_view(p).max_shapes for p in view.link_paths[0]]
        if sum(counts) != view.max_shapes:
            raise ValueError(f"randomize_feet_friction_shared: {sum(counts)} shapes parsed, {view.max_shapes} expected")
        self.foot_shapes = torch.zeros(view.max_shapes, dtype=torch.bool)
        for body_id in asset_cfg.body_ids:
            start = sum(counts[:body_id])
            self.foot_shapes[start : start + counts[body_id]] = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_range: tuple[float, float],
        other_friction: float,
    ):
        env_ids = torch.arange(env.scene.num_envs) if env_ids is None else env_ids.cpu()
        materials = self.asset.root_physx_view.get_material_properties()
        sampled = materials[env_ids]
        sampled[:, :, :2] = other_friction
        sampled[:, :, 2] = 0.0
        mu = torch.empty(len(env_ids)).uniform_(*friction_range).unsqueeze(1)
        sampled[:, self.foot_shapes, 0] = mu
        sampled[:, self.foot_shapes, 1] = mu
        materials[env_ids] = sampled
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


# -- commands


class SnapOnDrawVelocityCommand(UniformVelocityCommand):
    """Isaac Lab's UniformVelocityCommand plus mjlab's zero-snap: a drawn command whose size
    sqrt(cx^2 + cy^2 + cz^2) is at most 0.1 is set to 0. Only at the draw, as in mjlab: heading mode
    then overwrites cz every step, so a snapped heading-mode robot is told to turn toward its
    target. (UniformLevelVelocityCommand snaps every step, after heading mode.)"""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        command = self.vel_command_b[env_ids]
        self.vel_command_b[env_ids] = command * (torch.norm(command, dim=1) > 0.1).unsqueeze(1)


@configclass
class SnapOnDrawVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SnapOnDrawVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """The ranges after the last curriculum stage. Not used in training: export_deploy_cfg writes it
    to deploy.yaml as the /cmd_vel clamp."""


# -- curriculum


def commands_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, velocity_stages: list[dict]
) -> torch.Tensor:
    """Sets the command ranges of the last stage whose `step` (environment steps) has passed."""
    cfg = env.command_manager.get_term(command_name).cfg
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                if stage.get(axis) is not None:
                    setattr(cfg.ranges, axis, tuple(stage[axis]))
    return torch.tensor(cfg.ranges.lin_vel_x[1])
