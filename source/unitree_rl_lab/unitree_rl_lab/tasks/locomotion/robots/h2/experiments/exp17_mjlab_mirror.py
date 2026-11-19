"""Experiment 17: unitree_rl_mjlab's H2 flat velocity task (Unitree-H2-Flat, mjlab 1.2.0), rebuilt in
Isaac Lab as closely as Isaac Lab allows. Self-contained, not a baseline subclass: almost every
section differs. Terms that needed porting are in mdp/mjlab_terms.py.

Sources: unitree_rl_mjlab src/tasks/velocity/velocity_env_cfg.py (shared factory),
config/h2/env_cfgs.py and rl_cfg.py, src/assets/robots/unitree_h2/h2_constants.py.

Copied from mjlab: effort limits and the action scales derived from them (0.25 x effort /
stiffness), observations (no history, no scaling, gait phase 0.6 s), the critic's foot
observations, commands (heading mode for every moving robot, zero-snap only at the draw, the
range step at iteration 5000), rewards and weights (no alive bonus), terminations, foot-only
friction randomization with frictionless non-foot contact, torso centre-of-mass offset, PPO with
observation normalization and 10001 iterations.

Not the same:
- Physics: PhysX instead of MuJoCo; mesh feet instead of 7 capsules per foot.
- Robot: mjlab's H2 model has the head welded (29 joints) and ankle offsets 1-2 cm different from
  H2.urdf. The head stays actuated here; `pose` gives it 0.3 rad walking/running tolerance (its
  0.05 standing tolerance comes from mjlab's ".*" entry). Joint velocity limits stay H2.urdf's
  (MuJoCo's position actuators have none).
- Foot sites (clearance, slip, foot height): the sole centre of H2.urdf's foot mesh, (0.040,
  +-0.038, -0.054) in the ankle_pitch_link frame, where mjlab's model has its site.
- Self-collision: only contacts between each foot and the other leg, own knee and pelvis; Isaac
  Lab filters contacts for one sensor body per sensor. mjlab has no undesired-contacts term, so
  none is added.
- Angular momentum: summed from body masses, inertias and velocities instead of MuJoCo's
  subtreeangmom sensor; over torso_link and below (torso, head, arms), as mjlab's H2 does.
- Encoder bias: left out. mjlab draws it, but its joint_pos_rel only reads it with biased=True,
  which the H2 task does not set, so it has no effect there.
"""

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_H2_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from unitree_rl_lab.tasks.locomotion.mdp import mjlab_terms as mj
from unitree_rl_lab.tasks.mimic.mdp.events import randomize_rigid_body_com


def mirror_robot() -> ArticulationCfg:
    """H2 with mjlab's effort limits. Gains and armature already match."""
    robot = copy.deepcopy(UNITREE_H2_CFG)
    robot.actuators["ankle_pitch"].effort_limit_sim = 66.0  # H2.urdf 66.88
    robot.actuators["waist"].effort_limit_sim = 120.0  # H2.urdf 120 yaw, 180 roll/pitch
    robot.actuators["arms"].effort_limit_sim = 54.0  # H2.urdf 130 shoulder pitch, 60 the rest
    robot.actuators["wrist_pitch_yaw"].effort_limit_sim = 25.0  # H2.urdf 10
    return robot.replace(prim_path="{ENV_REGEX_NS}/Robot")


ROBOT_CFG = mirror_robot()
# mjlab's H2_ACTION_SCALE rule; the head (not actuated in mjlab) gets the same rule: 0.25 x 50 / 20.
ACTION_SCALE = {
    expr: 0.25 * actuator.effort_limit_sim / actuator.stiffness
    for actuator in ROBOT_CFG.actuators.values()
    for expr in actuator.joint_names_expr
}

FEET = SceneEntityCfg("robot", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], preserve_order=True)
FEET_CONTACT = SceneEntityCfg(
    "contact_forces", body_names=["left_ankle_pitch_link", "right_ankle_pitch_link"], preserve_order=True
)
FOOT_SITES = [[0.040, 0.038, -0.054], [0.040, -0.038, -0.054]]
GAIT_PERIOD_SECONDS = 0.6
COMMAND_THRESHOLD = 0.1
STEPS_PER_ITERATION = 24


def foot_self_contact_sensor(side: str) -> ContactSensorCfg:
    other = "right" if side == "left" else "left"
    bodies = [f"{other}_{b}_link" for b in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")]
    bodies += [f"{side}_knee_link", "pelvis"]
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_pitch_link",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in bodies],
        history_length=4,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # multiply x 1.0: a contact's friction is the robot shape's own, as with mjlab's foot priority
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    left_foot_self_contact = foot_self_contact_sensor("left")
    right_foot_self_contact = foot_self_contact_sensor("right")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    # startup
    foot_friction = EventTerm(
        func=mj.randomize_feet_friction_shared,
        mode="startup",
        params={"asset_cfg": FEET, "friction_range": (0.3, 1.6), "other_friction": 0.0},
    )
    torso_com = EventTerm(
        func=randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    # Articulation.reset() does not restore joint state; scale (1, 1) puts the joints exactly on the
    # default pose, as mjlab's zero-offset reset does.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (1.0, 1.0)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class CommandsCfg:
    base_velocity = mj.SnapOnDrawVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mj.SnapOnDrawVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
        limit_ranges=mj.SnapOnDrawVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=ACTION_SCALE, use_default_offset=True
    )


def gait_phase_term() -> ObsTerm:
    # Same gate as mjlab's `phase` (zero while the 3-component command norm is under 0.1); the C++
    # deploy side implements this term.
    return ObsTerm(
        func=mdp.gait_phase,
        params={
            "period": GAIT_PERIOD_SECONDS,
            "command_name": "base_velocity",
            "command_threshold": COMMAND_THRESHOLD,
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # mjlab's order; deploy.yaml and the C++ side follow it (velocity_commands before gait_phase)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = gait_phase_term()
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase = gait_phase_term()
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        foot_height = ObsTerm(func=mj.foot_height, params={"site_offsets": FOOT_SITES, "asset_cfg": FEET})
        foot_air_time = ObsTerm(func=mj.foot_air_time, params={"sensor_cfg": FEET_CONTACT})
        foot_contact = ObsTerm(func=mj.foot_contact, params={"sensor_cfg": FEET_CONTACT})
        foot_contact_forces = ObsTerm(func=mj.foot_contact_forces, params={"sensor_cfg": FEET_CONTACT})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# mjlab H2's per-joint pose tolerances, plus the head (welded in mjlab's model).
POSE_STD_WALKING = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_roll.*": 0.1,
    r".*ankle_pitch.*": 0.15,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
    r".*head.*": 0.3,
}
POSE_STD_RUNNING = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_roll.*": 0.1,
    r".*ankle_pitch.*": 0.25,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
    r".*head.*": 0.3,
}


@configclass
class RewardsCfg:
    track_linear_velocity = RewTerm(
        func=mj.track_linear_velocity, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_angular_velocity = RewTerm(
        func=mj.track_angular_velocity, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    body_orientation_l2 = RewTerm(
        func=mj.body_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}
    )
    pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "std_standing": {".*": 0.05},
            "std_walking": POSE_STD_WALKING,
            "std_running": POSE_STD_RUNNING,
            "walking_threshold": COMMAND_THRESHOLD,
            "running_threshold": 1.5,
        },
    )
    body_ang_vel = RewTerm(
        func=mj.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    angular_momentum = RewTerm(
        func=mj.angular_momentum_penalty,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="torso_link|head_.*|.*_shoulder_.*|.*_elbow_.*|.*_wrist_.*|.*_hand_.*"
            )
        },
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    foot_gait = RewTerm(
        func=mj.feet_gait,
        weight=0.5,
        params={
            "period": GAIT_PERIOD_SECONDS,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": COMMAND_THRESHOLD,
            "command_name": "base_velocity",
            "sensor_cfg": FEET_CONTACT,
        },
    )
    foot_clearance = RewTerm(
        func=mj.feet_clearance,
        weight=-1.0,
        params={
            "target_height": 0.10,
            "site_offsets": FOOT_SITES,
            "command_name": "base_velocity",
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": FEET,
        },
    )
    foot_slip = RewTerm(
        func=mj.feet_slip,
        weight=-0.25,
        params={
            "sensor_cfg": FEET_CONTACT,
            "site_offsets": FOOT_SITES,
            "command_name": "base_velocity",
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": FEET,
        },
    )
    soft_landing = RewTerm(
        func=mj.soft_landing,
        weight=-1e-3,
        params={"sensor_cfg": FEET_CONTACT, "command_name": "base_velocity", "command_threshold": COMMAND_THRESHOLD},
    )
    stand_still = RewTerm(
        func=mj.stand_still,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    self_collisions = RewTerm(
        func=mj.self_collision_cost,
        weight=-1.0,
        params={"sensor_names": ["left_foot_self_contact", "right_foot_self_contact"], "force_threshold": 10.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})


@configclass
class CurriculumCfg:
    command_vel = CurrTerm(
        func=mj.commands_vel,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
                {"step": 5000 * STEPS_PER_ITERATION, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
            ],
        },
    )


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # every physics substep, so the self-contact history covers the last 4 substeps as in mjlab
        for sensor in (self.scene.contact_forces, self.scene.left_foot_self_contact, self.scene.right_foot_self_contact):
            sensor.update_period = self.sim.dt


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.curriculum.command_vel = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    """mjlab's H2 runner: same PPO settings as BasePPORunnerCfg, plus observation normalization
    for actor and critic, and 10001 iterations."""

    max_iterations = 10001
    empirical_normalization = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
