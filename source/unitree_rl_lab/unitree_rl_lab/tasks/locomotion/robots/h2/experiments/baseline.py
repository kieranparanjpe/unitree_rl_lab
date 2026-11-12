"""H2 velocity baseline: the config every experiment in EXPERIMENTS.md is compared against.

Self-contained on purpose: it does not import from ../velocity_env_cfg.py (the 09-17-v2 config),
so editing that file can never silently change the baseline. Experiments subclass `EnvCfg` here
and change one thing.

Differences from ../velocity_env_cfg.py:
- Commands: resample every 1-8 s; sideways limit +-0.5; turn rate starts at +-0.1 and grows to
  +-1.0 through the new turn curriculum; heading mode for 50% of robots instead of 100%.
- Observations: no gait_phase (policy and critic).
- Rewards: no `pose` (variable_posture) and no `gait`; G1's joint_deviation penalties plus a head
  group instead; turn tracking at weight 0.5 with std 0.5 (the same std as linear tracking). Both
  tracking terms are the `_with_fixed_std_sum` versions: same reward, plus the fixed-std episode
  total the speed and turn curricula score on.
- Curriculum: turn curriculum added, with its bar at 0.4 instead of 0.8. Both use the
  `_fixed_std` versions, which make the same decisions as lin/ang_vel_cmd_levels with the same
  bar while each tracking std equals its fixed std (they only differ in exp5).
Everything else is copied unchanged.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_H2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

FOOT_BODIES = ".*ankle_pitch.*"  # H2's foot-bearing link (downstream of ankle_roll)
GAIT_PERIOD_SECONDS = 0.8
MOVING_COMMAND_THRESHOLD = 0.1  # below this command size the robot counts as standing


# Terms that are off in the baseline but switched on by experiments. Functions, so every
# experiment gets its own copy rather than sharing one mutable config object.
def gait_phase_observation() -> ObsTerm:
    return ObsTerm(
        func=mdp.gait_phase,
        params={
            "period": GAIT_PERIOD_SECONDS,
            "command_name": "base_velocity",
            "command_threshold": MOVING_COMMAND_THRESHOLD,
        },
    )


def gait_reward() -> RewTerm:
    return RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": GAIT_PERIOD_SECONDS,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
        },
    )


FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
        max_init_terrain_level=FLAT_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Not read by any observation or reward; kept so the scene matches 09-17-v2.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    # Required: Articulation.reset() does not restore joint state, so this is what puts the joints
    # back in the default pose. Position scale (1, 1) means no noise; the velocity range scales the
    # default joint velocity, which is 0, so it has no effect.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
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
    # `ranges` is where each curriculum starts; `limit_ranges` is where it stops, and is what
    # export_deploy_cfg writes to deploy.yaml as the /cmd_vel clamp.
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 8.0),
        rel_standing_envs=0.05,
        # Heading-mode robots steer to a random target heading every step:
        # ang_vel_z = clip(0.5 * wrap(target - heading), current ang_vel_z range).
        # The rest hold the drawn ang_vel_z, as /cmd_vel does on the deployed robot.
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        zero_command_threshold=MOVING_COMMAND_THRESHOLD,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # order preserved; deploy.yaml and the C++ deploy side follow it
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = None  # off in the baseline; see gait_phase_observation()

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = None  # off in the baseline; see gait_phase_observation()

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # -- tracking: linear and turn std 0.5 (G1/H1); each command axis is worth 0.5. `std` is
    # what is paid (exp5's curriculum shrinks it); `fixed_std` is what the speed curricula score
    # on. At turn std 0.5 the yaw reward never averages above 0.8, so the turn curriculum's bar is
    # 0.4 (CurriculumCfg). exp12's turn std 1.0 also let the turn limit grow, but a robot standing
    # still then earned most of a turn command's reward, so it rarely turned in place (exp14).
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_with_fixed_std_sum,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "fixed_std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_with_fixed_std_sum,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5, "fixed_std": 0.5},
    )

    # -- survival. H2's foot penalties pay nothing for standing, so falling has to be made
    # expensive explicitly (see the comments above `alive` in ../velocity_env_cfg.py).
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- pelvis
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 1.0})

    # -- joints
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- pose: G1's joint groups and weights, plus the head (G1 has no head joints)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"]
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*"])},
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["head_.*"])},
    )

    # -- feet
    gait = None  # off in the baseline; see gait_reward()
    feet_slide = RewTerm(
        func=mdp.feet_slide_when_moving,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "command_name": "base_velocity",
        },
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_penalty,
        weight=-1.0,
        params={
            "target_height": 0.1,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={
            "threshold": 500,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    # No torso-contact ending: self-contact triggers it (see ../velocity_env_cfg.py).


@configclass
class CurriculumCfg:
    # Flat ground everywhere, so this only changes which flat tile a robot starts on.
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # Each widens its ranges by 0.1 per side, once per episode length, when the matching
    # tracking term, scored at its fixed std (0.5), averages above its bar per second: 0.8 linear,
    # 0.4 turn. A turn score of 0.4 at std 0.5 means about the same error as 0.8 at std 1.0 (exp14).
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels_fixed_std)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels_fixed_std, params={"threshold": 0.4})


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

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None


def apply_play_settings(cfg: EnvCfg) -> None:
    """Fewer robots, a smaller ground, and the full command range from the start."""
    cfg.scene.num_envs = 32
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_play_settings(self)
