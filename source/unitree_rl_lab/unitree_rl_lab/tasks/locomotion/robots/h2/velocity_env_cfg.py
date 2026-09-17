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

# Terrain/scene boilerplate below is copied as-is from
# tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py (this repo) - generic, not H2-specific.
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
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
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # prim_path body = "pelvis", not "torso_link" like G1: unitree_rl_mjlab's H2 config
    # (unitree_rl_mjlab/src/tasks/velocity/config/h2/env_cfgs.py) explicitly re-points its raycast
    # terrain-height sensor's reference frame to "pelvis" for H2, so this mirrors that choice.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

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
            # body_names="torso_link": matches unitree_rl_mjlab/src/tasks/velocity/config/h2/env_cfgs.py
            # (cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)).
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    # Six-axis push, matching unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py's shared
    # "push_robot" term, which mjlab's H2 trains with (its config only pops the term under
    # `if play:`, i.e. for evaluation). G1 - and so H2 - pushed on x/y only, which
    # h2-isaac-training-notes.md identifies as one of the few places Isaac's randomization is
    # genuinely thinner than mjlab's. Interval widened to (5.0, 6.0) to match as well, so the
    # pushes aren't phase-locked to a fixed 5 s period.
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
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        # (3, 8) s and 0.05, from unitree_rl_mjlab. Resampling every 10 s gave roughly two
        # commands per 20 s episode, so transitions between commands were barely trained.
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        # heading_command replaces the sampled ang_vel_z, every step, with
        # clip(stiffness * wrap_to_pi(heading_target - heading_w), ang_vel_z range) - a P
        # controller on the robot's own heading. Yaw stops being an independent random variable
        # and becomes feedback, so any yaw a strafe induces immediately shows up as a corrective
        # command the policy has to track. Without it yaw is pure feedforward and nothing ever
        # tells the policy it drifted, which is why strafing curves.
        #
        # It also gives a better yaw profile than uniform sampling: a fresh target every 3-8 s
        # means a large error, a big turn, then decay to zero as the turn completes, instead of
        # a constant random rate held for the whole window.
        #
        # It does NOT buy heading holding at deploy - /cmd_vel supplies ang_vel_z directly and
        # nothing observes heading - so the gain is straighter open-loop travel, not correction.
        heading_command=True,
        heading_control_stiffness=0.5,  # unitree_rl_mjlab's value; the field defaulted to 1.0
        # Snap near-zero draws to exactly zero, matching mjlab. Equal to the command_threshold
        # gait_phase, feet_gait, feet_clearance, feet_slide and pose all gate on, so a sampled
        # command is either a clean stand or unambiguously a walk, never a creep just over the
        # line. Opt-in per robot - the shared cfg defaults it off.
        zero_command_threshold=0.1,
        debug_vis=True,
        # ang_vel_z is set to its full range here rather than in limit_ranges, because NOTHING
        # advances it: the only registered command curriculum is lin_vel_cmd_levels, which
        # touches lin_vel_x and lin_vel_y only (locomotion/mdp/curriculums.py:26-35). There is an
        # ang_vel_cmd_levels beside it that would, but no robot registers it. Leaving yaw to
        # limit_ranges meant training never sampled beyond +-0.1 while deploy.yaml advertised
        # +-0.5 - export_deploy_cfg exports limit_ranges - so policy_node fed the policy yaw
        # commands 5x outside anything it had seen. +-1.0 is unitree_rl_mjlab's step-0 stage.
        # heading must be set on BOTH: the command term rejects heading_command=True with
        # ranges.heading=None, and RobotPlayEnvCfg assigns limit_ranges onto ranges wholesale.
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_names=[".*"] includes head_pitch_joint/head_yaw_joint (see the "head" actuator note in
    # assets/robots/unitree.py's UNITREE_H2_CFG) - unlike unitree_rl_mjlab's H2 task, whose model has no
    # head joints to begin with. Drop them here if you'd rather match that and not actuate the head.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # From h1/velocity_env_cfg.py; G1 and therefore H2 had no phase signal at all, which
        # h2-isaac-training-notes.md identifies as the main reason mjlab's H2 gaits look better.
        # period MUST match the `gait` reward's period below (0.8 for H2, where H1 uses 0.6) -
        # the reward rewards contact at a phase the policy can only see through this term.
        # command_name gates the clock off below 0.1, as unitree_rl_mjlab's `phase` term does.
        # Ungated it hands the policy a periodic input at zero command with no reward term
        # opposing a periodic output, and the robot marches in place instead of standing.
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.8, "command_name": "base_velocity", "command_threshold": 0.1},
        )

        def __post_init__(self):
            # Kept at 5, unlike H1 which comments this out and runs single-frame. Dropping it
            # would be a separate, much larger change to the observation contract.
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.8, "command_name": "base_velocity", "command_threshold": 0.1},
        )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        # weight and std from unitree_rl_mjlab's track_angular_velocity: double the weight and a
        # looser std than G1's, so yaw tracking can outbid the foot terms that oppose turning.
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )

    # Survival economy. Every reward here is scaled by step_dt (0.02), so these are per-second
    # rates: alive pays +0.1/step and a full 20 s episode is worth +100, while a termination
    # costs -4 once.
    #
    # This pairing is NOT optional, and getting it wrong killed run yhyw3t6n. That run had
    # alive=0.15 with no termination penalty, and collapsed to a 3-step episode with 100%
    # base_contact termination inside ~1200 iterations - mean_reward went UP (-1.32 -> +0.026)
    # as mean_episode_length went DOWN (9.57 -> 3), i.e. the policy learned that ending the
    # episode immediately beat playing it. The cause was removing feet_clearance's +20 exp
    # bonus: that term peaked when the feet were still, so it was silently paying ~+20/step
    # (~+400/episode) for merely existing, and it was the only thing making survival
    # worthwhile. Measured cost of exploring in that run was -0.138/step, against alive's
    # +0.003/step.
    #
    # alive=1.0. It was briefly 5.0, sized to offset a -0.138/step exploration cost measured in
    # yhyw3t6n - but that was the COLLAPSED run, where the robot was flailing, so the figure was
    # roughly an order of magnitude too high for a healthy one. Run 3c6nxbca trained cleanly with
    # 5.0 (full 1000-step episodes, 99.7% time_out) and still could not walk: alive came to +100
    # of a +151 positive budget - 66% - against +17 for track_lin_vel_xy, so any change raising
    # fall probability cost 100 x delta_p. The policy converged on the safest gait that still
    # collected `gait` (+18.5, paid for stepping in rhythm whether or not the robot translates).
    # Probing that checkpoint against 09-16 on identical observations showed it holding both hips
    # extended by +0.10/+0.18 rad under a forward command where 09-16 sat at ~0, with smaller
    # ankle push-off and larger knee swing: full-amplitude stepping, weight back, no propulsion.
    #
    # At 1.0 the budget is roughly alive +20, gait +18.5, tracking +32, penalties -38, which puts
    # velocity tracking at ~46% of the positives instead of 21%. is_terminated below is what
    # actually makes termination unattractive - unitree_rl_mjlab carries no alive term at all and
    # relies on it alone - so alive only needs to be a modest dense positive, not a subsidy.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # -200 is unitree_rl_mjlab's value. dt-scaled that is -4 per termination, against ~+150 for
    # a full successful episode - the same ratio mjlab runs at.
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # weight -0.5, from h1/velocity_env_cfg.py, not G1's -0.05. H2 inherited the G1 number along
    # with the rest of this file; H1 is the more developed humanoid config of the two.
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # unitree_rl_mjlab's `pose` term, replacing the three joint_deviation_l1 penalties that used
    # to sit here (arms -0.1, waists -1, legs -1.0). Those apply one weight per joint group at
    # every speed, so they penalise the knee for doing exactly what walking requires; this gives
    # each joint its own TOLERANCE, widened by commanded speed. std tables copied verbatim from
    # unitree_rl_mjlab/src/tasks/velocity/config/h2/env_cfgs.py, with head_* added - mjlab's H2
    # model has no head joints, rl_lab's actuates both, and every joint must be covered or
    # _resolve_joint_stds raises.
    #
    # Note waist_roll/waist_pitch stay at 0.1 in every regime: that is the term meant to hold the
    # torso upright while the legs swing freely (hip_pitch/knee at 0.5).
    pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "walking_threshold": 0.1,
            "running_threshold": 1.5,
            "std_standing": {".*": 0.05},
            "std_walking": {
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
            },
            "std_running": {
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
            },
        },
    )

    # -- robot
    # Back to G1's -5.0. h1/velocity_env_cfg.py uses -1.0, and the tier-1 session copied that,
    # but H1 only gets away with it because it leans on a torso-contact termination - which this
    # config no longer has. At -1.0 run 3c6nxbca held a persistent ~5 degree root tilt
    # (Episode_Reward/flat_orientation_l2 = -0.0079 => mean(g_x^2+g_y^2) = 0.0079) for 0.14% of
    # its return: the lean was effectively free. At -5.0 the same tilt costs 0.79, still under 1%
    # of return, so it adds gradient pressure without crowding out the tracking terms.
    #
    # Complements rather than duplicates `pose`: this penalises ROOT pitch/roll, `pose` penalises
    # JOINT deviation. A lean can be built out of either.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # target_height: NOT sourced from unitree_rl_mjlab - its H2 task uses a "pose" tracking reward instead
    # of an explicit base-height term, so there's no equivalent number to port. Estimated here from H2's
    # init pos z (1.03) using the same target/spawn ratio as G1 (0.78/0.8) - needs tuning once H2 is
    # actually standing correctly in sim.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 1.0})

    # -- feet
    # body_names = ".*ankle_pitch.*", NOT ".*ankle_roll.*" like G1: in H2's kinematic chain (both
    # github.com/unitreerobotics/unitree_ros H2.urdf and unitree_rl_mjlab's h2.xml), ankle_pitch is
    # downstream of ankle_roll, i.e. the actual foot-bearing link. Confirmed by
    # unitree_rl_mjlab/src/tasks/velocity/config/h2/env_cfgs.py, which uses
    # "left_ankle_pitch_link"/"right_ankle_pitch_link" for ground-contact and foot body_names throughout.
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_when_moving,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch.*"),
            "command_name": "base_velocity",
        },
    )
    # unitree_rl_mjlab's shape, not H1's: a linear penalty on horizontal foot motion away from
    # target_height, gated on the command. H1's exp() form is a bonus that peaks when the feet
    # are still, so at H1's weight of 20 it paid ~20/step for standing and cost more to turn in
    # place than track_ang_vel_z could pay back. target_height stays 0.1 - H2's ankle_pitch link
    # sits at world z ~= 0.045 standing (H2.urdf), so 0.1 is a ~5.5 cm swing.
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_penalty,
        weight=-1.0,
        params={
            "target_height": 0.1,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch.*"),
        },
    )
    # From h1/velocity_env_cfg.py; G1 and therefore H2 had no impact penalty at all. Keeps the
    # feet from being slammed into the ground to satisfy the gait and clearance terms.
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={
            "threshold": 500,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch.*"),
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
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # minimum_height: copied from G1's value (0.2) - no H2-specific number found in mjlab either. Loose
    # enough to just catch falls; not meant to be a precise threshold.
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # 70 degrees, from unitree_rl_mjlab. Was 0.8 rad (45.8 deg), which ends the episode on a
    # lean a humanoid can still recover from. That is survivable when termination is free, but
    # not once is_terminated charges -4 for it: frequent AND expensive terminations make the
    # policy refuse to move rather than learn to catch itself.
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # DELIBERATELY ABSENT: a torso-contact termination (h1 has one; unitree_rl_mjlab has none).
    # It was added here from h1, and run yhyw3t6n terminated on it 100% of the time at exactly
    # 3 steps (0.06 s) - far too fast to have fallen, since free-fall from 1.03 m takes ~450 ms.
    # The scene's contact_forces sensor spans Robot/.* with an empty filter_prim_paths_expr and
    # the articulation has enabled_self_collisions=True, so illegal_contact on torso_link fires
    # on SELF-contact: an arm (velocity limit ~19 rad/s) reaching its own torso ends the episode
    # almost instantly. Episode_Reward/undesired_contacts pinned at exactly one body in contact,
    # consistent with that.
    #
    # Restricting it to ground contact needs a filtered ContactSensor plus a termination reading
    # force_matrix_w, which is machinery this repo has nowhere else. Matching mjlab's termination
    # set is the cheaper correct answer; bad_orientation and base_height still catch falls.


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
