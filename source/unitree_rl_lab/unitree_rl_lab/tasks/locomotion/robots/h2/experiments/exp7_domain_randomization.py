"""Experiment 7: baseline + domain randomization, all in one run. Motor gains are not touched.

- Delay: new joint targets reach the motors 0-15 ms late (0-3 physics steps), redrawn per robot
  every policy step. Stands in for the deployed loop's latency and its step-to-step wobble.
- Mass: every link's mass x0.9-1.1 (except the torso, which keeps the baseline's payload), plus
  the torso centre of mass moved up to +-5 cm on each axis. Drawn once per robot at startup.
- Start state: every joint starts up to +-0.1 rad from the default pose and with up to +-0.1 rad/s
  of velocity, instead of exactly on the pose and at rest.

None of it changes what the deploy side runs: deploy.yaml exports the nominal gains and default
pose, and the delay only adds a max_delay_steps field to the action entry, which the C++ side
ignores.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

# Same implementation as Isaac Lab's; the copy in this repo is used so this file does not depend
# on which Isaac Lab release added it.
from unitree_rl_lab.tasks.mimic.mdp.events import randomize_rigid_body_com

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # -- delay
        action = self.actions.JointPositionAction
        self.actions.JointPositionAction = mdp.DelayedJointPositionActionCfg(
            asset_name=action.asset_name,
            joint_names=action.joint_names,
            scale=action.scale,
            use_default_offset=action.use_default_offset,
            # 3 x 0.005 s = 15 ms. Not 4: a full-step delay followed by a zero delay would skip a
            # target entirely, which at 4 happens to 1 in 25 targets. At 3 every target is used
            # for at least one physics step.
            max_delay_steps=3,
        )

        # -- mass
        # randomize_rigid_body_mass resets the bodies it is given to their default mass before
        # applying its change, so including the torso would erase the baseline's -1..+3 kg
        # payload (add_base_mass). The torso keeps that instead.
        self.events.link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="(?!torso_link$).*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
        # unitree_rl_mjlab's base_com range, which its H2 config applies to torso_link. Adds to the
        # current CoM, so it must stay a startup event.
        self.events.torso_com = EventTerm(
            func=randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

        # -- start state
        # Offsets rather than the 0.5-1.5 scale Isaac Lab uses for quadrupeds: a scale leaves every
        # joint whose default is 0 untouched (waist, hip roll/yaw, ankle roll, most of the arms,
        # head), and on the legs it changes leg length by up to ~7 cm while the pelvis is still
        # placed at 1.03 m, starting the feet up to ~2 cm inside the ground. +-0.1 rad on every
        # joint keeps them within ~1 cm below and ~3 cm above their default height (H2.urdf thigh
        # 0.41 m, shin 0.50 m). Clamped to the soft joint limits.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)},
        )


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
