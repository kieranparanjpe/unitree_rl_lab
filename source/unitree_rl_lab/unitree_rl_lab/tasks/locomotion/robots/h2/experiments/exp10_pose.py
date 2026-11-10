"""Experiment 10: baseline with the four joint_deviation_* penalties replaced by one `pose`
penalty (variable_posture_penalty, -1.0).

The joint_deviation penalties apply one weight per joint group at every speed, so they penalise
the hips and knees for doing what walking requires. `pose` gives each joint its own tolerance,
picked from one of three tables by commanded speed (|lin| + |ang|): standing below 0.1, walking
up to 1.5, running above. It is the 09-17-v2 `pose` term (variable_posture, +1.0) as a penalty,
so like the terms it replaces it is 0 at the default pose and adds nothing to the value of
staying alive. -1.0 is the largest weight among the terms it replaces. The tables are copied from
../velocity_env_cfg.py rather than imported, so editing that file can't change this experiment.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import baseline


@configclass
class EnvCfg(baseline.EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waist = None
        self.rewards.joint_deviation_hips = None
        self.rewards.joint_deviation_head = None
        self.rewards.pose = RewTerm(
            func=mdp.variable_posture_penalty,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "base_velocity",
                "walking_threshold": baseline.MOVING_COMMAND_THRESHOLD,
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


@configclass
class PlayEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        baseline.apply_play_settings(self)
