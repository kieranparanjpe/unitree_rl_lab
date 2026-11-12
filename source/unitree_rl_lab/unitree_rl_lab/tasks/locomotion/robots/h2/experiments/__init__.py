"""H2 training experiments. See EXPERIMENTS.md at the repo root for the plan.

Each experiment is one file and one task. The log folder is the task name, lowercased with
underscores (e.g. logs/rsl_rl/unitree_h2_exp1_gaitreward/).
"""

import gymnasium as gym

PPO_CFG = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"

# task name -> module in this folder
EXPERIMENTS = {
    "Unitree-H2-Baseline": "baseline",
    "Unitree-H2-Exp1-GaitReward": "exp1_gait_reward",
    "Unitree-H2-Exp2-GaitRewardPhase": "exp2_gait_reward_phase",
    "Unitree-H2-Exp4-Heading0": "exp4_heading_0",
    "Unitree-H2-Exp4-Heading100": "exp4_heading_100",
    "Unitree-H2-Exp5-TrackingStdCurriculum": "exp5_tracking_std_curriculum",
    "Unitree-H2-Exp6-ZeroAxes": "exp6_zero_axes",
    "Unitree-H2-Exp7-DomainRandomization": "exp7_domain_randomization",
    "Unitree-H2-Exp10-Pose": "exp10_pose",
    "Unitree-H2-Exp11-Alive2": "exp11_alive_2",
    "Unitree-H2-Exp12-AngStd1": "exp12_ang_std_1",
    "Unitree-H2-Exp14-AngStd05Bar04": "exp14_ang_std_05_bar_04",
}

for task_name, module_name in EXPERIMENTS.items():
    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{module_name}:EnvCfg",
            "play_env_cfg_entry_point": f"{__name__}.{module_name}:PlayEnvCfg",
            "rsl_rl_cfg_entry_point": PPO_CFG,
        },
    )
