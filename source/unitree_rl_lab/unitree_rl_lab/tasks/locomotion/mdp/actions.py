from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """`JointPositionAction` whose new joint targets reach the motors a few physics steps late.

    Stands in for the deployed loop (state read, policy, bus, motor driver) not being locked to
    the physics. Every policy step, each robot draws a new delay of 0..max_delay_steps physics
    steps; for that many physics steps the joints keep tracking the previous target, then switch
    to the new one. So each target lands at a random point in its policy step. Targets never
    arrive out of order. With max_delay_steps = decimation a target can miss its own policy step
    entirely and, if the next step's delay is 0, never be used at all; below decimation every
    target is used for at least one physics step.

    The PD itself is untouched (still PhysX's implicit drive), so with a delay of 0 this is exactly
    `JointPositionAction`. Isaac Lab's `DelayedPDActuatorCfg` would also delay, but only by
    replacing the implicit PD with an explicit one (a second change), and it draws its delay only
    at reset.

    Only the previous target is kept, so the delay cannot exceed one policy step (decimation
    physics steps).
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0 <= cfg.max_delay_steps <= env.cfg.decimation:
            raise ValueError(
                f"max_delay_steps must be between 0 and decimation ({env.cfg.decimation}), got {cfg.max_delay_steps}"
            )
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_targets = self._processed_actions.clone()
        self._physics_step = 0

    def process_actions(self, actions: torch.Tensor):
        self._previous_targets = self._processed_actions.clone()
        super().process_actions(actions)
        self._delay_steps = torch.randint_like(self._delay_steps, 0, self.cfg.max_delay_steps + 1)
        self._physics_step = 0

    def apply_actions(self):
        late = (self._physics_step < self._delay_steps).unsqueeze(1)
        targets = torch.where(late, self._previous_targets, self._processed_actions)
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)
        self._physics_step += 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # A reset robot starts in the default pose, so the target it was already holding is the
        # default pose (zero action), not the last target of its previous episode.
        self._processed_actions[env_ids] = self._offset[env_ids]


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction

    max_delay_steps: int = 0
    """Largest delay in physics steps (0.005 s each for H2); each robot draws 0..max_delay_steps,
    uniformly, every policy step. At most the env's decimation."""
