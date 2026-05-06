from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    """Tunable weights for each reward term."""
    vel_tracking:  float =  1.5    # exponential reward for matching cmd_vel (lin)
    yaw_tracking:  float =  0.5    # exponential reward for matching cmd yaw rate
    alive:         float =  0.3    # per-step survival bonus
    control_cost:  float = -1e-3   # penalise large torques (energy)
    foot_slip:     float = -0.1    # penalise tangential foot velocity during stance
    body_height:   float = -0.5    # penalise deviation from target height
    lateral_vel:   float = -0.2    # penalise lateral drift beyond command


class RewardFunction:
    """Computes a scalar reward from the current robot state.

    Responsibility (SRP):
        Only compute the reward signal. Does not manage simulation,
        observations, or control.
    """

    TARGET_HEIGHT: float = 0.27   # metres above ground — matches Go1 keyframe

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self._w = weights or RewardWeights()

    def __call__(
        self,
        lin_vel: np.ndarray,      # (3,) base linear velocity in world frame
        ang_vel: np.ndarray,      # (3,) base angular velocity
        torques: np.ndarray,      # (12,) applied joint torques
        foot_vels: np.ndarray,    # (4,)  foot tangential speed (slip proxy)
        body_height: float,
        cmd_vel: np.ndarray,      # (3,) commanded velocity [vx, vy, yaw_rate]
    ) -> float:
        """Compute and return the scalar step reward."""
        w = self._w

        # Exponential tracking rewards (max=1 when perfect, decays with error)
        lin_vel_err_sq = float(np.sum((lin_vel[:2] - cmd_vel[:2]) ** 2))
        r  = w.vel_tracking * float(np.exp(-lin_vel_err_sq / 0.25))

        yaw_err_sq = float((ang_vel[2] - cmd_vel[2]) ** 2)
        r += w.yaw_tracking * float(np.exp(-yaw_err_sq / 0.25))

        r += w.alive
        r += w.control_cost  * float(np.sum(torques ** 2))
        r += w.foot_slip     * float(np.sum(foot_vels ** 2))
        r += w.body_height   * float(abs(body_height - self.TARGET_HEIGHT))

        # Penalise lateral motion beyond the commanded lateral velocity
        lateral_excess = float(abs(lin_vel[1]) - abs(cmd_vel[1]))
        if lateral_excess > 0:
            r += w.lateral_vel * lateral_excess

        return r
