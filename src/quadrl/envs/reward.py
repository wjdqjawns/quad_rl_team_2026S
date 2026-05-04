from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    """Tunable weights for each reward term."""
    forward_vel:  float =  1.0    # reward forward base velocity
    lateral_vel:  float = -0.3    # penalise lateral drift
    yaw_rate:     float = -0.1    # penalise unwanted yaw
    alive:        float =  0.5    # per-step survival bonus
    control_cost: float = -1e-3   # penalise large torques (energy)
    foot_slip:    float = -0.1    # penalise tangential foot velocity during stance
    body_height:  float = -0.5    # penalise deviation from target height


class RewardFunction:
    """Computes a scalar reward from the current robot state.

    Responsibility (SRP):
        Only compute the reward signal. Does not manage simulation,
        observations, or control — those belong to their own classes.
    """

    TARGET_HEIGHT: float = 0.30   # metres above ground

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self._w = weights or RewardWeights()

    def __call__(
        self,
        lin_vel: np.ndarray,     # (3,) base linear velocity in world frame
        ang_vel: np.ndarray,     # (3,) base angular velocity
        torques: np.ndarray,     # (12,) applied joint torques
        foot_vels: np.ndarray,   # (4,)  foot tangential speed (slip proxy)
        body_height: float,
    ) -> float:
        """Compute and return the scalar step reward."""
        w = self._w
        r  = w.forward_vel  * float(lin_vel[0])
        r += w.lateral_vel  * float(abs(lin_vel[1]))
        r += w.yaw_rate     * float(abs(ang_vel[2]))
        r += w.alive
        r += w.control_cost * float(np.sum(torques ** 2))
        r += w.foot_slip    * float(np.sum(foot_vels ** 2))
        r += w.body_height  * float(abs(body_height - self.TARGET_HEIGHT))
        return r
