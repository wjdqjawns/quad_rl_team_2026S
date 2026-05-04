from __future__ import annotations

import numpy as np

class PDController:
    """Per-joint proportional-derivative position controller.

    Computes joint torques as:
        τ = kp * (q_des − q) − kd * dq

    Responsibility (SRP):
        Convert joint position targets into torques. Does not manage
        gait references or RL actions — those are composed upstream.
    """

    def __init__(
        self,
        kp: float,
        kd: float,
        torque_limit: float,
    ) -> None:
        """
        Args:
            kp:            Proportional gain (position error → torque).
            kd:            Derivative gain (velocity → damping torque).
            torque_limit:  Maximum absolute torque per joint (Nm).
        """
        if kp < 0 or kd < 0:
            raise ValueError("kp and kd must be non-negative")
        if torque_limit <= 0:
            raise ValueError("torque_limit must be positive")

        self._kp    = kp
        self._kd    = kd
        self._limit = torque_limit

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def kp(self) -> float:
        return self._kp

    @property
    def kd(self) -> float:
        return self._kd

    @property
    def torque_limit(self) -> float:
        return self._limit

    # ── Public methods ─────────────────────────────────────────────────────────

    def compute(
        self,
        q_des: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """Compute torques for all joints.

        Args:
            q_des: Desired joint positions,   shape (n_joints,).
            q:     Current joint positions,   shape (n_joints,).
            dq:    Current joint velocities,  shape (n_joints,).

        Returns:
            Torques clipped to ±torque_limit, shape (n_joints,).
        """
        tau = self._kp * (q_des - q) - self._kd * dq
        return np.clip(tau, -self._limit, self._limit)
