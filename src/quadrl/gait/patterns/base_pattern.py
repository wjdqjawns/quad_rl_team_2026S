from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class BaseGaitPattern(ABC):
    """Defines the temporal and kinematic structure of a single gait.

    Responsibilities (SRP):
        - Declare leg phase offsets (timing).
        - Declare stance ratio (duty cycle).
        - Map per-leg phase → reference joint positions (kinematics).

    Subclasses must implement all abstract members.
    """

    @property
    @abstractmethod
    def leg_offsets(self) -> np.ndarray:
        """Phase offsets for legs [FL, FR, RL, RR] ∈ [0, 1), shape (4,)."""

    @property
    @abstractmethod
    def stance_ratio(self) -> float:
        """Fraction of the gait cycle spent in stance phase."""

    @abstractmethod
    def reference_joint_positions(
        self,
        leg_phases: np.ndarray,
        step_height: float,
    ) -> np.ndarray:
        """Compute reference joint positions from current per-leg phases.

        Args:
            leg_phases:  Per-leg phase ∈ [0, 1), shape (4,).
                         Legs in stance: phi < stance_ratio.
                         Legs in swing:  phi >= stance_ratio.
            step_height: Maximum swing foot lift height in meters.

        Returns:
            Reference joint positions, shape (12,).
            Layout: [FL_hip_a, FL_hip_f, FL_knee,
                     FR_hip_a, FR_hip_f, FR_knee,
                     RL_hip_a, RL_hip_f, RL_knee,
                     RR_hip_a, RR_hip_f, RR_knee].
        """