from __future__ import annotations

import numpy as np

from quadrl.gait.patterns.base_pattern import BaseGaitPattern

# Nominal standing joint angles (radians) for Unitree Go1
# Order per leg: [hip_abduction, hip_flexion, knee]
_HIP_A  =  0.0
_HIP_F  =  0.67   # ~38 deg
_KNEE   = -1.30   # ~-74 deg

_STAND_POS: np.ndarray = np.array(
    [_HIP_A, _HIP_F, _KNEE] * 4, dtype=np.float64
)


class TrotPattern(BaseGaitPattern):
    """Diagonal trot: FL+RR in phase, FR+RL offset by 0.5.

    Stance ratio defaults to 0.6 (60% of cycle in contact).
    During swing the hip_f and knee follow a sinusoidal trajectory
    to lift the foot, approximating a smooth swing arc.
    """

    _STANCE_RATIO: float = 0.6

    @property
    def leg_offsets(self) -> np.ndarray:
        # FL=0.0, FR=0.5, RL=0.5, RR=0.0  →  diagonal pairs move together
        return np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64)

    @property
    def stance_ratio(self) -> float:
        return self._STANCE_RATIO

    def reference_joint_positions(
        self,
        leg_phases: np.ndarray,
        step_height: float,
    ) -> np.ndarray:
        q_ref = _STAND_POS.copy()
        swing_duration = 1.0 - self._STANCE_RATIO

        for i, phi in enumerate(leg_phases):
            if phi >= self._STANCE_RATIO:
                # Normalised progress through swing phase ∈ [0, 1]
                swing_frac = (phi - self._STANCE_RATIO) / swing_duration

                # Sinusoidal deltas: peak at mid-swing (swing_frac = 0.5)
                k = np.pi * swing_frac
                hip_f_delta =  0.20 * np.sin(k)   # flex hip to lift leg
                knee_delta  = -0.40 * np.sin(k)   # extend knee symmetrically

                q_ref[i * 3 + 1] += hip_f_delta
                q_ref[i * 3 + 2] += knee_delta

        return q_ref
