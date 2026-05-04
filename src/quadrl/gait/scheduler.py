from __future__ import annotations

import numpy as np

from quadrl.gait.clock import GaitClock
from quadrl.gait.patterns.base_pattern import BaseGaitPattern


class GaitScheduler:
    """Combines a GaitClock and a GaitPattern to produce reference joint targets.

    Responsibility (SRP):
        Convert the current gait phase into reference joint positions and
        a phase observation vector. Does NOT simulate, control, or learn.
    """

    def __init__(
        self,
        pattern: BaseGaitPattern,
        frequency: float,
        dt: float,
        step_height: float = 0.08,
    ) -> None:
        """
        Args:
            pattern:     Gait pattern (TrotPattern, etc.).
            frequency:   Gait frequency in Hz.
            dt:          Control timestep in seconds.
            step_height: Swing foot lift height in meters.
        """
        self._pattern     = pattern
        self._step_height = step_height
        self._clock = GaitClock(
            frequency=frequency,
            leg_offsets=pattern.leg_offsets,
            dt=dt,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def clock(self) -> GaitClock:
        """Expose the underlying clock (e.g., to read phase_obs in the env)."""
        return self._clock

    @property
    def pattern(self) -> BaseGaitPattern:
        return self._pattern

    # ── Public methods ─────────────────────────────────────────────────────────

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance the clock by one control step.

        Returns:
            q_ref     : Reference joint positions, shape (12,).
            phase_obs : Sin/cos phase encoding for observations, shape (8,).
        """
        self._clock.step()
        leg_phases = self._clock.leg_phases()
        q_ref      = self._pattern.reference_joint_positions(leg_phases, self._step_height)
        phase_obs  = self._clock.phase_obs()
        return q_ref, phase_obs

    def reset(self) -> None:
        """Reset the phase clock to t = 0."""
        self._clock.reset()

    def stance_mask(self) -> np.ndarray:
        """Boolean mask: True if leg is in stance phase, shape (4,)."""
        return self._clock.is_stance(self._pattern.stance_ratio)
