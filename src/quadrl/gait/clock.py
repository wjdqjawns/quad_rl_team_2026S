from __future__ import annotations

import numpy as np

NUM_LEGS = 4  # [FL, FR, RL, RR]


class GaitClock:
    """Advances per-leg phase signals at a fixed gait frequency.

    Phase φ ∈ [0, 1) advances uniformly with time.
    Each leg has a constant phase offset that defines the gait pattern.
    """

    def __init__(
        self,
        frequency: float,
        leg_offsets: np.ndarray,
        dt: float,
    ) -> None:
        """
        Args:
            frequency:   Gait frequency in Hz.
            leg_offsets: Phase offset for each leg ∈ [0, 1), shape (4,).
                         Order: [FL, FR, RL, RR].
            dt:          Control timestep in seconds.
        """
        if len(leg_offsets) != NUM_LEGS:
            raise ValueError(f"leg_offsets must have length {NUM_LEGS}")

        self._freq    = frequency
        self._offsets = np.asarray(leg_offsets, dtype=np.float64)
        self._dt      = dt
        self._phase   = 0.0   # global phase ∈ [0, 1)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def phase(self) -> float:
        """Global gait phase in [0, 1)."""
        return self._phase

    # ── Public methods ─────────────────────────────────────────────────────────

    def leg_phases(self) -> np.ndarray:
        """Per-leg phase in [0, 1), shape (4,)."""
        return (self._phase + self._offsets) % 1.0

    def phase_obs(self) -> np.ndarray:
        """Sin/cos encoding of per-leg phase for use as observation, shape (8,).

        Using sin/cos avoids the discontinuity at φ = 0 ↔ 1.
        """
        phi = 2.0 * np.pi * self.leg_phases()
        return np.concatenate([np.sin(phi), np.cos(phi)])

    def is_stance(self, stance_ratio: float) -> np.ndarray:
        """Boolean mask: True if leg is in stance phase, shape (4,)."""
        return self.leg_phases() < stance_ratio

    def step(self) -> None:
        """Advance global phase by one control timestep."""
        self._phase = (self._phase + self._freq * self._dt) % 1.0

    def reset(self) -> None:
        """Reset global phase to zero."""
        self._phase = 0.0
