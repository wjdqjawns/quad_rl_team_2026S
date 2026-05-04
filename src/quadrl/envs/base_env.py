from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym


class BaseQuadrupedEnv(gym.Env, ABC):
    """Abstract base class for MuJoCo quadruped environments.

    Responsibility (SRP):
        Define the gymnasium.Env interface and enforce that all
        subclasses implement the three core internal methods.

    Subclasses must implement:
        _get_obs()      → np.ndarray
        _is_terminated() → bool
        _get_info()     → dict
    """

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Assemble and return the current observation vector."""

    @abstractmethod
    def _is_terminated(self) -> bool:
        """Return True when the episode should end (fall, bound violation, …)."""

    @abstractmethod
    def _get_info(self) -> dict:
        """Return auxiliary diagnostic information for logging / debugging."""
