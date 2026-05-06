"""Heuristic footstep planner using Raibert's method.

Paper Eq. (5): p_{f,i}^+ = p_{f,i}^- + δp_{f,heuristic,i}

For each leg:
  - Stance: hold current foot position (no-slip assumption).
  - Swing:  land under the shoulder plus a velocity feedforward term.
            p_f,i = p_shoulder,i + v_cmd * T_stance / 2

Also provides swing trajectory for Cartesian-space swing control:
  p_des(s) = lerp(p_liftoff, p_land, s) + [0, 0, h_max * sin²(π*s)]
where s ∈ [0, 1] is normalised swing progress.

sin²(πs) ensures zero foot velocity at liftoff AND touchdown, preventing
the impact impulse that arises with the naive sin(πs) profile.

Reference:
    Raibert (1986), Legged Robots That Balance.
    Pratt et al. (2006), Capture point.
"""
from __future__ import annotations

import mujoco
import numpy as np

from quadrl.control.mpc.dynamics import euler_to_rotation

# Leg order matches MuJoCo site/actuator order: [FR, FL, RR, RL]
_LEG_NAMES = ("FR", "FL", "RR", "RL")
_NUM_LEGS  = 4

# Hip (shoulder) offsets in body frame [x, y, z] from XML body positions
_HIP_OFFSETS_BODY = np.array([
    [ 0.1881, -0.04675, 0.0],   # FR
    [ 0.1881,  0.04675, 0.0],   # FL
    [-0.1881, -0.04675, 0.0],   # RR
    [-0.1881,  0.04675, 0.0],   # RL
], dtype=np.float64)

# Foot sphere radius from XML (z-offset for foot above ground contact point)
_FOOT_RADIUS = 0.023   # m


class FootstepPlanner:
    """Raibert heuristic footstep planner with Cartesian swing trajectory.

    Args:
        model:        MuJoCo model (for site IDs).
        data:         MuJoCo data (updated externally before each call).
        stance_ratio: Fraction of gait cycle in stance (e.g. 0.6).
        gait_freq:    Gait frequency in Hz.
        step_height:  Max foot clearance during swing (m).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stance_ratio: float,
        gait_freq: float,
        step_height: float = 0.08,
    ) -> None:
        self._data         = data
        self._stance_ratio = stance_ratio
        self._gait_freq    = gait_freq
        self._T_stance     = stance_ratio / gait_freq       # stance duration (s)
        self._T_swing      = (1.0 - stance_ratio) / gait_freq  # swing duration (s)
        self._step_height  = step_height

        self._foot_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in _LEG_NAMES
        ]

        # Liftoff state (initialised on first update call)
        self._prev_stance: np.ndarray | None = None
        self._liftoff_pos = np.zeros((_NUM_LEGS, 3))

    # ── Public interface ────────────────────────────────────────────────────────

    def update(self, stance_mask: np.ndarray) -> None:
        """Detect stance→swing transitions and record liftoff positions.

        Must be called once per control step, BEFORE compute() or swing_trajectory().

        Args:
            stance_mask: (4,) bool, current stance state.
        """
        if self._prev_stance is not None:
            for i in range(_NUM_LEGS):
                if self._prev_stance[i] and not stance_mask[i]:
                    # Leg i just lifted off — save world-frame foot position
                    self._liftoff_pos[i] = (
                        self._data.site_xpos[self._foot_site_ids[i]].copy()
                    )
        self._prev_stance = stance_mask.copy()

    def compute(
        self,
        com_pos: np.ndarray,     # (3,) CoM position in world frame
        euler: np.ndarray,        # (3,) roll, pitch, yaw
        cmd_vel: np.ndarray,     # (3,) [vx, vy, yaw_rate]
        stance_mask: np.ndarray, # (4,) bool
    ) -> np.ndarray:
        """Compute planned foot positions for MPC B-matrix.

        Stance legs → hold current foot position.
        Swing legs  → Raibert heuristic landing target.

        Args:
            com_pos:     CoM position in world frame.
            euler:       Body Euler angles [roll, pitch, yaw].
            cmd_vel:     Velocity command [vx, vy, yaw_rate].
            stance_mask: Stance/swing mask per leg.

        Returns:
            foot_targets: (4, 3) planned foot positions in world frame.
        """
        R = euler_to_rotation(*euler)
        foot_targets = np.zeros((_NUM_LEGS, 3))
        v_des = np.array([cmd_vel[0], cmd_vel[1], 0.0])

        for i in range(_NUM_LEGS):
            if stance_mask[i]:
                foot_targets[i] = self._data.site_xpos[self._foot_site_ids[i]].copy()
            else:
                foot_targets[i] = self._landing_target(i, com_pos, R, v_des)

        return foot_targets

    def swing_trajectory(
        self,
        leg_idx: int,
        swing_progress: float,   # s ∈ [0, 1]
        com_pos: np.ndarray,
        euler: np.ndarray,
        cmd_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired foot state (position, velocity) during swing.

        Trajectory: p_des(s) = lerp(p_lift, p_land, s) + [0, 0, h_max * sin²(πs)]

        sin²(πs) profile gives zero foot velocity at liftoff (s=0) and
        touchdown (s=1), eliminating impact impulses.

        Args:
            leg_idx:       Leg index [0=FR, 1=FL, 2=RR, 3=RL].
            swing_progress: Normalised swing progress s ∈ [0, 1].
            com_pos:       Current CoM position.
            euler:         Current body Euler angles.
            cmd_vel:       Velocity command.

        Returns:
            p_des: (3,) desired foot position in world frame.
            v_des: (3,) desired foot velocity in world frame.
        """
        R     = euler_to_rotation(*euler)
        v_des_body = np.array([cmd_vel[0], cmd_vel[1], 0.0])
        p_land = self._landing_target(leg_idx, com_pos, R, v_des_body)
        p_lift = self._liftoff_pos[leg_idx]

        s = np.clip(swing_progress, 0.0, 1.0)

        # Horizontal: cosine profile — zero foot velocity at liftoff AND touchdown
        # p(s) = p_lift + (p_land - p_lift) * 0.5*(1 - cos(πs))
        # dp/ds = (p_land - p_lift) * 0.5*π*sin(πs)  → zero at s=0 and s=1
        cos_ramp = 0.5 * (1.0 - np.cos(np.pi * s))
        p_des_xy = p_lift[:2] + (p_land[:2] - p_lift[:2]) * cos_ramp
        v_des_xy = (p_land[:2] - p_lift[:2]) * (0.5 * np.pi * np.sin(np.pi * s)) / self._T_swing

        # Vertical: sin² arc — zero velocity at liftoff and touchdown
        h = self._step_height
        sin2 = np.sin(np.pi * s) ** 2
        p_des_z = _FOOT_RADIUS + h * sin2
        # d(sin²(πs))/dt = h * π * sin(2πs) / T_swing
        v_des_z = h * np.pi * np.sin(2.0 * np.pi * s) / self._T_swing

        p_des = np.array([p_des_xy[0], p_des_xy[1], p_des_z])
        v_des = np.array([v_des_xy[0], v_des_xy[1], v_des_z])
        return p_des, v_des

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _landing_target(
        self,
        leg_idx: int,
        com_pos: np.ndarray,
        R_body: np.ndarray,
        v_des: np.ndarray,
    ) -> np.ndarray:
        """Raibert heuristic landing target for leg leg_idx."""
        p_shoulder = com_pos + R_body @ _HIP_OFFSETS_BODY[leg_idx]
        p_land = p_shoulder + v_des * (self._T_stance * 0.5)
        p_land[2] = _FOOT_RADIUS   # flat-ground projection
        return p_land
