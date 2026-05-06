"""Low-level joint controller for quadruped locomotion.

Paper Eq. (23)-(25):
    τ_ff,i = J_i^T * u_i                              (stance feedforward)
    τ_fb,i = J_i^T * [K_P * e_p + K_D * e_v]         (swing Cartesian PD)
    τ_i    = τ_ff,i + τ_fb,i

The Go1 XML uses position actuators (kp_act = 100 Nm/rad for all joints).
To achieve torque τ_des, we set:
    ctrl = q + τ_des / kp_act          (implicit torque control)

This is valid because the actuator force = kp_act * (ctrl - q), so
setting ctrl = q + τ/kp_act produces exactly the desired torque.

Stance legs → feedforward from MPC GRFs (τ = -J^T F), then ctrl.
Swing  legs → Cartesian PD to track foot trajectory, then ctrl.

Leg index convention: [FR=0, FL=1, RR=2, RL=3]  (matches MuJoCo actuator order)
"""
from __future__ import annotations

import mujoco
import numpy as np

_LEG_NAMES = ("FR", "FL", "RR", "RL")
_NUM_LEGS  = 4
_JOINTS_PER_LEG = 3


class LowLevelController:
    """Per-leg joint controller bridging MPC GRFs and MuJoCo position actuators.

    Stance legs:
        τ_ff = -J^T * F_mpc  (feedforward from MPC)
        ctrl = q + τ_ff / kp_actuator

    Swing legs:
        F_cart = Kp_cart * (p_des - p_foot) + Kd_cart * (v_des - v_foot)
        τ_fb = J^T * F_cart  (Cartesian PD)
        ctrl = q + τ_fb / kp_actuator

    Args:
        model:       MuJoCo model.
        data:        MuJoCo data (updated externally before each call).
        kp_actuator: Position actuator gain [Nm/rad] from XML (default 100).
        torque_limit: Maximum |torque| per joint [Nm].
        kp_cart:     Cartesian position stiffness for swing [N/m].
        kd_cart:     Cartesian velocity damping for swing [N·s/m].
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        kp_actuator: float = 100.0,
        torque_limit: float = 33.5,
        kp_cart: float = 500.0,
        kd_cart: float = 50.0,
    ) -> None:
        self._model    = model
        self._data     = data
        self._kp_act   = kp_actuator
        self._tau_lim  = torque_limit
        self._kp_cart  = kp_cart
        self._kd_cart  = kd_cart

        self._foot_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in _LEG_NAMES
        ]

    def compute(
        self,
        grfs: np.ndarray,              # (4, 3) MPC GRFs in world frame
        stance_mask: np.ndarray,       # (4,) bool
        swing_pos_des: np.ndarray,     # (4, 3) desired foot positions  (swing only)
        swing_vel_des: np.ndarray,     # (4, 3) desired foot velocities (swing only)
    ) -> np.ndarray:
        """Compute MuJoCo ctrl (position targets) for all 12 joints.

        Args:
            grfs:          (4, 3) GRFs from MPC; swing-leg entries are ignored.
            stance_mask:   (4,) bool stance/swing mask.
            swing_pos_des: (4, 3) desired foot positions for swing legs.
            swing_vel_des: (4, 3) desired foot velocities for swing legs.

        Returns:
            ctrl: (12,) position targets for MuJoCo position actuators.
        """
        q   = self._data.qpos[7:19]    # (12,) current joint positions
        qvel = self._data.qvel          # (nv,) full velocity vector
        n_v = self._model.nv

        ctrl = q.copy()   # neutral: zero actuator force initially
        jacp = np.zeros((3, n_v))

        for i in range(_NUM_LEGS):
            jacp[:] = 0.0
            sid = self._foot_site_ids[i]
            mujoco.mj_jacSite(self._model, self._data, jacp, None, sid)

            # Per-leg joint Jacobian (columns for leg i joints only)
            col_start = 6 + i * _JOINTS_PER_LEG
            J_leg = jacp[:, col_start: col_start + _JOINTS_PER_LEG]   # (3, 3)

            if stance_mask[i]:
                # ── Stance: feedforward from MPC GRF ─────────────────────────
                tau_leg = J_leg.T @ (-grfs[i])                          # (3,)

            else:
                # ── Swing: Cartesian PD tracking foot trajectory ──────────────
                p_foot = self._data.site_xpos[sid].copy()              # (3,)
                v_foot = jacp @ qvel                                    # (3,)

                e_p = swing_pos_des[i] - p_foot
                e_v = swing_vel_des[i] - v_foot
                F_cart = self._kp_cart * e_p + self._kd_cart * e_v     # (3,)
                tau_leg = J_leg.T @ F_cart                              # (3,)

            tau_leg = np.clip(tau_leg, -self._tau_lim, self._tau_lim)

            # Convert torque → position target: ctrl = q + τ / kp
            j0 = i * _JOINTS_PER_LEG
            ctrl[j0: j0 + _JOINTS_PER_LEG] = (
                q[j0: j0 + _JOINTS_PER_LEG] + tau_leg / self._kp_act
            )

        return ctrl
