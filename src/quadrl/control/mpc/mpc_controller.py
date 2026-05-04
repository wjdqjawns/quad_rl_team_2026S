"""MPC Controller — main interface.

Responsibility (SRP):
    Given the current robot state and gait scheduler output, solve the
    SRBD convex MPC QP and return reference joint torques.
    Does NOT handle simulation, RL policy, or gait scheduling.

Residual RL interface:
    τ_cmd = τ_mpc + action_scale * τ_residual
    where τ_residual ∈ [-1, 1]^12 is the RL policy's output.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from quadrl.control.mpc.dynamics import (
    STATE_DIM,
    CTRL_DIM,
    NUM_LEGS,
    compute_A,
    compute_B,
    gravity_term,
)
from quadrl.control.mpc.qp_problem import build_qp, solve_qp
from quadrl.utils.config import MpcConfig


@dataclass
class RobotState:
    """Snapshot of body state required by the MPC (world frame)."""
    euler:      np.ndarray   # (3,) roll, pitch, yaw
    position:   np.ndarray   # (3,) CoM position
    ang_vel:    np.ndarray   # (3,) angular velocity
    lin_vel:    np.ndarray   # (3,) linear velocity

    def to_vec(self) -> np.ndarray:
        return np.concatenate([self.euler, self.position, self.ang_vel, self.lin_vel])


class MPCController:
    """Convex MPC based on Single Rigid Body Dynamics.

    Computes optimal ground reaction forces (GRFs) over a receding horizon,
    then converts them to joint torques via foot Jacobians.

    Args:
        cfg:   MPC configuration (horizon, weights, friction, …).
        model: MuJoCo MjModel (for mass, inertia, Jacobian queries).
        data:  MuJoCo MjData (updated externally before each call).
    """

    # MuJoCo body / site names for Go1 foot sites (must match go1.xml)
    _FOOT_SITE_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __init__(self, cfg: MpcConfig, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._cfg   = cfg
        self._model = model
        self._data  = data
        self._N     = cfg.horizon

        # Robot physical parameters (from MuJoCo model)
        self._mass = float(model.body_subtreemass[1])   # total mass (root body)
        self._A    = compute_A(cfg.dt_mpc)

        # Cost matrices
        self._Q = np.diag(cfg.state_weights)    # (STATE_DIM,)
        self._R = np.diag(cfg.ctrl_weights)     # (CTRL_DIM,)

        # Foot site IDs for Jacobian queries
        self._foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self._FOOT_SITE_NAMES
        ]

    # ── Public interface ───────────────────────────────────────────────────────

    def compute(
        self,
        state: RobotState,
        contact_schedule: np.ndarray,   # (N, 4) bool stance mask over horizon
        desired_velocity: np.ndarray,   # (3,) desired body linear velocity
    ) -> np.ndarray:
        """Solve MPC QP and return reference joint torques.

        Args:
            state:            Current robot body state.
            contact_schedule: Stance contact plan over the horizon, shape (N, 4).
            desired_velocity: Target body velocity [vx, vy, vz].

        Returns:
            tau_mpc: Joint torques from MPC, shape (12,).
        """
        foot_pos = self._foot_positions()
        com_pos  = state.position
        I_w      = self._inertia_world(state.euler)

        # Build B matrices and gravity terms for each step in the horizon
        Bs = [
            compute_B(I_w, foot_pos, com_pos, self._mass, self._cfg.dt_mpc, contact_schedule[k])
            for k in range(self._N)
        ]
        gs = [gravity_term(self._cfg.dt_mpc)] * self._N

        # Reference trajectory: constant desired velocity, current height
        x_ref = self._build_reference(state, desired_velocity)

        qp   = build_qp(self._A, Bs, gs, state.to_vec(), x_ref,
                        self._Q, self._R, contact_schedule,
                        self._cfg.friction_mu, self._cfg.f_max)
        U    = solve_qp(qp)

        # Extract first-step GRFs and convert to joint torques
        grfs = U[:CTRL_DIM].reshape(NUM_LEGS, 3)
        return self._grfs_to_torques(grfs)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _foot_positions(self) -> np.ndarray:
        """World-frame foot positions from MuJoCo site data, shape (4, 3)."""
        return np.array([self._data.site_xpos[sid] for sid in self._foot_ids])

    def _inertia_world(self, euler: np.ndarray) -> np.ndarray:
        """Approximate body inertia in world frame via R I_body R^T."""
        from quadrl.control.mpc.dynamics import euler_to_rotation
        R = euler_to_rotation(*euler)
        # Body inertia from MuJoCo (body index 1 = root floating body)
        I_body = np.diag(self._model.body_inertia[1])
        return R @ I_body @ R.T

    def _build_reference(
        self,
        state: RobotState,
        desired_velocity: np.ndarray,
    ) -> np.ndarray:
        """Constant-velocity reference trajectory over horizon, shape (N, STATE_DIM)."""
        cfg    = self._cfg
        x_ref  = np.zeros((self._N, STATE_DIM))
        dt_mpc = cfg.dt_mpc

        for k in range(self._N):
            # Orientation: track zero roll/pitch, keep current yaw
            x_ref[k, 0:3] = [0.0, 0.0, state.euler[2]]
            # Position: propagate from current at desired velocity
            x_ref[k, 3:6] = state.position + desired_velocity * (k + 1) * dt_mpc
            x_ref[k, 5]   = cfg.target_height       # keep body height
            # Angular velocity: zero (no rotation commanded)
            x_ref[k, 6:9] = [0.0, 0.0, 0.0]
            # Linear velocity: desired
            x_ref[k, 9:12] = desired_velocity

        return x_ref

    def _grfs_to_torques(self, grfs: np.ndarray) -> np.ndarray:
        """Convert GRFs to joint torques via foot Jacobians: τ = -J^T F.

        Args:
            grfs: (4, 3) GRFs in world frame.

        Returns:
            tau: (12,) joint torques.
        """
        tau = np.zeros(12)
        n_v = self._model.nv

        for i, sid in enumerate(self._foot_ids):
            jacp  = np.zeros((3, n_v))
            jacr  = np.zeros((3, n_v))
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, sid)

            # Joint columns only (skip 6 floating-base DoFs)
            J_joints = jacp[:, 6:]   # (3, 12)
            tau += J_joints.T @ (-grfs[i])   # reaction: τ = -J^T F

        return np.clip(tau, -self._cfg.torque_limit, self._cfg.torque_limit)
