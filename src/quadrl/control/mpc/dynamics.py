"""Single Rigid Body Dynamics (SRBD) linearization.

Reference:
    Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through
    Convex Model-Predictive Control", IROS 2018.

State vector (12-dim, world frame):
    x = [Θ(3), p(3), ω(3), v(3)]
    Θ = [roll, pitch, yaw]  body Euler angles
    p = body CoM position
    ω = body angular velocity
    v = body linear velocity

Control vector (12-dim):
    u = [F_FL(3), F_FR(3), F_RL(3), F_RR(3)]  ground reaction forces
"""
from __future__ import annotations

import numpy as np

STATE_DIM = 12   # [Θ(3), p(3), ω(3), v(3)]
CTRL_DIM  = 12   # [F_FL(3), F_FR(3), F_RL(3), F_RR(3)]
NUM_LEGS  = 4


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of vector v (used for cross-product)."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles → 3×3 rotation matrix (body → world)."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,             cp*cr  ],
    ])


def compute_A(dt: float) -> np.ndarray:
    """State-transition matrix A (constant, independent of foot positions).

    Discretised with forward Euler at timestep dt:
        Θ_{k+1} = Θ_k + dt * ω_k
        p_{k+1} = p_k + dt * v_k
        ω_{k+1} = ω_k          (updated through B)
        v_{k+1} = v_k          (updated through B)

    Args:
        dt: MPC discretisation timestep (seconds).

    Returns:
        A: (STATE_DIM, STATE_DIM)
    """
    A = np.eye(STATE_DIM)
    A[0:3, 6:9]  = dt * np.eye(3)   # Θ += dt * ω
    A[3:6, 9:12] = dt * np.eye(3)   # p  += dt * v
    return A


def compute_B(
    inertia_world: np.ndarray,
    foot_positions: np.ndarray,
    com_position: np.ndarray,
    mass: float,
    dt: float,
    contact: np.ndarray,
) -> np.ndarray:
    """Input matrix B_k (depends on foot positions and contact state).

    For each stance leg i with foot position r_i (relative to CoM):
        ω_dot += I^{-1} * (r_i × F_i)
        v_dot += F_i / m

    Args:
        inertia_world: (3, 3) body inertia tensor in world frame.
        foot_positions: (4, 3) foot positions in world frame.
        com_position:   (3,) CoM position in world frame.
        mass:           Total robot mass (kg).
        dt:             MPC discretisation timestep (seconds).
        contact:        (4,) bool array — True if leg is in stance.

    Returns:
        B: (STATE_DIM, CTRL_DIM)
    """
    B = np.zeros((STATE_DIM, CTRL_DIM))
    I_inv = np.linalg.inv(inertia_world)

    for i in range(NUM_LEGS):
        col = slice(i * 3, i * 3 + 3)
        if not contact[i]:
            continue   # swing foot: zero contribution

        r = foot_positions[i] - com_position   # CoM-to-foot vector (world)
        B[6:9,  col] = dt * I_inv @ skew(r)   # ω row
        B[9:12, col] = dt / mass * np.eye(3)  # v row

    return B


def gravity_term(dt: float) -> np.ndarray:
    """Constant gravity contribution to the state, shape (STATE_DIM,).

    v_{k+1} += dt * g    (g = [0, 0, -9.81])
    """
    g = np.zeros(STATE_DIM)
    g[11] = -9.81 * dt   # v_z component
    return g
