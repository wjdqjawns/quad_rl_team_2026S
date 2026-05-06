"""Interactive nominal MPC controller visualiser for Go1.

Controls (keyboard — focus the MuJoCo viewer window):
    W / ↑       increase forward speed  (+0.1 m/s)
    S / ↓       decrease forward speed  (-0.1 m/s)
    A / ←       turn left               (+0.2 rad/s yaw)
    D / →       turn right              (-0.2 rad/s yaw)
    Q / Shift+A lateral left            (+0.1 m/s vy)
    E / Shift+D lateral right           (-0.1 m/s vy)
    Space       stop (zero all velocity)
    R           reset simulation
    Esc / Ctrl  quit

Usage (from project root):
    source .venv/bin/activate
    python scripts/run_nominal.py
    python scripts/run_nominal.py --vx 0.3
    python scripts/run_nominal.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco
import mujoco.viewer
import numpy as np

from quadrl.control.nominal_controller import NominalController
from quadrl.utils.config import load_config

# ── GLFW key codes ──────────────────────────────────────────────────────────────
_KEY_W         = 87
_KEY_A         = 65
_KEY_S         = 83
_KEY_D         = 68
_KEY_Q         = 81
_KEY_E         = 69
_KEY_R         = 82
_KEY_SPACE     = 32
_KEY_ARROW_UP  = 265
_KEY_ARROW_DOWN  = 264
_KEY_ARROW_LEFT  = 263
_KEY_ARROW_RIGHT = 262

# Velocity increments per key press
_DVX  = 0.1   # m/s
_DVY  = 0.1   # m/s
_DYAW = 0.2   # rad/s
_VX_MAX  = 2.0
_VY_MAX  = 0.5
_YAW_MAX = 1.0


class SimState:
    """Shared mutable state between the main loop and key callback."""

    def __init__(self, vx: float, vy: float, yaw: float) -> None:
        self._lock    = threading.Lock()
        self._cmd_vel = np.array([vx, vy, yaw], dtype=np.float64)
        self.reset_requested = False
        self.quit_requested  = False

    @property
    def cmd_vel(self) -> np.ndarray:
        with self._lock:
            return self._cmd_vel.copy()

    def _update(self, dvx: float = 0.0, dvy: float = 0.0, dyaw: float = 0.0) -> None:
        with self._lock:
            self._cmd_vel[0] = np.clip(self._cmd_vel[0] + dvx,  -_VX_MAX,  _VX_MAX)
            self._cmd_vel[1] = np.clip(self._cmd_vel[1] + dvy,  -_VY_MAX,  _VY_MAX)
            self._cmd_vel[2] = np.clip(self._cmd_vel[2] + dyaw, -_YAW_MAX, _YAW_MAX)

    def key_callback(self, keycode: int) -> None:
        if keycode in (_KEY_W, _KEY_ARROW_UP):
            self._update(dvx=+_DVX)
        elif keycode in (_KEY_S, _KEY_ARROW_DOWN):
            self._update(dvx=-_DVX)
        elif keycode in (_KEY_A, _KEY_ARROW_LEFT):
            self._update(dyaw=+_DYAW)
        elif keycode in (_KEY_D, _KEY_ARROW_RIGHT):
            self._update(dyaw=-_DYAW)
        elif keycode == _KEY_Q:
            self._update(dvy=+_DVY)
        elif keycode == _KEY_E:
            self._update(dvy=-_DVY)
        elif keycode == _KEY_SPACE:
            with self._lock:
                self._cmd_vel[:] = 0.0
        elif keycode == _KEY_R:
            with self._lock:
                self._cmd_vel[:] = 0.0
            self.reset_requested = True

    def print_status(self, t: float, h: float, vx_actual: float) -> None:
        with self._lock:
            c = self._cmd_vel.copy()
        print(
            f"\r  t={t:5.1f}s  h={h:.3f}m  vx={vx_actual:+.2f}m/s  "
            f"cmd=[vx={c[0]:+.2f} vy={c[1]:+.2f} yaw={c[2]:+.2f}]   ",
            end="", flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive nominal MPC viewer — Go1")
    p.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    p.add_argument("--vx",  type=float, default=0.0,  help="Initial forward velocity [m/s]")
    p.add_argument("--vy",  type=float, default=0.0,  help="Initial lateral velocity [m/s]")
    p.add_argument("--yaw", type=float, default=0.0,  help="Initial yaw rate [rad/s]")
    return p.parse_args()


def _load_model(cfg_env) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = Path(cfg_env.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)
    return model, data


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    model, data = _load_model(cfg.env)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    state      = SimState(args.vx, args.vy, args.yaw)
    controller = NominalController(cfg, model, data)
    controller.reset()

    sim_steps = round(cfg.env.control_dt / cfg.env.sim_dt)
    log_every = max(1, int(1.0 / cfg.env.control_dt))   # ~1 Hz console update
    step      = 0

    print("=" * 60)
    print("Nominal MPC Controller — Go1 Interactive Viewer")
    print("=" * 60)
    print("  W/↑  forward   S/↓  backward   Space  stop")
    print("  A/←  turn left  D/→  turn right")
    print("  Q    strafe left  E  strafe right")
    print("  R    reset simulation")
    print("=" * 60)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=state.key_callback
    ) as viewer:
        viewer.cam.azimuth   = 180
        viewer.cam.elevation = -15
        viewer.cam.distance  = 2.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

        while viewer.is_running():
            # Reset if R was pressed
            if state.reset_requested:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                controller.reset()
                step = 0
                state.reset_requested = False
                print("\n[RESET]")

            cmd_vel = state.cmd_vel

            # Nominal controller step
            ctrl = controller.step(cmd_vel)
            data.ctrl[:12] = ctrl

            # Simulate
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)

            viewer.sync()
            step += 1

            # Console status
            if step % log_every == 0:
                state.print_status(data.time, data.qpos[2], data.qvel[0])

            # Fall detection
            if data.qpos[2] < 0.15:
                print(f"\n[WARN] Robot fell at t={data.time:.2f}s — press R to reset.")
                # Keep viewer open so user can press R
                while viewer.is_running() and not state.reset_requested:
                    viewer.sync()
                if not viewer.is_running():
                    break

    print("\nSimulation finished.")


if __name__ == "__main__":
    main()
