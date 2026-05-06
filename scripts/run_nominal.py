"""Standalone runner for the nominal MPC controller.

Runs the Go1 quadruped in MuJoCo simulation using the MPC nominal controller
only — no RL residual. Use this to verify the MPC baseline before training.

Usage (from project root):
    python scripts/run_nominal.py
    python scripts/run_nominal.py --vx 0.5 --yaw 0.3
    python scripts/run_nominal.py --config config/config.yaml --duration 20

Keys in the MuJoCo viewer window:
    Space  - pause/unpause
    Esc    - quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco
import mujoco.viewer
import numpy as np

from quadrl.control.nominal_controller import NominalController
from quadrl.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run nominal MPC controller on Go1 in MuJoCo"
    )
    p.add_argument(
        "--config", type=Path, default=Path("config/config.yaml"),
        help="Path to YAML config (default: config/config.yaml)",
    )
    p.add_argument("--vx",  type=float, default=0.5,  help="Forward velocity [m/s]")
    p.add_argument("--vy",  type=float, default=0.0,  help="Lateral velocity [m/s]")
    p.add_argument("--yaw", type=float, default=0.0,  help="Yaw rate [rad/s]")
    p.add_argument(
        "--duration", type=float, default=10.0, help="Simulation duration [s]"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # Resolve model path relative to project root
    model_path = Path(cfg.env.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    # Build MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)

    # Start from the XML keyframe "home" (standing posture)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Velocity command: [vx, vy, yaw_rate]
    cmd_vel = np.array([args.vx, args.vy, args.yaw], dtype=np.float64)

    controller = NominalController(cfg, model, data)
    controller.reset()

    sim_steps      = round(cfg.env.control_dt / cfg.env.sim_dt)
    max_ctrl_steps = int(args.duration / cfg.env.control_dt)
    log_freq       = max(1, int(1.0 / cfg.env.control_dt))  # once per simulated second

    print("=" * 60)
    print("Nominal MPC controller — Go1")
    print(f"  cmd_vel : vx={args.vx:.2f}  vy={args.vy:.2f}  yaw={args.yaw:.2f}")
    print(f"  duration: {args.duration}s   ctrl_dt={cfg.env.control_dt}s")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(max_ctrl_steps):
            if not viewer.is_running():
                break

            # ── Nominal controller ────────────────────────────────────────────
            ctrl = controller.step(cmd_vel)
            data.ctrl[:12] = ctrl

            # ── Simulate at MuJoCo frequency ──────────────────────────────────
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)

            viewer.sync()

            # ── Logging ───────────────────────────────────────────────────────
            if step % log_freq == 0:
                t  = data.time
                h  = data.qpos[2]
                vx = data.qvel[0]
                vy = data.qvel[1]
                print(f"  t={t:5.1f}s  h={h:.3f}m  vx={vx:+.3f}m/s  vy={vy:+.3f}m/s")

            # ── Fall detection ────────────────────────────────────────────────
            if data.qpos[2] < 0.15:
                print(f"\n[WARN] Robot fell at t={data.time:.2f}s — stopping.")
                break

    print("Simulation finished.")


if __name__ == "__main__":
    main()
