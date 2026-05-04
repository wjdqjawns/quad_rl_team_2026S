# QuadRL: Quadruped Reinforcement Learning Framework

QuadRL is a modular reinforcement learning framework for quadruped robot control using MuJoCo simulation. It is designed for research on locomotion, control, and sim-to-real transfer.

---

## Project Metadata

| Field | Description |
|------|------------|
| **Project Title** | QuadRL: Quadruped Reinforcement Learning Framework |
| **Course Name** | CSE402 Reinforcement Learning |
| **Participants** | 김도훈, 이재찬, 정범준 |
| **Institution** | DGIST |
| **Year / Term** | 2026 / Spring |
| **Project Type** | Course Project |
| **Simulation Platform** | MuJoCo |
| **License** | Academic Use Only |

---

## Key Features

- Quadruped robot simulation using MuJoCo
- Reinforcement Learning (PPO / SAC ready structure)
- Modular environment design (Gym-style API)
- Experiment tracking with structured run system
- Evaluation and rollout tools for policy analysis
- Clean separation of core / RL / env / apps layers

---

## Project Structure

```text
src/quadrl/
├── cli.py                  # Main entry point
├── apps/                  # Executable workflows
│   ├── train.py
│   ├── eval.py
│   ├── rollout.py
│   └── sweep.py
│
├── core/                  # Simulation core (stable layer)
│   ├── sim/               # MuJoCo wrapper
│   ├── envs/              # Environment logic
│   ├── robot/             # Robot model + kinematics
│   └── utils/
│
├── rl/                    # Reinforcement learning
│   ├── algorithms/
│   ├── policies/
│   ├── buffers/
│   └── trainer/
│
├── eval/                 # Evaluation & metrics
└── config/               # YAML configurations