# Quantum Reinforcement Learning Playground

This playground explores two classic control tasks—**CartPole** and **Lunar Lander**—using quantum-enhanced reinforcement learning and an interactive game interface.

# Video with explanation

https://youtu.be/LqCtLmpwdqI

## Environments Overview

- **CartPole**: Balance a pole upright on a moving cart. Observation space (4): cart position, cart velocity, pole angle, pole angular velocity. Action space (2): push cart left or right. Episode ends when the pole falls beyond ~12° or the cart moves outside the track.
- **Lunar Lander**: Control a lunar module to land between the flags with minimal impact. Observation space (8): lander position/velocity, angle/angular velocity, left/right leg contact sensors. Action space (4): do nothing, fire left thruster, fire main thruster, fire right thruster. Rewards encourage soft, centred landings and penalise crashes or excessive fuel burn.

This project bundles together two reinforcement learning agents—an Actor–Critic
policy gradient and a Deep Q-Network (DQN) with quantum variational circuits—
alongside a playable Lunar Lander game to compare their behaviour against a
human pilot. Training utilities, pre-trained checkpoints, and a CLI experience
are provided so you can retrain the agents or generate new evaluation runs.

## Project Layout

| Path | Description |
| --- | --- |
| `Actor_Critic/` | Training pipeline for the hybrid quantum Actor–Critic agent. Handles learning loops, evaluation, and persistence of artefacts. |
| `DQN/` | Modular DQN implementation supporting both quantum-enhanced and classic neural baselines. Includes replay buffer, quantum encoders, CLI entry point, and comparison notebooks. |
| `Game/` | Playable Lunar Lander split-screen experience (`main.py`) pitting a human against selected pre-trained agents. Menu supports neural DQN, quantum Actor–Critic, and quantum DQN opponents. |
| `Game/models/` | Expected location for the pre-trained checkpoints. Replace these files with retrained weights while keeping the same filenames—for example, new neural DQN weights from `DQN/Classic/Lunar_lander_classic_neuronal.ipynb` should overwrite `Game/models/Classic_DQN_Model.pt`. |
| `executions/` | Created automatically during training to store checkpoints, plots, metrics, and evaluation GIFs. |
| `pyproject.toml` | Project dependencies (Gymnasium, PennyLane, Torch, etc.) and Python toolchain details. |

## Actor–Critic Pipeline (`Actor_Critic/`)

* `config.py` – Centralises hyperparameters for the actor/critic training loops
  (learning rates, discount, quantum architecture, output paths). Inline comments
  document the intent of each field.
* `training.py` – Implements the training loop with live plotting and optional
  output directories. Produces checkpoints, reward plot, and metrics archive.
* `testing.py` – Runs evaluation episodes and produces animated GIFs.
* `main.py` – CLI wiring: spins up the environment, launches training, stores
  artefacts in `executions/actor_critic_lunar_lander_<timestamp>/`, and records
  an evaluation GIF.

The actor is an 8-qubit quantum policy head (`Actor/actor.py`) that mirrors the
original notebook training circuit, while the critic is a classical value network.

## Quantum DQN Pipeline (`DQN/`)

* `DQN/Classic/Lunar_lander_classic_neuronal.ipynb` – trains a neural-network DQN on Lunar Lander to provide a baseline for comparison with the quantum agents.

* `config.py` – Defines shared hyperparameters (replay buffer sizes, epsilon
  schedule, Polyak update cadence) plus environment-specific quantum settings
  for CartPole and Lunar Lander (qubits, feature groupings, solve thresholds).
* `quantum_network.py` – Supplies reusable quantum models. For CartPole, a
  four-qubit circuit with learnable input/output scalers; for Lunar Lander, an
  encoder mapping state groups into rotations followed by qubit expectation
  readings.
* `value_network.py` – Wraps quantum models to expose DQN-friendly utilities
  (epsilon-greedy, RMSProp optimisers with distinct parameter groups).
* `replay_buffer.py` – Simple experience replay buffer matching the legacy
  implementation’s burn-in and sampling strategy.
* `agent.py` – Implements the legacy CartPole DQN training loop with live
  plotting, target network sync, epsilon decay, and checkpoint saving.
* `training.py` – Orchestrates environment creation, quantum model selection,
  and agent instantiation. Returns metrics for persistence.
* `main.py` – CLI entry point. Choose `--environment cartpole` or
  `--environment lunarlander` to train. Outputs artefacts to
  `executions/dqn_<environment>_<timestamp>/`, including model weights, reward
  plot, metrics, and an evaluation GIF rendered from the trained policy.

## Game (`Game/main.py`)

The Pygame split-screen interface lets a human pilot compete against one of the
pre-trained agents. Key bindings:

* Left panel – AI view; right panel – human controlled craft.
* Arrow keys fire thrusters (up = main, left/right = lateral). Escape returns to
  menu, `R` restarts the episode.
* Menu options allow selecting the quantum Actor–Critic or quantum DQN checkpoints
  found in `Game/models/`, or opening the in-game controls screen.

## Quantum Variational Circuits (QVCs)

* **Actor–Critic** – Uses an 8-qubit layered circuit with learnable rotations
  (`Actor/actor.py`). Observations are arctan-scaled before entering the circuit,
  whose outputs are linearly rescaled to logits for a categorical distribution.
* **DQN (CartPole)** – Four-qubit circuit with repeated encode/variational
  layers, plus learnable input/output scalers to adapt the classical features.
* **DQN (Lunar Lander)** – Feature groups aggregate the 8D state into qubit
  rotations, including directional thresholds for leg contact sensors, producing
  a per-action expectation value readout.

All circuits run on PennyLane’s `lightning.qubit` simulator by default.

## Technologies Used

* **Python 3.11** – Primary language.
* **PyTorch** – Classical components, optimisers, and tensor ops.
* **PennyLane** – Quantum circuit definition and differentiation.
* **Gymnasium** – Reinforcement learning environments (LunarLander-v3,
  CartPole-v1).
* **Pygame** – Desktop game interface.
* **Matplotlib** – Real-time and saved plotting of training progress.
* **ImageIO** – Generation of evaluation GIFs.

## Getting Started

```bash
uv sync                # install dependencies

# Train the actor-critic agent (Lunar Lander)
uv run python3 -m actor_critic.main

# Train the quantum DQN (CartPole or Lunar Lander)
uv run python3 -m DQN.main --environment cartpole
uv run python3 -m DQN.main --environment lunarlander

# Launch the game
uv run python3 -m game.main
```

Each training command writes artefacts under `executions/<method>_<environment>_<timestamp>/`.
Inside each run folder you'll find model checkpoints, reward plots, metrics, and evaluation GIFs.
Reinserting a trained policy into the game simply involves copying the relevant checkpoints (keeping
their filenames) into `Game/models/`.
Use the existing names so the menu recognises them:
- `Game/models/Quantum_Actor_Model.pt` – Actor network for the Actor–Critic agent.
- `Game/models/Quantum_Critic_Model.pt` – Critic network for the Actor–Critic agent.
- `Game/models/Classic_DQN_Model.pt` – Neural DQN policy loaded when selecting option 1 in the game menu.
- `Game/models/Quantum_DQN_Model.pt` – Quantum DQN policy loaded when selecting option 3 in the game menu.
