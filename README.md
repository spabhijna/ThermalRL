# ThermalRL

> Deep Q-Network agent for data center cooling optimization — beats rule-based control with 2.5% lower PUE and 6% higher cumulative reward across 500 evaluation episodes.

**SDG 13 — Climate Action** | Python · PyTorch · NumPy · uv

---

## Problem Statement

Data centers consume 1–2% of global electricity, with cooling accounting for a significant fraction of that load. Traditional fixed-timer and threshold-based controllers are energy-inefficient because they react to temperature thresholds rather than learning the underlying workload dynamics.

ThermalRL trains a DQN agent to control cooling setpoints in a simulated single-zone data center, minimising Power Usage Effectiveness (PUE) while keeping server temperatures within safe operating limits. The agent learns to run warmer than a naive baseline — using just enough cooling — rather than over-cooling defensively.

---

## Results

Evaluated over 500 episodes under identical stochastic conditions (seed fixed to 42).

| Metric | Fixed-Timer Baseline | ThermalRL (DQN v10) | Δ |
|---|---|---|---|
| Average Reward | −250.82 | −235.83 | **+6.0%** |
| Average PUE | 1.19 | 1.16 | **−2.5%** |
| Average Server Temp (°C) | 27.17 | 30.42 | +3.25°C |
| Temp Violations (%) | 0.00% | 0.09% | ~0 |

The DQN operates at a higher temperature band (28–32°C vs the baseline's 26–30°C), which is the mechanism behind the PUE gain — less cooling power per step while remaining within safe limits. The 0.09% violation rate represents approximately 9 timesteps out of 100,000.

At 1 MW IT load, the 2.5% PUE reduction corresponds to ~30 kW less cooling overhead continuously. At India's grid carbon intensity (~0.71 kg CO₂/kWh), this avoids roughly **186 tonnes of CO₂ per year per MW of IT load**, directly supporting SDG 13.

---

## Project Structure

```
ThermalRL/
├── sim/
│   └── datacenter_env.py       # Custom Gym-style environment (NumPy only)
├── configs/
│   ├── dqn_v1.yaml             # Baseline hyperparameters
│   ├── dqn_v2.yaml             # Slower epsilon decay variant
│   └── dqn_v10.yaml            # Final production config
├── experiments/
│   ├── results_v1.csv          # Per-episode metrics, all runs
│   └── results_v10.csv
├── policies/
│   ├── policy_v1.pkl           # Saved after exp-dqn-1
│   └── policy_v10.pkl          # Final policy (best result)
├── plots/
│   ├── training_curve.png      # Reward convergence over episodes
│   └── temp_comparison.png     # Baseline vs DQN temperature trace
├── train.py                    # Training script
├── evaluate.py                 # Evaluation + comparison table + plot
├── plot_results.py             # Training curve plotter
└── pyproject.toml
```

---

## Setup

Requires Python 3.10+. Uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/<your-username>/ThermalRL.git
cd ThermalRL
uv sync
```

Run commands with `uv run` (no manual activation needed), or activate the environment:

```bash
source .venv/bin/activate
```

**Dependencies:** `torch` (CPU), `numpy`, `matplotlib`, `pandas`, `pyyaml`

No GPU required. Full training completes in under 10 minutes on a standard laptop CPU.

---

## Reproducing the Final Result

To reproduce the exact numbers in the results table:

```bash
# Train the final policy
uv run python train.py --config configs/dqn_v10.yaml

# Evaluate against baseline over 500 episodes
uv run python evaluate.py --model policies/policy_v10.pkl --config configs/dqn_v10.yaml --episodes 500

# Plot training convergence
uv run python plot_results.py --csv experiments/results_v10.csv --output plots/training_curve.png
```

Seeds are locked (`numpy.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`) so results are deterministic across machines.

---

## Environment

`sim/datacenter_env.py` implements a lightweight single-zone thermal model with no external dependencies.

| Component | Detail |
|---|---|
| **State** | `[server_temp, cooling_load, external_temp]` — normalised to [0, 1] |
| **Action** | 5 discrete cooling levels (0 = min, 4 = max) |
| **Reward** | `−PUE − overcool_penalty` where overcool_penalty activates below 25°C |
| **Episode length** | 200 timesteps |
| **Thermal dynamics** | Heat from workload + ambient transfer − cooling capacity |

The overcooling floor (penalty below 25°C) is physically motivated — real data centres enforce cold aisle minimums to prevent condensation damage. This constraint drives the agent into the efficient 27–32°C operating band without explicit temperature targeting.

---

## Agent Architecture

| Component | Detail |
|---|---|
| **Algorithm** | Deep Q-Network (DQN) |
| **Network** | Input(3) → Linear(64) → ReLU → Linear(64) → ReLU → Output(5) |
| **Replay buffer** | Capacity 10,000, batch size 64 |
| **Target network** | Hard update every 100 steps |
| **Optimiser** | Adam, lr = 0.001 |
| **Discount factor (γ)** | 0.99 |
| **Exploration** | ε-greedy, ε: 1.0 → 0.05, decay 0.995 |
| **Training episodes** | 600 |

---

## Experiment Tracking

Every training run appends to a CSV in `experiments/` and is tagged in Git.

```
run_id, episode, reward, avg_pue, epsilon
run_v10, 1, -287.43, 1.31, 0.995
run_v10, 2, -271.18, 1.28, 0.990
...
```

Git tags mark each experiment version:

```bash
git tag exp-dqn-v1     # Initial run, reward hacking observed
git tag exp-dqn-v4     # Band penalty introduced
git tag exp-dqn-v9     # Physical floor constraint
git tag exp-dqn-v10-final  # Best result, beats baseline
```

---

## Reward Shaping History

Achieving a policy that beats the baseline required 10 reward iterations. The progression is documented here as it illustrates key RL engineering challenges.

| Version | Reward Function | Outcome |
|---|---|---|
| v1 | `−PUE` only | Reward hacking — agent overcooled to 16°C |
| v4 | `−PUE − band_penalty` | Overcooling resolved, undercooling oscillations appeared |
| v6 | Asymmetric band penalty | Improved, still 5.87% violations |
| v7 | Exponential gradient above 28°C | Agent oscillated wildly — too steep for thermal inertia |
| v8 | Directional trend reward | Reverted to overcooling — trend signal too weak |
| v9 | Physical floor at 22°C | Stable oscillation, agent still 3°C too cold |
| **v10** | **Physical floor at 25°C, rate 1.0** | **Beats baseline — PUE 1.16 vs 1.19** |

Key lesson: physically motivated constraints (condensation floor) outperformed mathematically constructed shaping terms because they align with the environment's thermal dynamics rather than fighting them.

---

## Monitoring Plan

If deployed in a real data centre, the following metrics would be tracked continuously:

- **Average PUE** (15-minute rolling window) — alert if > 1.6
- **Maximum server temperature** across all racks — alert if > 34°C
- **Reward trend** over a 24-hour rolling window — flag policy drift if declining > 5%
- **Action distribution** — alert if agent locks into a single cooling level for > 30 consecutive steps (indicates degenerate policy)
- **Violation rate** — alert if > 0.5% over any 1-hour window

---

## Limitations

- **Single-zone model** — real data centres have dozens of independent thermal zones
- **No airflow physics** — CFD and hot-aisle/cold-aisle dynamics are omitted
- **Linear thermal approximation** — heat transfer is simplified; no thermal inertia modelling
- **Synthetic workload** — stochastic load changes are mild; real traffic has diurnal patterns and sudden spikes
- **DQN brittleness** — results are sensitive to reward design and epsilon schedule; small changes can cause catastrophic forgetting
- **Simulation only** — results should not be interpreted as directly transferable to production infrastructure

---

## Future Work

- Multi-zone thermal modelling with inter-zone heat transfer
- Continuous action space with DDPG or SAC
- Carbon-aware reward incorporating real-time grid carbon intensity
- Adaptive reward shaping based on observed workload distribution
- Dynamic workload scheduling co-optimised with cooling

---

## SDG 13 — Climate Action

> *"Reducing PUE from 1.19 to 1.16 (2.5% improvement) in a data centre consuming 1 MW of IT load eliminates approximately 30 kW of continuous cooling overhead. At India's average grid carbon intensity of 0.71 kg CO₂/kWh, this avoids roughly 186 tonnes of CO₂ per year — without any hardware changes, purely through learned control policy."*

ThermalRL demonstrates that reinforcement learning can discover energy-aware operational strategies that fixed-rule systems cannot, providing a scalable path toward lower-carbon compute infrastructure.

---

## License

MIT