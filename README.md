# Lightweight Custom Simulator for Data Center Cooling

## Overview
A custom, lightweight simulator and Deep Q-Network (DQN) agent designed to minimize data center cooling energy consumption while maintaining safe operational temperatures. Built entirely from scratch using PyTorch and NumPy, avoiding heavy external RL frameworks.

## SDG 13 Relevance (Climate Action)
By optimizing the Power Usage Effectiveness (PUE) of data centers, this project directly supports **UN Sustainable Development Goal 13: Climate Action**. Data centers consume vast amounts of global electricity, primarily for cooling. Applying Reinforcement Learning to dynamically optimize cooling based on workload and external temperatures drastically reduces carbon footprints and operational energy waste.

## Setup Instructions
The project uses `uv` for blazing-fast, deterministic dependency management.

1. Ensure `uv` is installed on your system.
2. Clone the repository and initialize the virtual environment:
   ```bash
   uv sync
   source .venv/bin/activate
   ```

## Training Instructions
Hyperparameters are managed via YAML configurations in the `configs/` directory. 
To train the DQN agent, run:

```bash
uv run python train.py --config configs/dqn_v1.yaml
```

*Results append directly to CSV files located in the `experiments/` folder.*

## Evaluation Instructions
To evaluate a trained policy against a rule-based baseline under identical deterministic conditions, run:

```bash
uv run python evaluate.py --model policies/policy_v1.pkl --episodes 10
```

To plot the training convergence curve:
```bash
uv run python plot_results.py --csv experiments/results_1.csv --output plots/training_curve.png
```

## Results Discussion
The DQN agent demonstrates successful cost-minimization. In early episodes, the agent explores broadly, leading to inefficient PUE values and frequent temperature violation penalties. However, as epsilon decays and the policy converges, the agent proactively balances cooling power against stochastic workloads, eventually achieving a highly stabilized average PUE (~1.30) with near-zero temperature violations. The DQN visibly outperforms the aggressive zigzagging profile of the naive rule-based baseline controller.

## MLOps & Reproducibility
For strict experiment reproducibility, the framework locks standard seeds.

It is highly recommended to track your experiments using Git tags so you can easily revert or track specific policy versions. For example:
```bash
git tag exp-dqn-1
git tag exp-dqn-2
git push --tags
```

## Limitations
While the DQN successfully reduces PUE, this simulator abstracts several real-world complexities:
- **Simplified thermal dynamics**: Heat transfer is approximated linearly rather than using complex thermodynamic modeling.
- **No airflow simulation**: Computational Fluid Dynamics (CFD) and spatial air recirculation are entirely omitted.
- **Single-zone approximation**: The data center is modeled as a unified thermal mass, ignoring hot-aisle/cold-aisle isolation anomalies.
- **DQN instability risks**: Deep Q-Networks are notoriously brittle; changes in epsilon decay or learning rate can result in catastrophic forgetting or failure to converge.
