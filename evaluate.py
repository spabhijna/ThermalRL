import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sim.datacenter_env import DataCenterEnv

# -----------------------------------------
# Required DQN Architecture for weights loading
# -----------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=3, output_dim=5):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------
# Baseline Controller Logic
# -----------------------------------------
def baseline_policy(server_temp):
    """
    Rule-based comparison controller.
    Reactively cools based on explicit thresholds.
    """
    if server_temp > 32.0:
        return 4  # Maximum cooling
    elif server_temp < 28.0:
        return 0  # Minimum cooling
    else:
        return 2  # Medium cooling

# -----------------------------------------
# Evaluation Runner
# -----------------------------------------
def evaluate_policy(env, model=None, is_baseline=False, num_episodes=10):
    metrics = {
        'rewards': [],
        'pues': [],
        'temps': [],
        'violations': 0,
        'total_steps': 0,
        'temp_history': []
    }
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_temp_history = []
        ep_reward = 0.0
        
        while not done:
            # We bypass normalization strictly to read the actual physical value 
            # for baseline conditionals and accurate logging.
            actual_temp = env.server_temp
            
            # Action Selection
            if is_baseline:
                action = baseline_policy(actual_temp)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_t)
                    action = q_values.argmax(dim=1).item()
                    
            next_state, reward, done = env.step(action)
            
            # Post-action environment state
            actual_temp_after = env.server_temp
            ep_temp_history.append(actual_temp_after)
            metrics['temps'].append(actual_temp_after)
            
            if actual_temp_after > 35.0:
                metrics['violations'] += 1
                
            ep_reward += reward
            
            # PUE Re-calculation (robust to config)
            actual_load = state[1] * 100.0
            step_pue = (actual_load + (action * 15.0)) / actual_load
            metrics['pues'].append(step_pue)
            
            metrics['total_steps'] += 1
            state = next_state
            
        metrics['rewards'].append(ep_reward)
        
        # Save the temperature trace of the very first episode for plotting
        if ep == 0:
            metrics['temp_history'] = ep_temp_history
            
    return metrics

# -----------------------------------------
# Main Execution & Reporting
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Controllers")
    parser.add_argument('--model', type=str, default='policies/policy_v1.pkl', help='Trained DQN policy')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes to evaluate')
    parser.add_argument('--config', type=str, default='configs/dqn_v1.yaml', help='YAML config to sync reward functions')
    args = parser.parse_args()
    
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    env = DataCenterEnv(config)
    
    # Initialize and load DQN
    device = torch.device("cpu")
    dqn_model = DQN().to(device)
    if os.path.exists(args.model):
        dqn_model.load_state_dict(torch.load(args.model, map_location=device))
        dqn_model.eval()
    else:
        print(f"Warning: {args.model} missing. Using untrained weights.")
        
    print(f"Running deterministic evaluations over {args.episodes} episodes...")
    
    # Evaluate Baseline
    np.random.seed(42)  # Lock RNG to enforce identical load & external temp conditions
    base_metrics = evaluate_policy(env, is_baseline=True, num_episodes=args.episodes)
    
    # Evaluate DQN
    np.random.seed(42)  # Reset RNG so DQN faces the exact same sequence
    dqn_metrics = evaluate_policy(env, model=dqn_model, is_baseline=False, num_episodes=args.episodes)
    
    # Data aggregation helper
    def aggregate(m):
        return {
            'avg_reward': np.mean(m['rewards']),
            'avg_pue': np.mean(m['pues']),
            'avg_temp': np.mean(m['temps']),
            'viol_pct': (m['violations'] / m['total_steps']) * 100
        }
        
    base_agg = aggregate(base_metrics)
    dqn_agg = aggregate(dqn_metrics)
    
    # Output Requested Table Format
    print("\n### Evaluation Results ###\n")
    print("| Metric | Baseline | DQN |")
    print("|---|---|---|")
    print(f"| Average Reward | {base_agg['avg_reward']:.2f} | {dqn_agg['avg_reward']:.2f} |")
    print(f"| Average PUE | {base_agg['avg_pue']:.2f} | {dqn_agg['avg_pue']:.2f} |")
    print(f"| Average Temp (°C) | {base_agg['avg_temp']:.2f} | {dqn_agg['avg_temp']:.2f} |")
    print(f"| Temp Violation (%) | {base_agg['viol_pct']:.2f}% | {dqn_agg['viol_pct']:.2f}% |")
    print("\n")
    
    # Generate Overlay Plot
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/temp_comparison.png'
    
    plt.figure(figsize=(12, 6))
    plt.plot(base_metrics['temp_history'], label='Baseline Controller', color='crimson', linewidth=2.5, alpha=0.75)
    plt.plot(dqn_metrics['temp_history'], label='DQN Agent', color='royalblue', linewidth=2.5, alpha=0.75)
    
    # Adding semantic threshold lines
    plt.axhline(y=35.0, color='black', linestyle='--', linewidth=2, label='Violation Penalty Threshold (>35°C)')
    plt.axhline(y=32.0, color='darkorange', linestyle=':', label='Baseline Max Trigger (32°C)')
    plt.axhline(y=28.0, color='forestgreen', linestyle=':', label='Baseline Min Trigger (28°C)')
    
    plt.title('Baseline vs DQN Server Temperature Control (Identical Test Conditions)', fontsize=15, fontweight='bold')
    plt.xlabel('Simulation Timestep', fontsize=12)
    plt.ylabel('Server Temperature (°C)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    
    print(f"Comparison plot successfully saved to {plot_path}")

if __name__ == "__main__":
    main()
