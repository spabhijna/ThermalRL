import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot DQN Training Convergence")
    parser.add_argument('--csv', type=str, default='experiments/results_1.csv', help='Path to the training CSV file')
    parser.add_argument('--output', type=str, default='plots/training_curve.png', help='Path to save the plot')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Could not find data file {args.csv}")
        return

    # Load training metrics
    df = pd.read_csv(args.csv)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards with lower opacity
    plt.plot(df['episode'], df['reward'], color='royalblue', alpha=0.3, label='Raw Reward')
    
    # Plot a rolling average to clearly visualize the stabilization/convergence
    # Uses a window of 20 episodes
    if len(df) >= 20:
        rolling_reward = df['reward'].rolling(window=20, min_periods=1).mean()
        plt.plot(df['episode'], rolling_reward, color='darkblue', linewidth=2.5, label='Rolling Average (20 eps)')

    # Formatting per requirements
    plt.title('DQN Training Convergence', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward (Negative PUE)', fontsize=12)
    
    # Enable grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    # Save to disk
    plt.savefig(args.output, dpi=300)
    print(f"Plot successfully saved to {args.output}")

if __name__ == '__main__':
    main()
