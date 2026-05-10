import os
import argparse
import random
import csv
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .sim.datacenter_env import DataCenterEnv

# -----------------------------------------
# Neural Network Architecture
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
# Replay Buffer
# -----------------------------------------
class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

# -----------------------------------------
# Main Training Loop
# -----------------------------------------
def main():
    # Enforce Deterministic Execution for Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dqn_v1.yaml', help='Path to YAML config')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract Hyperparameters
    learning_rate = config.get('learning_rate', 0.001)
    gamma = config.get('gamma', 0.99)
    batch_size = config.get('batch_size', 64)
    episodes = config.get('episodes', 500)
    target_update_freq = config.get('target_update_freq', 100)
    
    eps_config = config.get('epsilon', {})
    epsilon = eps_config.get('start', 1.0)
    epsilon_min = eps_config.get('min', 0.05)
    epsilon_decay = eps_config.get('decay', 0.995)
    
    run_id = config.get('run_id', 'run_1')
    model_save_path = config.get('model_save_path', 'policies/policy_v1.pkl')
    csv_save_path = config.get('csv_save_path', 'experiments/results_1.csv')
    
    device = torch.device("cpu")
    env = DataCenterEnv(config)
    
    # Initialize Networks
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    memory = ReplayBuffer(maxlen=10000)
    
    # Prepare logging
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('policies', exist_ok=True)
    
    file_exists = os.path.isfile(csv_save_path)
    with open(csv_save_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['run_id', 'episode', 'reward', 'avg_pue', 'epsilon'])
        
    global_step = 0
    
    print(f"Starting DQN Training for {run_id} using {args.config}...")
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_pue = 0.0
        steps = 0
        done = False
        
        while not done:
            # Epsilon-greedy exploration strategy
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.argmax(dim=1).item()
                    
            next_state, reward, done = env.step(action)
            
            # Calculate true PUE locally so logging is immune to reward function changes
            actual_load = state[1] * 100.0  # Denormalize
            step_pue = (actual_load + (action * 15.0)) / actual_load
            episode_pue += step_pue
            
            # Store transition in replay buffer
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            global_step += 1
            
            # Train model if enough samples exist
            if len(memory) >= batch_size:
                s, a, r, s_prime, d = memory.sample(batch_size)
                
                s_t = torch.FloatTensor(s).to(device)
                a_t = torch.LongTensor(a).unsqueeze(1).to(device)
                r_t = torch.FloatTensor(r).unsqueeze(1).to(device)
                s_prime_t = torch.FloatTensor(s_prime).to(device)
                d_t = torch.FloatTensor(d).unsqueeze(1).to(device)
                
                # Current Q values
                q_values = policy_net(s_t).gather(1, a_t)
                
                # Target Q values
                with torch.no_grad():
                    max_next_q = target_net(s_prime_t).max(1)[0].unsqueeze(1)
                    target_q = r_t + (gamma * max_next_q * (1 - d_t))
                    
                loss = criterion(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Hard target network update
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        # Epsilon decay at the end of the episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        avg_pue = episode_pue / steps
        print(f"Episode: {episode:3d} | Reward: {episode_reward:7.2f} | Avg PUE: {avg_pue:5.2f} | Epsilon: {epsilon:.3f}")
        
        # Log episode metrics to CSV
        with open(csv_save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id, episode, episode_reward, avg_pue, epsilon])
            
    # Save the trained model
    torch.save(policy_net.state_dict(), model_save_path)
    print(f"Training complete! Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
