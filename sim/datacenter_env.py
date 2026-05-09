import numpy as np

class DataCenterEnv:
    """
    Custom lightweight Data Center Environment for cooling simulation.
    Built entirely without Gym/Gymnasium.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.reward_type = self.config.get('reward_type', 'v1')
        
        # Episode length constraint
        self.max_steps = 200
        self.current_step = 0
        
        # State variables
        self.server_temp = 25.0
        self.cooling_load = 50.0
        self.external_temp = 20.0
        
    def _get_normalized_state(self):
        """
        State must be: [server_temp, cooling_load, external_temp]
        Normalized to [0, 1] assuming practical maximums.
        """
        norm_server_temp = np.clip(self.server_temp / 60.0, 0.0, 1.0)
        norm_cooling_load = np.clip(self.cooling_load / 100.0, 0.0, 1.0)
        norm_external_temp = np.clip(self.external_temp / 40.0, 0.0, 1.0)
        
        return np.array([norm_server_temp, norm_cooling_load, norm_external_temp], dtype=np.float32)

    def reset(self):
        """
        Resets the environment to an initial state.
        Returns the normalized initial state.
        """
        self.current_step = 0
        
        # Initialize with some random variance for diverse starting conditions
        self.server_temp = 25.0 + np.random.uniform(-2.0, 2.0)
        self.cooling_load = 50.0 + np.random.uniform(-10.0, 10.0)
        self.external_temp = 20.0 + np.random.uniform(-5.0, 5.0)
        
        return self._get_normalized_state()

    def step(self, action):
        """
        Takes an action and updates the environment.
        Action Space:
            0 -> minimum cooling
            1 -> low cooling
            2 -> medium cooling
            3 -> high cooling
            4 -> maximum cooling
        
        Returns: next_state, reward, done
        """
        self.current_step += 1
        
        # 1. External temperature fluctuates randomly
        self.external_temp += np.random.normal(0, 0.5)
        self.external_temp = np.clip(self.external_temp, 10.0, 40.0)
        
        # 2. Server workload (cooling_load) changes stochastically
        self.cooling_load += np.random.normal(0, 2.0)
        self.cooling_load = np.clip(self.cooling_load, 10.0, 100.0)  # min 10 to avoid div by zero
        
        # 3. Cooling action reduces server temperature
        action = np.clip(action, 0, 4)
        cooling_capacity = action * 2.5  # Temperature reduction impact per step
        
        # Server temperature dynamics
        heat_from_load = (self.cooling_load / 100.0) * 4.0  # Workload generates heat
        ext_heat_transfer = (self.external_temp - self.server_temp) * 0.05  # Ambient thermal influence
        
        self.server_temp += heat_from_load + ext_heat_transfer - cooling_capacity
        self.server_temp = np.clip(self.server_temp, 15.0, 60.0)
        
        # 4. Reward Calculation
        # Higher cooling increases power usage -> Higher PUE
        it_power = self.cooling_load
        cooling_power = action * 15.0  # Power consumed by cooling (scales with action)
        pue = (it_power + cooling_power) / it_power
        
        if self.reward_type == 'v2':
            reward = -pue - (0.05 * self.cooling_load)
        else:
            reward = -pue
        
        # Additional penalty if server_temp > 35
        if self.server_temp > 35.0:
            reward -= 10.0
            
        # Check termination
        done = self.current_step >= self.max_steps
        
        return self._get_normalized_state(), float(reward), done
