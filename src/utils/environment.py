import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class HealthInterventionEnv(gym.Env):
    def __init__(self, user_data, similar_users_data=None):
        super(HealthInterventionEnv, self).__init__()
        
        # Action space: 0=No action, 1=Walk, 2=Meditate, 3=Sleep early
        self.action_space = spaces.Discrete(4)
        
        # State space: [steps, heart_rate, sleep_quality, day_of_week, fatigue]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.user_data = user_data
        self.similar_users_data = similar_users_data or []
        self.current_day = 0
        self.max_days = min(30, len(user_data))
        
        # Initial state
        self.state = self._get_initial_state()
        
    def _get_initial_state(self):
        """Get initial state from user data"""
        if self.current_day < len(self.user_data):
            day_data = self.user_data.iloc[self.current_day]
            return np.array([
                day_data.get('TotalSteps', 5000),
                70,  # Default heart rate
                0.8,  # Default sleep quality
                self.current_day % 7,  # Day of week
                0     # Fatigue level
            ], dtype=np.float32)
        else:
            return np.array([5000, 70, 0.8, 0, 0], dtype=np.float32)
    
    def step(self, action):
        """Execute one time step within the environment"""
        current_state = self.state.copy()
        
        # Apply action effects
        reward = self._calculate_reward(current_state, action)
        
        # State transition (simplified for demo)
        self.current_day += 1
        next_state = self._get_next_state(current_state, action)
        self.state = next_state
        
        # Check if episode is done - GYMNASIUM USES TWO FLAGS
        terminated = self.current_day >= self.max_days
        truncated = False  # We don't use time limits, so always False
        
        return next_state, reward, terminated, truncated, {}  # ← 5 returns!
    
    def _calculate_reward(self, state, action):
        """Calculate reward based on state and action"""
        reward = 0
        
        # Base rewards for good health metrics
        steps, heart_rate, sleep_quality, day_of_week, fatigue = state
        
        if steps > 8000:  # Step goal achieved
            reward += 1.0
        if sleep_quality > 0.8:  # Good sleep
            reward += 1.0
        if heart_rate < 80:  # Healthy heart rate
            reward += 0.5
            
        # Action-specific rewards/penalties
        if action == 1:  # Walk recommendation
            if steps < 6000:  # If user needs more steps
                reward += 0.5
            else:
                reward -= 0.3  # Don't over-recommend
                
        elif action == 3:  # Sleep early
            if sleep_quality < 0.7:  # If poor sleep
                reward += 0.5
            else:
                reward -= 0.3
                
        elif action == 0:  # No action
            if steps < 4000 or sleep_quality < 0.6:  # Should have recommended
                reward -= 0.2
                
        # Penalize fatigue
        reward -= fatigue * 0.1
        
        return reward
    
    def _get_next_state(self, current_state, action):
        """Simulate next state based on action"""
        steps, heart_rate, sleep_quality, day_of_week, fatigue = current_state
        
        # Action effects
        if action == 1:  # Walk - increases steps, slightly increases heart rate
            steps = min(steps + 2000, 20000)
            heart_rate = min(heart_rate + 5, 120)
            fatigue += 0.1
            
        elif action == 2:  # Meditate - lowers heart rate
            heart_rate = max(heart_rate - 10, 60)
            sleep_quality = min(sleep_quality + 0.1, 1.0)
            
        elif action == 3:  # Sleep early - improves sleep quality
            sleep_quality = min(sleep_quality + 0.15, 1.0)
            fatigue = max(fatigue - 0.2, 0)
            
        # Natural daily variation
        steps = max(steps + random.randint(-1000, 1000), 0)
        sleep_quality = max(sleep_quality + random.uniform(-0.1, 0.1), 0.3)
        
        return np.array([steps, heart_rate, sleep_quality, (day_of_week + 1) % 7, fatigue])
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state - GYMNASIUM VERSION"""
        super().reset(seed=seed)  # Important for reproducibility
        self.current_day = 0
        self.state = self._get_initial_state()
        return self.state, {}  # ← Returns tuple with info dict!