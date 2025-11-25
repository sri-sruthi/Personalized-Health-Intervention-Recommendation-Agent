import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import SAC  # ← Special version that supports discrete actions
import numpy as np
import pandas as pd
from utils.environment import HealthInterventionEnv
from utils.data_processor import DataProcessor

class FastRLTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        
    def create_demo_environment(self, user_id="1503960366"):
        """Create environment with demo data"""
        activity, sleep, heartrate = self.processor.load_data()
        
        # Use data from specific user + similar users
        user_data = activity[activity['Id'] == user_id].head(30)
        
        # Get similar users for narrowing RL
        self.processor.create_user_profiles(activity)
        similar_users = self.processor.get_similar_users(user_id, n_users=3)
        similar_users_data = []
        
        for similar_user in similar_users:
            similar_data = activity[activity['Id'] == similar_user].head(30)
            similar_users_data.append(similar_data)
            
        env = HealthInterventionEnv(user_data, similar_users_data)
        return env
    
    def train_ppo_fast(self, total_timesteps=10000):
        """Fast PPO training for demo"""
        print("Training PPO agent (fast)...")
        
        env = self.create_demo_environment()
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=32,
            n_epochs=5,
            gamma=0.99,
            verbose=1
        )
        
        model.learn(total_timesteps=total_timesteps)
        model.save("models/ppo_health_agent")
        return model, env
    
    def train_sac_fast(self, total_timesteps=10000):
        """Fast SAC training for demo - USING DISCRETE VERSION"""
        print("Training SAC agent (fast)...")
        
        env = self.create_demo_environment()
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Use SAC with discrete actions
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=32,
            gamma=0.99,
            use_sde=False,  # Important for discrete actions
            verbose=1
        )
        
        model.learn(total_timesteps=total_timesteps)
        model.save("models/sac_health_agent")
        return model, env

class HybridAgent:
    """Simple hybrid agent combining PPO and SAC"""
    def __init__(self, ppo_model, sac_model, alpha=0.7):
        self.ppo_model = ppo_model
        self.sac_model = sac_model
        self.alpha = alpha
        
    def predict(self, observation):
        """Hybrid prediction - sometimes use PPO, sometimes SAC"""
        if np.random.random() < self.alpha:
            action, _ = self.ppo_model.predict(observation, deterministic=True)
        else:
            action, _ = self.sac_model.predict(observation, deterministic=False)
        return action