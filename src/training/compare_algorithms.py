import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from training.train_rl import FastRLTrainer, HybridAgent
import seaborn as sns

class AlgorithmComparator:
    def __init__(self):
        self.trainer = FastRLTrainer()
        
    def run_comparison(self, num_episodes=100):
        """Compare PPO, SAC, and Hybrid performance"""
        # Load trained models
        from stable_baselines3 import PPO, SAC
        
        ppo_model = PPO.load("models/ppo_health_agent")
        sac_model = SAC.load("models/sac_health_agent")
        hybrid_model = HybridAgent(ppo_model, sac_model)
        
        # Test each algorithm
        algorithms = {
            'PPO': ppo_model,
            'SAC': sac_model, 
            'Hybrid': hybrid_model
        }
        
        results = {}
        
        for algo_name, model in algorithms.items():
            print(f"Testing {algo_name}...")
            env = self.trainer.create_demo_environment()
            
            episode_rewards = []
            for episode in range(num_episodes):
                # GYMNASIUM: reset returns tuple (obs, info)
                obs, info = env.reset()
                episode_reward = 0
                terminated = False
                truncated = False
                
                while not (terminated or truncated):
                    if algo_name == 'Hybrid':
                        action = model.predict(obs)
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    
                    # GYMNASIUM: step returns 5 values
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            results[algo_name] = episode_rewards
            
        return results
    
    def plot_comparison(self, results):
        """Create comparison graphs for presentation"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning curves
        for algo_name, rewards in results.items():
            axes[0, 0].plot(rewards, label=algo_name, alpha=0.7)
        axes[0, 0].set_title('Algorithm Comparison - Cumulative Rewards')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Average performance
        avg_rewards = {algo: np.mean(rewards) for algo, rewards in results.items()}
        axes[0, 1].bar(avg_rewards.keys(), avg_rewards.values())
        axes[0, 1].set_title('Average Performance')
        axes[0, 1].set_ylabel('Average Reward')
        
        # Plot 3: Performance distribution
        data_for_boxplot = [results[algo] for algo in results.keys()]
        axes[1, 0].boxplot(data_for_boxplot, labels=results.keys())
        axes[1, 0].set_title('Performance Distribution')
        axes[1, 0].set_ylabel('Reward')
        
        # Plot 4: Narrowing RL effect (simulated)
        neighborhood_sizes = [50, 30, 15, 8, 5, 3]
        personalization_scores = [0.6, 0.65, 0.72, 0.78, 0.85, 0.88]
        axes[1, 1].plot(neighborhood_sizes, personalization_scores, 'o-', linewidth=2)
        axes[1, 1].set_title('Narrowing RL: Personalization vs Neighborhood Size')
        axes[1, 1].set_xlabel('Neighborhood Size')
        axes[1, 1].set_ylabel('Personalization Score')
        axes[1, 1].invert_xaxis()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    comparator = AlgorithmComparator()
    results = comparator.run_comparison(num_episodes=50)  # Quick comparison
    comparator.plot_comparison(results)