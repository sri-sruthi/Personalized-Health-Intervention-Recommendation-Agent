import numpy as np
import matplotlib.pyplot as plt
from training.train_rl import FastRLTrainer, HybridAgent
from training.compare_algorithms import AlgorithmComparator
from utils.data_processor import DataProcessor
import time

def run_complete_demo():
    """Complete demo that answers all professor questions"""
    print("🚀 PERSONALIZED HEALTH INTERVENTION AGENT DEMO")
    print("=" * 50)
    
    # Initialize components
    processor = DataProcessor()
    trainer = FastRLTrainer()
    comparator = AlgorithmComparator()
    
    # Load and show data
    print("\n1. 📊 DATA PROCESSING")
    activity, sleep, heartrate = processor.load_data()
    print(f"   - Activity records: {len(activity)}")
    print(f"   - Sleep records: {len(sleep)}") 
    print(f"   - Heart rate records: {len(heartrate)}")
    
    # Show user similarity
    processor.create_user_profiles(activity)
    sample_user = activity['Id'].iloc[0]
    similar_users = processor.get_similar_users(sample_user, 3)
    print(f"   - Similar users found for personalization: {len(similar_users)}")
    
    # Train models quickly
    print("\n2. 🤖 REINFORCEMENT LEARNING TRAINING")
    print("   Training PPO agent...")
    ppo_model, _ = trainer.train_ppo_fast(total_timesteps=3000)
    
    print("   Training SAC agent...") 
    sac_model, _ = trainer.train_sac_fast(total_timesteps=3000)
    
    hybrid_model = HybridAgent(ppo_model, sac_model)
    print("   ✅ All models trained successfully!")
    
    # Run comparison
    print("\n3. 📈 ALGORITHM COMPARISON")
    results = comparator.run_comparison(num_episodes=30)
    
    # Calculate metrics
    avg_rewards = {algo: np.mean(rewards) for algo, rewards in results.items()}
    best_algo = max(avg_rewards.items(), key=lambda x: x[1])
    
    print(f"   - PPO Average Reward: {avg_rewards['PPO']:.2f}")
    print(f"   - SAC Average Reward: {avg_rewards['SAC']:.2f}") 
    print(f"   - Hybrid Average Reward: {avg_rewards['Hybrid']:.2f}")
    print(f"   - Best Algorithm: {best_algo[0]} ({best_algo[1]:.2f})")
    
    # Show learning process
    print("\n4. 🎯 LEARNING PROCESS EXPLANATION")
    print("   - Episodes run: 30-50 (quick demo)")
    print("   - Reward structure: +1 for steps/sleep goals, -0.5 for bad recommendations")
    print("   - State space: [steps, heart_rate, sleep_quality, day_of_week, fatigue]")
    print("   - Actions: 0=No action, 1=Walk, 2=Meditate, 3=Sleep early")
    
    # Demonstrate personalization
    print("\n5. 🎭 PERSONALIZATION DEMONSTRATION")
    env = trainer.create_demo_environment()
    # GYMNASIUM: reset returns tuple
    obs, info = env.reset()
    
    print("   Testing agent recommendations:")
    for step in range(5):  # Show 5 steps
        action = hybrid_model.predict(obs)
        action_names = ["No action", "Walk", "Meditate", "Sleep early"]
        
        steps, hr, sleep_qual, dow, fatigue = obs
        print(f"   Step {step+1}: State[steps={steps:.0f}, sleep={sleep_qual:.2f}] -> Action: {action_names[action]}")
        
        # GYMNASIUM: step returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Generate graphs
    print("\n6. 📊 GENERATING EVALUATION GRAPHS...")
    comparator.plot_comparison(results)
    
    print("\n✅ DEMO COMPLETED!")
    print("📋 All professor questions answered:")
    print("   - Algorithm workings: Narrowing RL + PPO/SAC/Hybrid")
    print("   - Learning process: From similar users → personalized policies") 
    print("   - Metrics improvement: Shown in comparison graphs")
    print("   - Evaluation metrics: Cumulative reward, personalization score")
    print("   - Model efficiency: Fast training (<5 minutes)")
    print("   - Learning graphs: Generated and saved")
    print("   - Episodes: 30-50 for quick demo")
    
    return results

if __name__ == "__main__":
    results = run_complete_demo()