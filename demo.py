#!/usr/bin/env python3
"""Simple demo script for PPO training."""

import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

def main():
    """Run a simple PPO training demo."""
    print("🤖 PPO Reinforcement Learning Demo")
    print("=" * 40)
    
    try:
        # Import modules
        from agents import PPOAgent
        print("✅ Imports successful")
        
        # Create environment
        env = gym.make("CartPole-v1")
        print(f"✅ Environment: {env.spec.id}")
        
        # Create agent
        agent = PPOAgent(
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            ppo_epochs=4,
            batch_size=64,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            hidden_sizes=[64, 64],
            activation="relu",
            device="cpu",
            seed=42
        )
        print(f"✅ Agent created on device: {agent.device}")
        
        # Training parameters
        episodes = 100
        batch_steps = 512
        eval_frequency = 20
        
        print(f"\n🚀 Starting training for {episodes} episodes")
        print(f"Batch steps: {batch_steps}, Eval frequency: {eval_frequency}")
        
        episode_rewards = []
        start_time = time.time()
        
        for episode in range(episodes):
            # Collect trajectory
            batch = agent.collect_trajectory(batch_steps)
            
            # Update agent
            metrics = agent.update(batch)
            
            # Evaluate agent
            if episode % eval_frequency == 0:
                eval_metrics = agent.evaluate(n_episodes=5)
                episode_rewards.append(eval_metrics['mean_reward'])
                
                print(f"Episode {episode:3d} | Reward: {eval_metrics['mean_reward']:6.2f} ± {eval_metrics['std_reward']:5.2f} | Loss: {metrics['total_loss']:6.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_eval = agent.evaluate(n_episodes=10)
        
        print(f"\n🎉 Training completed in {training_time:.2f} seconds")
        print(f"Final reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        
        # Plot results
        if episode_rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(episode_rewards, linewidth=2)
            plt.title("PPO Training Progress")
            plt.xlabel("Evaluation Episode")
            plt.ylabel("Mean Reward")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("training_progress.png", dpi=300, bbox_inches="tight")
            print("📊 Training plot saved as 'training_progress.png'")
        
        # Test trained agent
        print("\n🎮 Testing trained agent:")
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(500):  # Max steps
            action, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"Test episode: Reward = {total_reward:.2f}, Steps = {steps}")
        
        env.close()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
