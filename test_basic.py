#!/usr/bin/env python3
"""Simple test script to verify PPO implementation."""

import sys
from pathlib import Path
import gymnasium as gym
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

def test_basic_functionality():
    """Test basic PPO functionality."""
    print("Testing PPO implementation...")
    
    try:
        # Import modules
        from agents import PPOAgent
        from utils import load_config
        print("✅ Imports successful")
        
        # Create environment
        env = gym.make("CartPole-v1")
        print("✅ Environment created")
        
        # Create agent
        agent = PPOAgent(
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            ppo_epochs=2,
            batch_size=32,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            hidden_sizes=[64],
            activation="relu",
            device="cpu",
            seed=42
        )
        print("✅ Agent created")
        
        # Test action selection
        state = np.random.randn(4)
        action, log_prob = agent.select_action(state)
        print(f"✅ Action selection: {action}, log_prob: {log_prob.item():.4f}")
        
        # Test trajectory collection
        batch = agent.collect_trajectory(n_steps=64)
        print(f"✅ Trajectory collection: {len(batch['states'])} steps")
        
        # Test agent update
        metrics = agent.update(batch)
        print(f"✅ Agent update: loss = {metrics['total_loss']:.4f}")
        
        # Test evaluation
        eval_metrics = agent.evaluate(n_episodes=3)
        print(f"✅ Evaluation: reward = {eval_metrics['mean_reward']:.2f}")
        
        # Test configuration loading
        config = load_config("configs/default.yaml")
        print("✅ Configuration loaded")
        
        env.close()
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
