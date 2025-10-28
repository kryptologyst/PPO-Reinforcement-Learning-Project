"""Unit tests for PPO agent."""

import pytest
import torch
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

from src.agents import PPOAgent, PPOActorCritic, BaseAgent
from src.utils import load_config, Logger, CheckpointManager


class TestPPOActorCritic:
    """Test cases for PPOActorCritic network."""
    
    def test_discrete_action_space(self):
        """Test network with discrete action space."""
        network = PPOActorCritic(
            state_dim=4,
            action_dim=2,
            hidden_sizes=[64, 64],
            continuous=False
        )
        
        state = torch.randn(1, 4)
        probs, value, log_std = network(state)
        
        assert probs.shape == (1, 2)
        assert value.shape == (1, 1)
        assert log_std is None
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
    
    def test_continuous_action_space(self):
        """Test network with continuous action space."""
        network = PPOActorCritic(
            state_dim=3,
            action_dim=1,
            hidden_sizes=[64],
            continuous=True
        )
        
        state = torch.randn(1, 3)
        mean, value, log_std = network(state)
        
        assert mean.shape == (1, 1)
        assert value.shape == (1, 1)
        assert log_std.shape == (1, 1)
    
    def test_batch_processing(self):
        """Test network with batch input."""
        network = PPOActorCritic(
            state_dim=4,
            action_dim=2,
            hidden_sizes=[64],
            continuous=False
        )
        
        batch_size = 32
        state = torch.randn(batch_size, 4)
        probs, value, log_std = network(state)
        
        assert probs.shape == (batch_size, 2)
        assert value.shape == (batch_size, 1)
        assert log_std is None


class TestPPOAgent:
    """Test cases for PPOAgent."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        return gym.make("CartPole-v1")
    
    @pytest.fixture
    def agent(self, env):
        """Create test agent."""
        return PPOAgent(
            env=env,
            learning_rate=1e-3,
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
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.device == torch.device("cpu")
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert not agent.continuous
        assert agent.training_step == 0
    
    def test_action_selection(self, agent):
        """Test action selection."""
        state = np.random.randn(4)
        
        # Test stochastic action selection
        action, log_prob = agent.select_action(state, deterministic=False)
        assert isinstance(action, int)
        assert action in [0, 1]
        assert isinstance(log_prob, torch.Tensor)
        
        # Test deterministic action selection
        action_det, log_prob_det = agent.select_action(state, deterministic=True)
        assert isinstance(action_det, int)
        assert action_det in [0, 1]
        assert isinstance(log_prob_det, torch.Tensor)
    
    def test_trajectory_collection(self, agent):
        """Test trajectory collection."""
        batch = agent.collect_trajectory(n_steps=100)
        
        assert "states" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "dones" in batch
        assert "log_probs" in batch
        assert "values" in batch
        assert "returns" in batch
        assert "advantages" in batch
        
        assert batch["states"].shape[0] == 100
        assert batch["actions"].shape[0] == 100
        assert batch["rewards"].shape[0] == 100
        assert batch["dones"].shape[0] == 100
    
    def test_gae_computation(self, agent):
        """Test GAE computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        dones = torch.tensor([False, False, False, True])
        
        returns, advantages = agent._compute_gae(rewards, values, dones)
        
        assert returns.shape == rewards.shape
        assert advantages.shape == rewards.shape
        assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    
    def test_agent_update(self, agent):
        """Test agent update."""
        # Collect trajectory
        batch = agent.collect_trajectory(n_steps=64)
        
        # Update agent
        metrics = agent.update(batch)
        
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert "training_step" in metrics
        
        assert metrics["training_step"] == 1
        assert isinstance(metrics["total_loss"], float)
    
    def test_agent_evaluation(self, agent):
        """Test agent evaluation."""
        eval_metrics = agent.evaluate(n_episodes=5)
        
        assert "mean_reward" in eval_metrics
        assert "std_reward" in eval_metrics
        assert "mean_length" in eval_metrics
        assert "std_length" in eval_metrics
        
        assert isinstance(eval_metrics["mean_reward"], float)
        assert isinstance(eval_metrics["std_reward"], float)
    
    def test_save_load(self, agent, tmp_path):
        """Test agent save and load."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Save agent
        agent.save(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # Create new agent and load
        new_env = gym.make("CartPole-v1")
        new_agent = PPOAgent(
            env=new_env,
            learning_rate=1e-3,
            device="cpu",
            seed=42
        )
        
        new_agent.load(str(checkpoint_path))
        
        # Test that loaded agent works
        state = np.random.randn(4)
        action, _ = new_agent.select_action(state)
        assert isinstance(action, int)
        
        new_env.close()


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_config_loading(self, tmp_path):
        """Test configuration loading."""
        config_data = {
            "ppo": {
                "episodes": 1000,
                "learning_rate": 3e-4
            }
        }
        
        config_path = tmp_path / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_path))
        assert loaded_config["ppo"]["episodes"] == 1000
        assert loaded_config["ppo"]["learning_rate"] == 3e-4
    
    def test_logger_initialization(self, tmp_path):
        """Test logger initialization."""
        logger = Logger(
            log_dir=str(tmp_path),
            use_tensorboard=False,
            use_wandb=False
        )
        
        assert logger.log_dir == tmp_path
        assert not logger.use_tensorboard
        assert not logger.use_wandb
        
        logger.close()
    
    def test_checkpoint_manager(self, tmp_path):
        """Test checkpoint manager."""
        manager = CheckpointManager(
            save_dir=str(tmp_path),
            keep_last=3,
            save_frequency=10
        )
        
        assert manager.save_dir == tmp_path
        assert manager.keep_last == 3
        assert manager.save_frequency == 10
        
        # Test checkpoint listing
        checkpoints = manager.list_checkpoints()
        assert isinstance(checkpoints, list)
        
        # Test latest checkpoint
        latest = manager.get_latest_checkpoint()
        assert latest is None or isinstance(latest, str)


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""
    
    def test_base_agent_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        env = gym.make("CartPole-v1")
        
        with pytest.raises(TypeError):
            BaseAgent(env=env)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
