"""Base agent class for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from gymnasium import Env


class BaseAgent(ABC):
    """Abstract base class for reinforcement learning agents."""
    
    def __init__(
        self,
        env: Env,
        device: Union[str, torch.device] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the base agent.
        
        Args:
            env: The environment to interact with
            device: Device to run computations on ("auto", "cpu", "cuda", "mps")
            seed: Random seed for reproducibility
        """
        self.env = env
        self.device = self._get_device(device)
        self.seed = seed
        
        if seed is not None:
            self._set_seed(seed)
    
    def _get_device(self, device: Union[str, torch.device]) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if hasattr(self.env, "action_space"):
            self.env.action_space.seed(seed)
        if hasattr(self.env, "observation_space"):
            self.env.observation_space.seed(seed)
    
    @abstractmethod
    def select_action(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], Optional[torch.Tensor]]:
        """
        Select an action given a state.
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
            
        Returns:
            Action and optional log probability
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's parameters.
        
        Args:
            batch: Training batch containing states, actions, rewards, etc.
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        pass
    
    def evaluate(
        self, 
        env: Optional[Env] = None, 
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            env: Environment to evaluate on (uses self.env if None)
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_env = env or self.env
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }
