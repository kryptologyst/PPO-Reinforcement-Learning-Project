"""Proximal Policy Optimization (PPO) agent implementation."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from gymnasium import Env

from .base import BaseAgent


class PPOActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        continuous: bool = False,
    ) -> None:
        """
        Initialize the Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("relu", "tanh", "elu")
            continuous: Whether action space is continuous
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Build shared layers
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(activation)
            ])
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(prev_size, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(prev_size, action_dim)
        
        # Critic head
        self.critic = nn.Linear(prev_size, 1)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action distribution, value estimate, and log standard deviation (for continuous)
        """
        shared_features = self.shared(state)
        
        # Actor
        if self.continuous:
            mean = self.actor_mean(shared_features)
            log_std = self.actor_log_std.expand_as(mean)
            return mean, self.critic(shared_features), log_std
        else:
            action_probs = torch.softmax(self.actor(shared_features), dim=-1)
            return action_probs, self.critic(shared_features), None


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        device: Union[str, torch.device] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize PPO agent.
        
        Args:
            env: Environment to interact with
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            ppo_epochs: Number of PPO update epochs
            batch_size: Batch size for updates
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            device: Device to run on
            seed: Random seed
        """
        super().__init__(env, device, seed)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Environment info
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        self.continuous = not hasattr(env.action_space, 'n')
        
        # Network
        self.network = PPOActorCritic(
            self.state_dim,
            self.action_dim,
            hidden_sizes,
            activation,
            self.continuous,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training state
        self.training_step = 0
    
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
            Action and log probability
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous:
                mean, value, log_std = self.network(state)
                dist = Normal(mean, log_std.exp())
                
                if deterministic:
                    action = mean
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action).sum(dim=-1)
                return action.cpu().numpy().flatten(), log_prob
            else:
                probs, value, _ = self.network(state)
                dist = Categorical(probs)
                
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action)
                return action.item(), log_prob
    
    def collect_trajectory(
        self, 
        n_steps: int
    ) -> Dict[str, torch.Tensor]:
        """
        Collect trajectory data.
        
        Args:
            n_steps: Number of steps to collect
            
        Returns:
            Dictionary containing trajectory data
        """
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        
        state, _ = self.env.reset()
        
        for _ in range(n_steps):
            # Select action
            action, log_prob = self.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            
            # Get value estimate
            with torch.no_grad():
                if self.continuous:
                    _, value, _ = self.network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                else:
                    _, value, _ = self.network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                values.append(value.item())
            
            state = next_state
            
            if done:
                state, _ = self.env.reset()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device) if self.continuous else torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": log_probs,
            "values": values,
            "returns": returns,
            "advantages": advantages,
        }
    
    def _compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            
        Returns:
            Returns and advantages
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute returns
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G * (1 - dones[i].float())
            returns[i] = G
        
        # Compute advantages using GAE
        A = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (values[i + 1] if i + 1 < len(values) else 0) * (1 - dones[i].float()) - values[i]
            A = delta + self.gamma * self.gae_lambda * A * (1 - dones[i].float())
            advantages[i] = A
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's parameters using PPO.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        
        # PPO updates
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                if self.continuous:
                    mean, values, log_std = self.network(batch_states)
                    dist = Normal(mean, log_std.exp())
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    probs, values, _ = self.network(batch_states)
                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                
                # Compute losses
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss_batch = -torch.min(surr1, surr2).mean()
                
                value_loss_batch = nn.MSELoss()(values.squeeze(), batch_returns)
                
                total_loss_batch = (
                    policy_loss_batch + 
                    self.value_coef * value_loss_batch - 
                    self.entropy_coef * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy_loss += entropy.item()
        
        self.training_step += 1
        
        return {
            "total_loss": total_loss / self.ppo_epochs,
            "policy_loss": policy_loss / self.ppo_epochs,
            "value_loss": value_loss / self.ppo_epochs,
            "entropy_loss": entropy_loss / self.ppo_epochs,
            "training_step": self.training_step,
        }
    
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "ppo_epochs": self.ppo_epochs,
                "batch_size": self.batch_size,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "max_grad_norm": self.max_grad_norm,
            }
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
