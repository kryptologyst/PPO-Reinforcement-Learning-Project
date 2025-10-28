"""Main training script for PPO agent."""

import argparse
import time
from pathlib import Path
from typing import Dict, Any
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# from rich.console import Console
# from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.agents import PPOAgent
from src.utils import load_config, Logger, CheckpointManager, create_training_summary, save_training_summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--env", type=str, help="Environment name (overrides config)")
    parser.add_argument("--episodes", type=int, help="Number of episodes (overrides config)")
    parser.add_argument("--device", type=str, help="Device to use (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to load")
    return parser.parse_args()


def create_environment(config: Dict[str, Any]) -> gym.Env:
    """Create and configure environment."""
    env_config = config["environment"]
    env = gym.make(
        env_config["name"],
        render_mode=env_config.get("render_mode"),
        max_episode_steps=env_config.get("max_episode_steps"),
    )
    return env


def create_agent(env: gym.Env, config: Dict[str, Any]) -> PPOAgent:
    """Create PPO agent from configuration."""
    ppo_config = config["ppo"]
    
    agent = PPOAgent(
        env=env,
        learning_rate=ppo_config["actor_lr"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_epsilon=ppo_config["clip_epsilon"],
        ppo_epochs=ppo_config["ppo_epochs"],
        entropy_coef=ppo_config["entropy_coef"],
        value_coef=ppo_config["value_coef"],
        hidden_sizes=ppo_config["hidden_sizes"],
        activation=ppo_config["activation"],
        device=config.get("device", "auto"),
        seed=config.get("seed"),
    )
    
    return agent


def train_agent(
    agent: PPOAgent,
    config: Dict[str, Any],
    logger: Logger,
    checkpoint_manager: CheckpointManager,
) -> Dict[str, float]:
    """Train the PPO agent."""
    ppo_config = config["ppo"]
    eval_config = config["evaluation"]
    
    episodes = ppo_config["episodes"]
    batch_steps = ppo_config["batch_steps"]
    eval_frequency = eval_config["eval_frequency"]
    eval_episodes = eval_config["eval_episodes"]
    
    episode_rewards = []
    training_metrics = []
    
    print(f"Starting training for {episodes} episodes")
    
    start_time = time.time()
    
    for episode in range(episodes):
            # Collect trajectory
            batch = agent.collect_trajectory(batch_steps)
            
            # Update agent
            metrics = agent.update(batch)
            training_metrics.append(metrics)
            
            # Evaluate agent
            if episode % eval_frequency == 0:
                eval_metrics = agent.evaluate(n_episodes=eval_episodes)
                episode_rewards.append(eval_metrics["mean_reward"])
                
                # Log metrics
                log_metrics = {
                    "eval/mean_reward": eval_metrics["mean_reward"],
                    "eval/std_reward": eval_metrics["std_reward"],
                    "eval/mean_length": eval_metrics["mean_length"],
                    "train/total_loss": metrics["total_loss"],
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy_loss": metrics["entropy_loss"],
                }
                logger.log_scalars(log_metrics, episode)
                
                print(
                    f"Episode {episode:4d} | "
                    f"Reward: {eval_metrics['mean_reward']:6.2f} ± {eval_metrics['std_reward']:5.2f} | "
                    f"Loss: {metrics['total_loss']:6.4f}"
                )
            
            # Save checkpoint
            if episode % config["checkpoints"]["save_frequency"] == 0:
                checkpoint_manager.save_checkpoint(
                    agent, episode, metrics, {"episode_rewards": episode_rewards}
                )
            
    training_time = time.time() - start_time
    
    # Final evaluation
    final_eval = agent.evaluate(n_episodes=eval_episodes)
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    
    return {
        "final_reward": final_eval["mean_reward"],
        "final_std": final_eval["std_reward"],
        "training_time": training_time,
        "episode_rewards": episode_rewards,
    }


def plot_training_curves(episode_rewards: list, save_path: str) -> None:
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title("PPO Training Progress")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Mean Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Main training function."""
    args = parse_args()
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.env:
        config["environment"]["name"] = args.env
    if args.episodes:
        config["ppo"]["episodes"] = args.episodes
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed
    
    # Create environment
    env = create_environment(config)
    print(f"Environment: {config['environment']['name']}")
    
    # Create agent
    agent = create_agent(env, config)
    print(f"Device: {agent.device}")
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.load_checkpoint(agent, args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Setup logging
    logger = Logger(
        log_dir=config["logging"]["log_dir"],
        use_tensorboard=config["logging"]["tensorboard"],
        use_wandb=config["logging"]["wandb"]["enabled"],
        wandb_project=config["logging"]["wandb"]["project"],
        wandb_entity=config["logging"]["wandb"]["entity"],
        wandb_config=config,
    )
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=config["checkpoints"]["save_dir"],
        keep_last=config["checkpoints"]["keep_last"],
        save_frequency=config["checkpoints"]["save_frequency"],
    )
    
    if args.eval_only:
        # Evaluation only
        print("Running evaluation only")
        eval_metrics = agent.evaluate(n_episodes=config["evaluation"]["eval_episodes"])
        print(f"Evaluation results: {eval_metrics}")
    else:
        # Training
        final_metrics = train_agent(agent, config, logger, checkpoint_manager)
        
        # Save final checkpoint
        checkpoint_manager.save_checkpoint(agent, config["ppo"]["episodes"], final_metrics)
        
        # Plot training curves
        if final_metrics["episode_rewards"]:
            plot_path = Path(config["logging"]["log_dir"]) / "training_curves.png"
            plot_training_curves(final_metrics["episode_rewards"], str(plot_path))
            print(f"Training curves saved to: {plot_path}")
        
        # Save training summary
        summary = create_training_summary(config, final_metrics, final_metrics["training_time"])
        summary_path = Path(config["logging"]["log_dir"]) / "training_summary.json"
        save_training_summary(summary, str(summary_path))
        print(f"Training summary saved to: {summary_path}")
    
    # Cleanup
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
