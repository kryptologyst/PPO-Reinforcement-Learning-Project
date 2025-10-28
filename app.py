"""Streamlit UI for PPO training and visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yaml
from pathlib import Path
import json
import time

from src.agents import PPOAgent
from src.utils import load_config, Logger, CheckpointManager
import gymnasium as gym


def load_training_data(log_dir: str) -> pd.DataFrame:
    """Load training data from logs."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return pd.DataFrame()
    
    # Look for training summary
    summary_path = log_path / "training_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        if "episode_rewards" in summary["final_metrics"]:
            rewards = summary["final_metrics"]["episode_rewards"]
            return pd.DataFrame({
                "episode": range(len(rewards)),
                "reward": rewards
            })
    
    return pd.DataFrame()


def create_reward_plot(df: pd.DataFrame) -> go.Figure:
    """Create reward plot."""
    if df.empty:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["episode"],
        y=df["reward"],
        mode="lines+markers",
        name="Episode Reward",
        line=dict(color="blue", width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Evaluation Episode",
        yaxis_title="Mean Reward",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


def create_metrics_plot(df: pd.DataFrame) -> go.Figure:
    """Create metrics plot."""
    if df.empty:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Loss", "Policy Loss", "Value Loss", "Entropy Loss"),
        vertical_spacing=0.1
    )
    
    # Add traces (placeholder - would need actual metrics data)
    fig.add_trace(go.Scatter(x=[], y=[], name="Total Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], name="Policy Loss"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], name="Value Loss"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], name="Entropy Loss"), row=2, col=2)
    
    fig.update_layout(
        title="Training Metrics",
        template="plotly_white",
        height=600
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="PPO Training Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 PPO Training Dashboard")
    st.markdown("Interactive dashboard for Proximal Policy Optimization training")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Environment selection
    env_options = [
        "CartPole-v1",
        "LunarLander-v2", 
        "MountainCar-v0",
        "Acrobot-v1",
        "Pendulum-v1"
    ]
    selected_env = st.sidebar.selectbox("Environment", env_options)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    episodes = st.sidebar.slider("Episodes", 100, 2000, 1000)
    batch_steps = st.sidebar.slider("Batch Steps", 512, 4096, 2048)
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
    gamma = st.sidebar.slider("Gamma", 0.9, 0.999, 0.99)
    clip_epsilon = st.sidebar.slider("Clip Epsilon", 0.1, 0.3, 0.2)
    
    # Log directory
    log_dir = st.sidebar.text_input("Log Directory", "logs")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training", "🎮 Environment", "⚙️ Configuration", "📈 Analysis"])
    
    with tab1:
        st.header("Training Progress")
        
        # Load and display training data
        training_data = load_training_data(log_dir)
        
        if not training_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Reward Progress")
                reward_plot = create_reward_plot(training_data)
                st.plotly_chart(reward_plot, use_container_width=True)
            
            with col2:
                st.subheader("Training Statistics")
                if not training_data.empty:
                    stats = {
                        "Final Reward": f"{training_data['reward'].iloc[-1]:.2f}",
                        "Best Reward": f"{training_data['reward'].max():.2f}",
                        "Mean Reward": f"{training_data['reward'].mean():.2f}",
                        "Std Reward": f"{training_data['reward'].std():.2f}",
                    }
                    
                    for key, value in stats.items():
                        st.metric(key, value)
        else:
            st.info("No training data found. Start training to see progress.")
        
        # Training controls
        st.subheader("Training Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training in progress..."):
                    # This would start actual training
                    st.success("Training started!")
        
        with col2:
            if st.button("⏸️ Pause Training"):
                st.info("Training paused")
        
        with col3:
            if st.button("🛑 Stop Training"):
                st.warning("Training stopped")
    
    with tab2:
        st.header("Environment Information")
        
        # Environment details
        env_info = {
            "CartPole-v1": {
                "description": "Balance a pole on a cart",
                "state_space": "4D continuous",
                "action_space": "2 discrete actions",
                "max_steps": 500
            },
            "LunarLander-v2": {
                "description": "Land a spacecraft on the moon",
                "state_space": "8D continuous", 
                "action_space": "4 discrete actions",
                "max_steps": 1000
            },
            "MountainCar-v0": {
                "description": "Drive a car up a mountain",
                "state_space": "2D continuous",
                "action_space": "3 discrete actions", 
                "max_steps": 200
            },
            "Acrobot-v1": {
                "description": "Swing up a double pendulum",
                "state_space": "6D continuous",
                "action_space": "3 discrete actions",
                "max_steps": 500
            },
            "Pendulum-v1": {
                "description": "Balance an inverted pendulum",
                "state_space": "3D continuous",
                "action_space": "1D continuous",
                "max_steps": 200
            }
        }
        
        info = env_info.get(selected_env, {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Details")
            st.write(f"**Description:** {info.get('description', 'N/A')}")
            st.write(f"**State Space:** {info.get('state_space', 'N/A')}")
            st.write(f"**Action Space:** {info.get('action_space', 'N/A')}")
            st.write(f"**Max Steps:** {info.get('max_steps', 'N/A')}")
        
        with col2:
            st.subheader("Environment Preview")
            # Placeholder for environment visualization
            st.info("Environment visualization would go here")
    
    with tab3:
        st.header("Configuration")
        
        # Display current configuration
        config = {
            "environment": {
                "name": selected_env,
                "render_mode": None,
                "max_episode_steps": info.get('max_steps', 500)
            },
            "ppo": {
                "episodes": episodes,
                "batch_steps": batch_steps,
                "actor_lr": learning_rate,
                "critic_lr": learning_rate,
                "gamma": gamma,
                "gae_lambda": 0.95,
                "clip_epsilon": clip_epsilon,
                "ppo_epochs": 10,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "hidden_sizes": [128, 128],
                "activation": "relu"
            },
            "logging": {
                "log_dir": log_dir,
                "tensorboard": True,
                "wandb": {"enabled": False}
            },
            "checkpoints": {
                "save_frequency": 100,
                "save_dir": "checkpoints",
                "keep_last": 5
            },
            "evaluation": {
                "eval_frequency": 50,
                "eval_episodes": 10
            }
        }
        
        st.subheader("Current Configuration")
        st.json(config)
        
        # Download configuration
        config_yaml = yaml.dump(config, default_flow_style=False)
        st.download_button(
            label="📥 Download Config",
            data=config_yaml,
            file_name="config.yaml",
            mime="text/yaml"
        )
    
    with tab4:
        st.header("Analysis")
        
        if not training_data.empty:
            st.subheader("Training Analysis")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Episodes Trained", len(training_data))
            
            with col2:
                improvement = training_data['reward'].iloc[-1] - training_data['reward'].iloc[0]
                st.metric("Total Improvement", f"{improvement:.2f}")
            
            with col3:
                convergence_episode = len(training_data) // 2
                st.metric("Convergence Episode", convergence_episode)
            
            with col4:
                stability = training_data['reward'].tail(10).std()
                st.metric("Recent Stability", f"{stability:.2f}")
            
            # Training curve analysis
            st.subheader("Training Curve Analysis")
            
            # Moving average
            window_size = min(10, len(training_data) // 4)
            if window_size > 1:
                training_data['moving_avg'] = training_data['reward'].rolling(window=window_size).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=training_data["episode"],
                    y=training_data["reward"],
                    mode="lines",
                    name="Episode Reward",
                    opacity=0.3,
                    line=dict(color="blue")
                ))
                fig.add_trace(go.Scatter(
                    x=training_data["episode"],
                    y=training_data["moving_avg"],
                    mode="lines",
                    name=f"Moving Average ({window_size})",
                    line=dict(color="red", width=3)
                ))
                
                fig.update_layout(
                    title="Training Progress with Moving Average",
                    xaxis_title="Episode",
                    yaxis_title="Reward",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training data available for analysis.")


if __name__ == "__main__":
    main()
