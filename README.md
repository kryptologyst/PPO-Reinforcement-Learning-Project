# PPO Reinforcement Learning Project

A well-structured implementation of Proximal Policy Optimization (PPO) for reinforcement learning, featuring state-of-the-art techniques, comprehensive logging, and interactive visualization tools.

## Features

- **Modern PPO Implementation**: Clean, type-hinted implementation with support for both discrete and continuous action spaces
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration for experiment tracking
- **Interactive UI**: Streamlit dashboard for training monitoring and hyperparameter tuning
- **Jupyter Notebooks**: Interactive experimentation and analysis
- **Configuration Management**: YAML-based configuration system
- **Checkpoint Management**: Automatic model saving and loading
- **Unit Tests**: Comprehensive test coverage for all components
- **Multiple Environments**: Support for various Gymnasium environments

## 📁 Project Structure

```
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base agent class
│   │   └── ppo.py          # PPO implementation
│   └── utils/
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       ├── logger.py       # Logging utilities
│       └── training.py     # Training utilities
├── configs/
│   └── default.yaml       # Default configuration
├── notebooks/
│   └── ppo_training.ipynb # Interactive training notebook
├── tests/
│   ├── __init__.py
│   └── test_ppo.py        # Unit tests
├── train.py               # Main training script
├── app.py                 # Streamlit UI
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/PPO-Reinforcement-Learning-Project.git
   cd PPO-Reinforcement-Learning-Project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Training

Train a PPO agent on CartPole-v1:

```bash
python train.py
```

### Custom Configuration

Train with custom parameters:

```bash
python train.py --config configs/default.yaml --episodes 1000 --env LunarLander-v2
```

### Interactive UI

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

### Jupyter Notebook

Open the interactive notebook:

```bash
jupyter notebook notebooks/ppo_training.ipynb
```

## Supported Environments

| Environment | Type | State Space | Action Space | Description |
|-------------|------|-------------|--------------|-------------|
| CartPole-v1 | Discrete | 4D continuous | 2 discrete | Balance a pole on a cart |
| LunarLander-v2 | Discrete | 8D continuous | 4 discrete | Land a spacecraft on the moon |
| MountainCar-v0 | Discrete | 2D continuous | 3 discrete | Drive a car up a mountain |
| Acrobot-v1 | Discrete | 6D continuous | 3 discrete | Swing up a double pendulum |
| Pendulum-v1 | Continuous | 3D continuous | 1D continuous | Balance an inverted pendulum |

## Configuration

The project uses YAML configuration files. Key parameters:

```yaml
ppo:
  episodes: 1000
  batch_steps: 2048
  actor_lr: 3e-4
  critic_lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  ppo_epochs: 10
  entropy_coef: 0.01
  value_coef: 0.5
  hidden_sizes: [128, 128]
  activation: "relu"

environment:
  name: "CartPole-v1"
  render_mode: null
  max_episode_steps: 500

logging:
  log_dir: "logs"
  tensorboard: true
  wandb:
    enabled: false
    project: "ppo-rl-project"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_ppo.py
```

## Monitoring and Logging

### TensorBoard

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

### Weights & Biases

Enable W&B logging by setting `wandb.enabled: true` in your config and providing your API key.

### Streamlit Dashboard

The interactive dashboard provides:
- Real-time training progress
- Hyperparameter tuning interface
- Environment visualization
- Performance analysis

## 🔧 Advanced Usage

### Custom Environments

Create custom environments by extending Gymnasium's `Env` class:

```python
import gymnasium as gym
from gymnasium import spaces

class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = spaces.Discrete(2)
    
    def step(self, action):
        # Your environment logic
        pass
    
    def reset(self, seed=None):
        # Your reset logic
        pass
```

### Custom Agents

Extend the base agent class for custom algorithms:

```python
from src.agents import BaseAgent

class CustomAgent(BaseAgent):
    def select_action(self, state, deterministic=False):
        # Your action selection logic
        pass
    
    def update(self, batch):
        # Your update logic
        pass
```

### Checkpoint Management

Save and load trained models:

```python
from src.utils import CheckpointManager

# Save checkpoint
checkpoint_manager = CheckpointManager()
checkpoint_manager.save_checkpoint(agent, episode, metrics)

# Load checkpoint
checkpoint_manager.load_checkpoint(agent, "checkpoints/checkpoint_episode_0100.pt")
```

## Results

### CartPole-v1 Performance

| Metric | Value |
|--------|-------|
| Episodes to solve | ~200-300 |
| Final reward | 475-500 |
| Training time | ~2-5 minutes |
| Convergence | Stable |

### Hyperparameter Sensitivity

| Parameter | Impact | Recommended Range |
|-----------|--------|-------------------|
| Learning Rate | High | 1e-4 to 1e-3 |
| Clip Epsilon | Medium | 0.1 to 0.3 |
| GAE Lambda | Medium | 0.9 to 0.99 |
| Entropy Coef | Low | 0.001 to 0.1 |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the original PPO paper
- Gymnasium team for the excellent RL environment library
- PyTorch team for the deep learning framework
- Stable Baselines3 for reference implementations

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow training**: Check if you're using GPU acceleration
3. **Import errors**: Ensure all dependencies are installed
4. **Environment not found**: Install additional gymnasium environments


# PPO-Reinforcement-Learning-Project
