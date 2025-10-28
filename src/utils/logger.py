"""Logging utilities for training monitoring."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger:
    """Unified logging interface for TensorBoard and Weights & Biases."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity name
            wandb_config: W&B configuration
        """
        self.log_dir = Path(log_dir)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Setup TensorBoard
        if use_tensorboard:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.tb_writer = None
        
        # Setup Weights & Biases
        if use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config,
                dir=str(self.log_dir),
            )
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """
        Log a scalar value.
        
        Args:
            name: Name of the metric
            value: Value to log
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        
        if self.use_wandb:
            wandb.log({name: value}, step=step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log multiple scalar values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """
        Log a histogram.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
        
        if self.use_wandb:
            wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_figure(self, name: str, figure, step: int) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            name: Name of the figure
            figure: Matplotlib figure
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_figure(name, figure, step)
        
        if self.use_wandb:
            wandb.log({name: wandb.Image(figure)}, step=step)
    
    def close(self) -> None:
        """Close the logger."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()


def setup_logging(level: str = "INFO") -> None:
    """
    Setup basic logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
