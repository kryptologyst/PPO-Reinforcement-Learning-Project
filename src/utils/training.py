"""Training utilities and checkpoint management."""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import json
from datetime import datetime


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        keep_last: int = 5,
        save_frequency: int = 100,
    ) -> None:
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            keep_last: Number of recent checkpoints to keep
            save_frequency: Save checkpoint every N episodes
        """
        self.save_dir = Path(save_dir)
        self.keep_last = keep_last
        self.save_frequency = save_frequency
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        agent,
        episode: int,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save agent checkpoint.
        
        Args:
            agent: Agent to save
            episode: Current episode number
            metrics: Training metrics
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_episode_{episode:06d}.pt"
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "metadata": metadata or {},
        }
        
        # Save agent state
        agent.save(str(checkpoint_path))
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, agent, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load agent checkpoint.
        
        Args:
            agent: Agent to load state into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
        """
        # Load agent state
        agent.load(checkpoint_path)
        
        # Load metadata
        metadata_path = Path(checkpoint_path).with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return {}
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = list(self.save_dir.glob("checkpoint_episode_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by episode number
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return str(checkpoint_files[-1])
    
    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_files = list(self.save_dir.glob("checkpoint_episode_*.pt"))
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return [str(f) for f in checkpoint_files]
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_files = list(self.save_dir.glob("checkpoint_episode_*.pt"))
        
        if len(checkpoint_files) <= self.keep_last:
            return
        
        # Sort by episode number
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        
        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[:-self.keep_last]:
            checkpoint_file.unlink()
            # Also remove metadata file
            metadata_file = checkpoint_file.with_suffix(".json")
            if metadata_file.exists():
                metadata_file.unlink()


def create_training_summary(
    config: Dict[str, Any],
    final_metrics: Dict[str, float],
    training_time: float,
) -> Dict[str, Any]:
    """
    Create a training summary.
    
    Args:
        config: Training configuration
        final_metrics: Final training metrics
        training_time: Total training time in seconds
        
    Returns:
        Training summary dictionary
    """
    return {
        "config": config,
        "final_metrics": final_metrics,
        "training_time": training_time,
        "timestamp": datetime.now().isoformat(),
    }


def save_training_summary(summary: Dict[str, Any], filepath: str) -> None:
    """
    Save training summary to file.
    
    Args:
        summary: Training summary dictionary
        filepath: Path to save summary
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
