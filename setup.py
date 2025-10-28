#!/usr/bin/env python3
"""Setup script for PPO Reinforcement Learning Project."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["logs", "checkpoints", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def run_tests():
    """Run basic tests."""
    return run_command("python -c \"import sys; sys.path.append('src'); from agents import PPOAgent; print('Import test passed')\"", "Testing imports")

def main():
    """Main setup function."""
    print("🚀 Setting up PPO Reinforcement Learning Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Run basic tests
    if not run_tests():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python train.py")
    print("2. Launch UI: streamlit run app.py")
    print("3. Open notebook: jupyter notebook notebooks/ppo_training.ipynb")
    print("4. Run tests: pytest tests/")

if __name__ == "__main__":
    main()
