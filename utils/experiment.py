from pathlib import Path
import json
from datetime import datetime

def create_versioned_directory(base_dir: Path, name: str) -> Path:
    """
    Create a versioned directory to avoid overwriting existing experiments.
    
    Args:
        base_dir: Base directory for experiments
        name: Base name for the experiment directory
        
    Returns:
        Path: Path to the created directory
    """
    base_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = 1
    
    while True:
        versioned_name = f"{name}__{timestamp}_v{version}"
        dir_path = base_dir / versioned_name
        if not dir_path.exists():
            dir_path.mkdir()
            return dir_path
        version += 1

def save_experiment_config(exp_dir: Path, params: dict, model_configs: list):
    """
    Save experiment configuration to a JSON file.
    
    Args:
        exp_dir: Experiment directory
        params: Dictionary of parameters
        model_configs: List of model configurations
    """
    config = {
        'parameters': params,
        'model_configs': model_configs,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(exp_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=4) 