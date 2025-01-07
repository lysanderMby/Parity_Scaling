'''
Main file to run when performing experiments.

Trains neural networks on synthetic data generated to represent a variant of the sparse parity problem.

The sparse parity problem is a problem where the input is a binary vector of length n, 
where the first k bits are the task code and the remaining n-k bits are the message.

The neural network is trained to learn the mapping from the input to the output.
This mapping involves taking the parity (the sum mod 2) of certain bits of the message.
Which bits to use is determined by the task code.
'''

# Package imports
from pathlib import Path
import torch
from tqdm import tqdm

# Imports from within project files
from models.neural_net import NeuralNetwork
from data.synthetic_data import generate_dict
from training.trainer import train_and_evaluate
from utils.experiment import create_versioned_directory
from utils.plotting import main_plot

# Parameters
PARAMS = {
    'n_tasks': 1, # number of unique tasks being trained over
    'len_taskcode': 4, # number of bits in the task code
    'num_checks': 5, # number of bits in the message that are used to determine the output
    'len_message': 16, # number of bits in the message
    'num_samples': 1000, # number of samples to generate for each task
    'input_size': 20,  # len_taskcode + len_message. Used for model initialisation
    'output_size': 1,
    'learning_rate': 0.005,
    'batch_size': 32,
    'flop_budget': 1e10, # total number of estimated flops expended per training run
    'task_sample_freq': 1e5,  # flop_budget/1e3
    'plot_freq': 2e7,  # flop_budget/5
    'samples_per_task': 100
}

MODEL_CONFIGS = [
    {"num_layers": 2, "hidden_size": 8},
    {"num_layers": 4, "hidden_size": 16},
    {"num_layers": 6, "hidden_size": 32},
    {"num_layers": 8, "hidden_size": 64},
    {"num_layers": 10, "hidden_size": 128},
    {"num_layers": 12, "hidden_size": 256},
    {"num_layers": 14, "hidden_size": 512}
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment directory
    exp_name = f"parity_scaling_flops_{PARAMS['flop_budget']:.0e}"
    exp_dir = create_versioned_directory(Path("experiments"), exp_name)
    print(f"Experiment directory: {exp_dir}")

    print(f"FLOP budget: {PARAMS['flop_budget']}")
    
    tasks_dict = generate_dict(
        PARAMS['n_tasks'], 
        PARAMS['len_taskcode'], 
        PARAMS['num_checks'], 
        PARAMS['len_message']
    )
    print(f"Generated tasks dictionary with {len(tasks_dict)} tasks")
    print("tasks_dict = ", tasks_dict.items())
    
    all_results = []
    
    # Add progress bar for model configurations
    for config in tqdm(MODEL_CONFIGS, desc="Training models", position=0, leave=True):
        print(f"\nTraining model with {config['num_layers']} layers and hidden size {config['hidden_size']}")
        model = NeuralNetwork(
            PARAMS['input_size'], 
            PARAMS['output_size'], 
            config["num_layers"], 
            config["hidden_size"]
        ).to(device)
        
        results = train_and_evaluate(
            model=model,
            params=PARAMS,
            tasks_dict=tasks_dict,
            exp_dir=exp_dir,
            model_config=config
        )
        all_results.append(results)
    
    # Create final plots
    main_plot(all_results, exp_dir)

if __name__ == '__main__':
    main() 