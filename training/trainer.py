import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from data.synthetic_data import generate_dataset, generate_dataset_for_task, CustomDataset
from utils.plotting import plot_progress
from typing import Dict, Tuple
from tqdm import tqdm
import logging
from datetime import datetime

class FlopCounter:
    def __init__(self, model: nn.Module, input_size: int, batch_size: int):
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        
    def count_linear_flops(self, in_features: int, out_features: int) -> Dict[str, int]:
        forward_flops = self.batch_size * out_features * (2 * in_features - 1)
        backward_flops = (
            self.batch_size * in_features * out_features * 2 +
            self.batch_size * out_features +
            self.batch_size * in_features * out_features * 2
        )
        return {"forward": forward_flops, "backward": backward_flops}
    
    def count_batch_norm_flops(self, num_features: int) -> Dict[str, int]:
        forward_flops = self.batch_size * num_features * 7
        backward_flops = self.batch_size * num_features * 10
        return {"forward": forward_flops, "backward": backward_flops}
    
    def count_relu_flops(self, num_elements: int) -> Dict[str, int]:
        return {
            "forward": num_elements,
            "backward": num_elements
        }
    
    def calculate_total_flops(self) -> Tuple[int, int]:
        total_forward_flops = 0
        total_backward_flops = 0
        current_size = self.input_size
        
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                flops = self.count_linear_flops(layer.in_features, layer.out_features)
                total_forward_flops += flops["forward"]
                total_backward_flops += flops["backward"]
                current_size = layer.out_features
                
            elif isinstance(layer, nn.BatchNorm1d):
                flops = self.count_batch_norm_flops(current_size)
                total_forward_flops += flops["forward"]
                total_backward_flops += flops["backward"]
                
            if isinstance(layer, nn.Linear) and layer != self.model.layers[-1]:
                flops = self.count_relu_flops(self.batch_size * current_size)
                total_forward_flops += flops["forward"]
                total_backward_flops += flops["backward"]
        
        return total_forward_flops, total_backward_flops

def train_and_evaluate(model, params, tasks_dict, exp_dir: Path, model_config: dict):
    # Setup logging (file only, no StreamHandler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_dir / f"training_{model_config['num_layers']}x{model_config['hidden_size']}.log"),
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    
    # Initialize FlopCounter
    flop_counter = FlopCounter(model, input_size=params['input_size'], batch_size=params['batch_size'])
    forward_flops, backward_flops = flop_counter.calculate_total_flops()
    
    logger.info(f"Starting training for model {model_config['num_layers']}x{model_config['hidden_size']}")
    logger.info(f"Estimated FLOPs per batch - Forward: {forward_flops}, Backward: {backward_flops}")
    
    loss_data = []
    accuracy_data = []
    task_accuracy_data = {i: [] for i in range(params['n_tasks'])}
    cumulative_flops = 0
    epoch = 0
    last_task_sample = 0
    last_plot = 0

    model_dir = exp_dir / f"model_{model_config['num_layers']}x{model_config['hidden_size']}"
    model_dir.mkdir(exist_ok=True)

    # Main progress bar for FLOPs
    progress_bar = tqdm(
        total=params['flop_budget'],
        desc=f"Training {model_config['num_layers']}x{model_config['hidden_size']}",
        position=1,
        leave=False,
        unit='FLOP',
        unit_scale=True,
        unit_divisor=1000
    )

    while cumulative_flops < params['flop_budget']:
        epoch += 1
        epoch_stats = train_epoch(model, params, tasks_dict, criterion, optimizer, 
                                forward_flops, backward_flops, logger)
        
        # Update progress and stats
        flops_delta = epoch_stats['cumulative_flops'] - cumulative_flops
        cumulative_flops = epoch_stats['cumulative_flops']
        progress_bar.update(flops_delta)
        progress_bar.set_postfix({
            'epoch': epoch,
            'loss': f"{epoch_stats['avg_loss']:.4f}",
            'acc': f"{epoch_stats['avg_accuracy']:.4f}"
        })
        
        # Log metrics to file only
        logger.info(f"Epoch {epoch} - Loss: {epoch_stats['avg_loss']:.4f}, "
                   f"Accuracy: {epoch_stats['avg_accuracy']:.4f}, "
                   f"Cumulative FLOPs: {cumulative_flops:,}")
        
        loss_data.append((cumulative_flops, epoch_stats['avg_loss']))
        accuracy_data.append((cumulative_flops, epoch_stats['avg_accuracy']))
        
        # Task-specific evaluation
        if (cumulative_flops - last_task_sample >= params['task_sample_freq'] or 
            cumulative_flops >= params['flop_budget']):
            last_task_sample = cumulative_flops
            evaluate_tasks(model, tasks_dict, params, device, forward_flops,
                         cumulative_flops, task_accuracy_data)
        
        # Plotting
        if cumulative_flops - last_plot >= params['plot_freq']:
            last_plot = cumulative_flops
            plot_progress(loss_data, accuracy_data, task_accuracy_data, 
                        cumulative_flops, model_dir)

    progress_bar.close()
    return {
        'loss_data': loss_data,
        'accuracy_data': accuracy_data,
        'task_accuracy_data': task_accuracy_data,
        'cumulative_flops': cumulative_flops,
        'model_config': model_config
    }

def train_epoch(model, params, tasks_dict, criterion, optimizer, forward_flops, backward_flops, logger):
    [data, value] = generate_dataset(tasks_dict, params['num_samples'], 
                                   params['len_taskcode'], params['len_message'])
    
    df = pd.DataFrame(np.concatenate((data, value.reshape(-1, 1)), axis=1))
    dataset = CustomDataset(df, next(model.parameters()).device)
    data_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    cumulative_flops = 0
    
    batch_progress = tqdm(data_loader, desc="Batches", position=2, leave=False)
    for inputs, labels in batch_progress:
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels.unsqueeze(1))
        predictions = (outputs >= 0.5).squeeze().long()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item() * inputs.size(0)
        cumulative_flops += forward_flops + backward_flops
        
        batch_progress.set_postfix({
            'loss': f'{batch_loss.item():.4f}',
            'acc': f'{(predictions == labels).float().mean().item():.4f}'
        })

    return {
        'avg_loss': epoch_loss / total,
        'avg_accuracy': correct / total,
        'cumulative_flops': cumulative_flops
    }

def evaluate_tasks(model, tasks_dict, params, device, forward_flops, 
                  cumulative_flops, task_accuracy_data):
    model.eval()
    for task_idx, task_code in enumerate(tasks_dict.keys()):
        [data, value] = generate_dataset_for_task(
            task_code, tasks_dict, params['samples_per_task'],
            params['len_taskcode'], params['len_message']
        )
        
        df = pd.DataFrame(np.concatenate((data, value.reshape(-1, 1)), axis=1))
        dataset = CustomDataset(df, device)
        loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                predictions = (outputs >= 0.5).squeeze().long()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                cumulative_flops += forward_flops
        
        task_accuracy = correct / total
        task_accuracy_data[task_idx].append((cumulative_flops, task_accuracy)) 