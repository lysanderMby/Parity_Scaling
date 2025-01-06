import matplotlib.pyplot as plt
from pathlib import Path

def plot_progress(loss_data, accuracy_data, task_accuracy_data, current_flops, save_dir: Path):
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    if loss_data:  # Only plot if we have data
        flops, losses = zip(*loss_data)
        plt.loglog(flops, losses)
        plt.xlim(min(flops)/2, max(flops)*2)  # Add some padding
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Loss')
    plt.title('Loss vs FLOPs')
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    if accuracy_data:  # Only plot if we have data
        flops, accuracies = zip(*accuracy_data)
        plt.semilogx(flops, accuracies)
        plt.xlim(min(flops)/2, max(flops)*2)  # Add some padding
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs FLOPs')
    
    # Task-specific accuracy plot
    plt.subplot(1, 3, 3)
    min_flops = float('inf')
    max_flops = 0
    for task, data in task_accuracy_data.items():
        if data:  # Only plot if we have data
            flops, accuracies = zip(*data)
            min_flops = min(min_flops, min(flops))
            max_flops = max(max_flops, max(flops))
            plt.semilogx(flops, accuracies, label=f'Task {task+1}')
    if min_flops != float('inf'):  # Only set limits if we have data
        plt.xlim(min_flops/2, max_flops*2)  # Add some padding
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Task-specific Accuracy')
    plt.title('Task-specific Accuracies vs FLOPs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f'progress_{current_flops:.0e}_flops.png')
    plt.close()

def main_plot(all_results, exp_dir: Path):
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    min_flops = float('inf')
    max_flops = 0
    for result in all_results:
        if not result['loss_data']:  # Skip if no data
            continue
        flops, losses = zip(*result['loss_data'])
        min_flops = min(min_flops, min(flops))
        max_flops = max(max_flops, max(flops))
        config = result['model_config']
        plt.loglog(flops, losses, 
                  label=f"{config['num_layers']}x{config['hidden_size']}")
    if min_flops != float('inf'):  # Only set limits if we have data
        plt.xlim(min_flops/2, max_flops*2)  # Add some padding
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Loss')
    plt.title('Loss vs FLOPs')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for result in all_results:
        if not result['accuracy_data']:  # Skip if no data
            continue
        flops, accuracies = zip(*result['accuracy_data'])
        config = result['model_config']
        plt.semilogx(flops, accuracies, 
                    label=f"{config['num_layers']}x{config['hidden_size']}")
    if min_flops != float('inf'):  # Only set limits if we have data
        plt.xlim(min_flops/2, max_flops*2)  # Add some padding
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs FLOPs')
    plt.legend()

    # Task-specific accuracy plot
    plt.subplot(1, 3, 3)
    for result in all_results:
        config = result['model_config']
        for task, data in result['task_accuracy_data'].items():
            if not data:  # Skip if no data
                continue
            flops, accuracies = zip(*data)
            plt.semilogx(flops, accuracies, 
                        label=f"Task {task+1} - {config['num_layers']}x{config['hidden_size']}")
    if min_flops != float('inf'):  # Only set limits if we have data
        plt.xlim(min_flops/2, max_flops*2)  # Add some padding
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Task-specific Accuracy')
    plt.title('Task-specific Accuracies vs FLOPs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(exp_dir / 'final_results.png')
    plt.close() 