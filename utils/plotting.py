import matplotlib.pyplot as plt
from pathlib import Path

def plot_progress(loss_data, accuracy_data, task_accuracy_data, current_flops, save_dir: Path):
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    flops, losses = zip(*loss_data)
    plt.plot(flops, losses)
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Loss')
    plt.title('Loss vs FLOPs')
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    flops, accuracies = zip(*accuracy_data)
    plt.plot(flops, accuracies)
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs FLOPs')
    
    # Task-specific accuracy plot
    plt.subplot(1, 3, 3)
    for task, data in task_accuracy_data.items():
        if data:
            flops, accuracies = zip(*data)
            plt.plot(flops, accuracies, label=f'Task {task+1}')
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
    for result in all_results:
        flops, losses = zip(*result['loss_data'])
        config = result['model_config']
        plt.loglog(flops, losses, 
                  label=f"{config['num_layers']}x{config['hidden_size']}")
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Loss')
    plt.title('Loss vs FLOPs')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for result in all_results:
        flops, accuracies = zip(*result['accuracy_data'])
        config = result['model_config']
        plt.semilogx(flops, accuracies, 
                    label=f"{config['num_layers']}x{config['hidden_size']}")
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs FLOPs')
    plt.legend()

    # Task-specific accuracy plot
    plt.subplot(1, 3, 3)
    for result in all_results:
        config = result['model_config']
        for task, data in result['task_accuracy_data'].items():
            flops, accuracies = zip(*data)
            plt.semilogx(flops, accuracies, 
                        label=f"Task {task+1} - {config['num_layers']}x{config['hidden_size']}")
    plt.xlabel('Cumulative FLOPs')
    plt.ylabel('Task-specific Accuracy')
    plt.title('Task-specific Accuracies vs FLOPs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(exp_dir / 'final_results.png')
    plt.close() 