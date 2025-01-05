import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_random_binary_string(length):
    return ''.join(random.choice(['0', '1']) for _ in range(length))

def generate_dict(n_tasks, len_taskcode, num_checks, len_message):
    unique_strings = set()
    tasks_dict = {}
    while len(unique_strings) < n_tasks:
        binary_string = generate_random_binary_string(len_taskcode)
        if binary_string not in unique_strings:
            unique_strings.add(binary_string)
            integer_list = [random.randint(0, len_message-1) for _ in range(num_checks)]
            tasks_dict[binary_string] = integer_list
    return tasks_dict

def generate_dataset(tasks_dict, num_samples, len_taskcode, len_message):
    data = np.zeros((num_samples, len_taskcode + len_message))
    value = np.zeros(num_samples)
    for i in range(num_samples):
        rand_task = np.random.choice(list(tasks_dict))
        rand_checkbits = tasks_dict[rand_task]
        message = generate_random_binary_string(len_message)
        parity_bit = sum(int(message[j]) for j in rand_checkbits) % 2
        data[i] = np.concatenate((np.array(list(rand_task)), np.array(list(message))))
        value[i] = parity_bit
    return [data, value]

def generate_dataset_for_task(task_code, tasks_dict, num_samples, len_taskcode, len_message):
    data = np.zeros((num_samples, len_taskcode + len_message))
    value = np.zeros(num_samples)
    rand_checkbits = tasks_dict[task_code]
    for i in range(num_samples):
        message = generate_random_binary_string(len_message)
        parity_bit = sum(int(message[j]) for j in rand_checkbits) % 2
        data[i] = np.concatenate((np.array(list(task_code)), np.array(list(message))))
        value[i] = parity_bit
    return [data, value]

class CustomDataset(Dataset):
    def __init__(self, dataframe, device):
        self.data = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float32, device=device)
        self.target = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32, device=device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx] 