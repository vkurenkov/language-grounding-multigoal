import torch
import numpy
import random
import os


def fix_random_seeds(seed: int) -> None:
    """
    Use in the beginning of the program only.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    numpy.random.seed(seed)
    random.seed(seed)

def create_experiment_folder(root_path: str, env_name: str, agent_name: str) -> str:
    experiment_path = os.path.join(root_path, env_name, agent_name)

    if os.path.isdir(experiment_path):
        print("This experiment already exists. Do you want to proceed (erase the old one)? (y/n)")
        answer = input()
        if answer == "y":
            os.rmdir(experiment_path)
        else:
            # Stop the experiment
            exit(0)

    # Make sure we do not overwrite existing experiments
    os.makedirs(experiment_path, exist_ok=False)

    return experiment_path