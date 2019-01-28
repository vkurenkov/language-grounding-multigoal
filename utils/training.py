import torch
import numpy
import random
import os
import shutil


def fix_random_seeds(seed: int) -> None:
    """
    Use in the beginning of the program only.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    numpy.random.seed(seed)
    random.seed(seed)

def create_experiment_folder(root_path: str, instructions_level, env_name: str, agent_name: str) -> str:
    experiment_path_without_agent = os.path.join(root_path, instructions_level, env_name)
    experiment_path_with_agent    = os.path.join(experiment_path_without_agent, agent_name)

    print("Experiment path: ")
    print(experiment_path_with_agent)
    print()
    if os.path.isdir(experiment_path_with_agent):
        print("This experiment already exists. Do you want to erase/add info/cancel? (e/a/c)")
        answer = input()
        if answer == "e":
            shutil.rmtree(experiment_path, ignore_errors=True)
        elif answer == "c":
            # Stop the experiment
            exit(0)
        elif answer == "a":
            print("Enter additional info (will be added as a prefix to the last folder.")
            info = input()
            while not info:
                print("Please, enter non-empty info.")
                info = input()

            experiment_path_with_agent = os.path.join(experiment_path_without_agent, info + "-" + agent_name)
        else:
            raise Exception("Unexpected input.")

    # Make sure we do not overwrite existing experiments
    os.makedirs(experiment_path_with_agent, exist_ok=False)

    return experiment_path_with_agent