import torch
import numpy
import random
import os
import shutil

from typing import Dict


def fix_random_seeds(seed: int) -> None:
    """
    Use in the beginning of the program only.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    numpy.random.seed(seed)
    random.seed(seed)

def create_experiment_folder(root_path: str, instructions_level, env_name: str, layouts_name: str, agent_name: str, erase_folder=None) -> str:
    experiment_path_with_agent = get_experiment_folder(root_path, instructions_level, env_name, layouts_name, agent_name)

    print("Experiment path: ")
    print(experiment_path_with_agent)
    print()
    if os.path.isdir(experiment_path_with_agent):
        if erase_folder is None:
            print("This experiment already exists. Do you want to erase/add info/cancel/skip? (e/a/c/s)")
            answer = input()
            if answer == "e":
                shutil.rmtree(experiment_path_with_agent, ignore_errors=True)
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
            elif answer == "s":
                pass
            else:
                raise Exception("Unexpected input.")
        elif erase_folder:
            shutil.rmtree(experiment_path_with_agent, ignore_errors=True)
        else:
            pass

    # Make sure we do not overwrite existing experiments
    os.makedirs(experiment_path_with_agent, exist_ok=True)

    return experiment_path_with_agent

def get_experiment_folder(root_path: str, instructions_level, env_name: str, layouts_name: str, agent_name: str) -> str:
    experiment_path_without_agent = os.path.join(root_path, instructions_level, env_name, layouts_name)
    experiment_path_with_agent    = os.path.join(experiment_path_without_agent, agent_name)
    return experiment_path_with_agent

def unroll_parameters_in_str(parameters: Dict) -> str:
    unrolled = ""
    for key in sorted(parameters.keys()):
        unrolled += "{}_{}-".format(key, parameters[key])

    return unrolled[0:-1] # I know, I know. Not the most elegant way.