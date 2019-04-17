"""
This file is responsible for the definitions of:
    - Environment  (training and testing)
    - Instructions (training and testing)
    - Agent's training parameters
    - Agent's testing parameters
"""

import os
import torch

from envs.definitions           import InstructionEnvironmentDefinition
from envs.gridworld_simple.env  import FindItemsEnvObsOnlyGrid
from envs.gridworld_simple.env  import FindItemsEnv
from utils.training             import create_experiment_folder
from utils.training             import unroll_parameters_in_str
from functools                  import partial


# Computing device
device = torch.device("cuda")

# Target environment
env_definition = InstructionEnvironmentDefinition(
                        FindItemsEnvObsOnlyGrid,
                        width=10, height=10, num_items=3,
                        must_avoid_non_targets=True,
                        reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS
)

# Experimental layouts description
layouts_parameters = {
    "seed":      0,
    "num_train": 1,
    "num_test":  1
}

# Experimental instructions description
instructions_parameters = {
    "seed":                0,
    "level":               1,
    "max_train_subgoals":  3,
    "unseen_proportion":   0.5,
    "conjunctions":        "only_comma" # [all, only_comma]
}

# Agent's training parameters
train_parameters = {
    "max_iterations":      4000,
    "max_episode_len":     30,

    "learning_rate":       0.001,
    "gamma":               1.0,
    "bootstrap_steps":     1,

    "stack_frames":        4,

    "eps_start":           0.95,
    "eps_end":             0.01,
    "eps_iterations":      4000,

    "replay_size":         1000000, # in frames
    "batch_size":          512,
    "model_switch":        5, # in iterations

    "padding_len":         24,
    "seed":                0
}

# Agent's on-training testing parameters
test_parameters = {
    "test_every":  50,
    "test_repeat": 1
}

# Experimental logging path setup
def get_experiment_folder():
    return create_experiment_folder(
        os.path.join(os.path.dirname(__file__), "logs"),
        "instructions_{}".format(unroll_parameters_in_str(instructions_parameters)),
        env_definition.name(),
        "layouts_{}".format(unroll_parameters_in_str(layouts_parameters)),
        unroll_parameters_in_str(train_parameters))