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
from instructions               import get_level0_instructions
from instructions               import get_level1_instructions
from instructions               import get_instructions_tokenizer
from utils.training             import create_experiment_folder
from utils.training             import unroll_parameters_in_str

# Computing device
device = torch.device("cuda")

# Target environment
env_definition = InstructionEnvironmentDefinition(
                        FindItemsEnvObsOnlyGrid,
                        width=10, height=10, num_items=3,
                        must_avoid_non_targets=True,
                        reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS,
                        fixed_positions=[(0, 0,), (5, 5), (3, 3), (7, 7)]
)

# Agent's training parameters
train_parameters = {
    "max_episodes":        60000*2,
    "max_episode_len":     30,

    "learning_rate":       0.001,
    "gamma":               1.0,

    "stack_frames":        2,

    "eps_start":           0.95,
    "eps_end":             0.01,
    "eps_episodes":        60000*2,

    "replay_size":         1000000, # in frames
    "batch_size":          512,
    "model_switch":        570, # in episodes

    "padding_len":         24,
    "seed":                0
}

# Agent's on-training testing parameters
test_parameters = {
    "test_every":  570,
    "test_repeat": 2
}

# Target instructions
instructions_level = "level1"
instructions       = get_level1_instructions()
tokenizer          = get_instructions_tokenizer(instructions, train_parameters["padding_len"])

# Experimental logging path setup
experiment_folder = create_experiment_folder(
                        os.path.join(os.path.dirname(__file__), "logs"), 
                        instructions_level,
                        env_definition.name(), 
                        unroll_parameters_in_str(train_parameters)
)