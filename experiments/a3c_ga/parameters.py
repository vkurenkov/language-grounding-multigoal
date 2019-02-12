"""
This file is responsible for the definitions of:
    - Environment  (training and testing)
    - Instructions (training and testing)
    - Agent's training parameters
    - Agent's testing parameters
"""

import os

from envs.definitions           import InstructionEnvironmentDefinition
from envs.gridworld_simple.env  import FindItemsEnvObsOnlyGrid
from envs.gridworld_simple.env  import FindItemsEnv
from instructions               import get_level0_instructions
from instructions               import get_instructions_tokenizer
from utils.training             import create_experiment_folder
from utils.training             import unroll_parameters_in_str


# Target environment
env_definition = InstructionEnvironmentDefinition(
                        FindItemsEnvObsOnlyGrid,
                        width=10, height=10, num_items=3,
                        must_avoid_non_targets=True,
                        reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS
)

# Agent's training parameters
train_parameters = {
    "max_episodes":        150000,
    "max_episode_len":     30,
    "num_processes":       2,
    "learning_rate":       0.001,
    "gamma":               0.85,
    "tau":                 1.00,
    "entropy_coeff":       0.01,
    "num_bootstrap_steps": 30,
    "seed":                0
}

# Agent's testing parameters
TEST_MODE_STOCHASTIC      = "Stochastic"
TEST_MODE_DETERMINISTIC   = "Determenistic"
test_parameters = {
    "test_every":  50,
    "test_repeat": 5,
    "seed":        1337,
    "mode":        TEST_MODE_STOCHASTIC
}

# Target instructions
instructions_level = "level0"
instructions       = get_level0_instructions()
tokenizer          = get_instructions_tokenizer(instructions)

# Experimental logging path setup
experiment_folder = create_experiment_folder(
                        os.path.join(os.path.dirname(__file__), "logs"), 
                        instructions_level,
                        env_definition.name(), 
                        unroll_parameters_in_str(train_parameters)
)