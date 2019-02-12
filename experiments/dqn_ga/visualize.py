"""
This file contains visualization process for pre-trained agents.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F

from utils.training            import fix_random_seeds
from envs.gridworld_simple.env import FindItemsVisualizator

# Target environment
from experiments.dqn_ga.parameters import env_definition

# Instructions
from experiments.dqn_ga.parameters import instructions
from experiments.dqn_ga.parameters import instructions_level
from experiments.dqn_ga.parameters import tokenizer

# Agent's training and testing parameters
from experiments.dqn_ga.parameters import train_parameters
from experiments.dqn_ga.parameters import test_parameters
from experiments.dqn_ga.parameters import experiment_folder
from experiments.dqn_ga.parameters import TEST_MODE_STOCHASTIC
from experiments.dqn_ga.parameters import TEST_MODE_DETERMINISTIC

# Load the computational model
from experiments.dqn_ga.model      import Model
from experiments.dqn_ga.model      import prepare_model_input
from experiments.dqn_ga.parameters import device

fix_random_seeds(test_parameters["seed"])

model = Model(tokenizer.get_vocabulary_size(), train_parameters["max_episode_len"])
model.load_state_dict(torch.load(os.path.join(experiment_folder, "best.model")))
model.to(device)
model.eval()

while True:
    # Instruction input
    print("Enter your navigational instruction: ")
    instruction_raw = input()
    print("Enter items to visit in the order defined by instruction:")
    instruction_items = [int(item) for item in input().split(" ")]

    # Instruction encoding
    instruction_idx = tokenizer.text_to_ids(instruction_raw)
    instruction_idx = np.array(instruction_idx)
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    
    # Environment reset
    env = env_definition.build_env(instruction_items)
    observation, reward, done, _ = env.reset()
    num_steps = 0
    FindItemsVisualizator.pyplot(env)

    while not done:
        # Action inference
        with torch.no_grad():
            model_input 	 = prepare_model_input(observation, instruction_idx)
            values 			 = model(model_input)
        
        # Greedy
        action = torch.argmax(values, dim=-1).cpu().item()

        observation, reward, done,  _ = env.step(action)
        num_steps += 1
        FindItemsVisualizator.pyplot(env)
    