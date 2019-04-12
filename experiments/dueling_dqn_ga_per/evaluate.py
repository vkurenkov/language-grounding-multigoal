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
from experiments.dueling_dqn_ga_per.parameters import env_definition

# Instructions
from experiments.dueling_dqn_ga_per.parameters import instructions
from experiments.dueling_dqn_ga_per.parameters import instructions_level
from experiments.dueling_dqn_ga_per.parameters import tokenizer

# Agent's training and testing parameters
from experiments.dueling_dqn_ga_per.parameters import train_parameters
from experiments.dueling_dqn_ga_per.parameters import test_parameters
from experiments.dueling_dqn_ga_per.parameters import experiment_folder
from experiments.dueling_dqn_ga_per.parameters import TEST_MODE_STOCHASTIC
from experiments.dueling_dqn_ga_per.parameters import TEST_MODE_DETERMINISTIC

# Load the computational model
from experiments.dueling_dqn_ga_per.model      import Model
from experiments.dueling_dqn_ga_per.model      import prepare_model_input
from experiments.dueling_dqn_ga_per.parameters import device

from collections import deque

fix_random_seeds(test_parameters["seed"])
stack_frames = train_parameters["stack_frames"]

model = Model(tokenizer.get_vocabulary_size(), stack_frames, train_parameters["max_episode_len"])
model.load_state_dict(torch.load(os.path.join(experiment_folder, "best.model"), map_location='cpu'))
model.to(device)
model.eval()

for instruction in instructions:
    instruction_raw   = instruction[0]
    instruction_items = instruction[1]

    # Instruction encoding
    instruction_idx = tokenizer.text_to_ids(instruction_raw)
    instruction_idx = np.array(instruction_idx)
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

    # Environment reset
    env = env_definition.build_env(instruction_items)
    observation, reward, done, _ = env.reset()

    # Keep last frames
    last_frames = deque([observation for _ in range(stack_frames)], stack_frames)

    num_steps = 0

    while not done:
        # Action inference
        with torch.no_grad():
            model_input 	 = prepare_model_input(list(last_frames), instruction_idx)
            values 			 = model(model_input)
        
        # Greedy
        action = torch.argmax(values, dim=-1).cpu().item()

        observation, reward, done,  _ = env.step(action)
        last_frames.append(observation)
        num_steps += 1
        FindItemsVisualizator.pyplot(env)


# We should check for different evaluation scenarios
    # 2 factors, instructions and layouts.
    # Setups:
    #   1 - Training (seen)  layouts, training instructions
    #   2 - Testing  (unsen) layouts, training instructions
    #
    