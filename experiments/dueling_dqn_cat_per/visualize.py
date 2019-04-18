"""
This file contains visualization process for pre-trained agents.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F

from utils.training            import fix_random_seeds
from envs.gridworld_simple.env import FindItemsVisualizator
from collections               import deque
from instructions			   import get_instructions
from instructions              import get_instructions_tokenizer

# Experimental environment and layouts
from experiments.dueling_dqn_cat_per.parameters import env_definition
from experiments.dueling_dqn_cat_per.parameters import layouts_parameters

# Experimental instructions
from experiments.dueling_dqn_cat_per.parameters import instructions_parameters

# Agent's training and testing parameters
from experiments.dueling_dqn_cat_per.parameters import train_parameters
from experiments.dueling_dqn_cat_per.parameters import test_parameters
from experiments.dueling_dqn_cat_per.parameters import get_experiment_folder

# Load the computational model
from experiments.dueling_dqn_cat_per.model      import Model
from experiments.dueling_dqn_cat_per.model      import prepare_model_input
from experiments.dueling_dqn_cat_per.parameters import device

# Experimental folder
experiment_folder        = get_experiment_folder()

# Retrieve instructions
train_instructions, _, _ = get_instructions(
							instructions_parameters["level"], 
							instructions_parameters["max_train_subgoals"], 
							instructions_parameters["unseen_proportion"],
							instructions_parameters["seed"],
                            instructions_parameters["conjunctions"])
tokenizer         		 = get_instructions_tokenizer(
							train_instructions,
							train_parameters["padding_len"])

# Retrieve layouts
layouts                  = env_definition.build_env().generate_layouts(
							layouts_parameters["num_train"] + layouts_parameters["num_test"],
							layouts_parameters["seed"])
train_layouts			 = layouts[:layouts_parameters["num_train"]]


fix_random_seeds(train_parameters["seed"])
stack_frames = train_parameters["stack_frames"]


model = Model(tokenizer.get_vocabulary_size(), stack_frames, train_parameters["max_episode_len"])
model.load_state_dict(torch.load(os.path.join(experiment_folder, "best.model"), map_location='cpu'))
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
    env.fix_initial_positions(train_layouts[0])
    observation, reward, done, _ = env.reset()

    # Keep last frames
    last_frames = deque([observation for _ in range(stack_frames)], stack_frames)

    num_steps = 0
    FindItemsVisualizator.pyplot(env)

    while not done:
        # Action inference
        with torch.no_grad():
            model_input 	 = prepare_model_input(list(last_frames), instruction_idx)
            values 			 = model(model_input)
            print("My life values: {}".format(values))
        
        # Greedy
        action = torch.argmax(values, dim=-1).cpu().item()

        observation, reward, done,  _ = env.step(action)
        print("What outside world thinks of me: {}".format(reward))
        last_frames.append(observation)
        num_steps += 1
        FindItemsVisualizator.pyplot(env)
    