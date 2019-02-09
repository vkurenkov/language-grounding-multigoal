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
from experiments.a3c.parameters import env_definition

# Instructions
from experiments.a3c.parameters import instructions
from experiments.a3c.parameters import instructions_level
from experiments.a3c.parameters import tokenizer

# Agent's training and testing parameters
from experiments.a3c.parameters import train_parameters
from experiments.a3c.parameters import test_parameters
from experiments.a3c.parameters import experiment_folder
from experiments.a3c.parameters import TEST_MODE_STOCHASTIC
from experiments.a3c.parameters import TEST_MODE_DETERMINISTIC

# Load the computational model
from experiments.a3c.model import A3C_LSTM_GA

fix_random_seeds(test_parameters["seed"])

model = A3C_LSTM_GA(tokenizer.get_vocabulary_size(), train_parameters["max_episode_len"])
model.load_state_dict(torch.load(os.path.join(experiment_folder, "best.model")))
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
    obs, rew, done, _ = env.reset()
    num_steps = 0
    FindItemsVisualizator.pyplot(env)

    # Agent's initial state
    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)

    while not done:
        # Action inference
        with torch.no_grad():
            observation     = torch.from_numpy(obs).float()
            tx = torch.tensor(np.array([num_steps + 1]), dtype=torch.int64)
            _, logit, (hx, cx) = model((
                                    torch.tensor(observation).unsqueeze(0),
                                    torch.tensor(instruction_idx),
                                    (tx, hx, cx)
                                ))
            prob = F.softmax(logit, dim=-1)

            if test_parameters["mode"] == TEST_MODE_DETERMINISTIC:
                action = torch.argmax(prob, dim=-1).item()
            elif test_parameters["mode"] == TEST_MODE_STOCHASTIC:
                action = prob.multinomial(1).data.numpy()[0, 0]

        obs, rew, done, _ = env.step(action)
        num_steps += 1
        FindItemsVisualizator.pyplot(env)
    