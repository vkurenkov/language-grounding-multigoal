import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.dueling_dqn_cat.parameters import device
from typing import Tuple

def prepare_model_input(observations, instruction_indices) -> Tuple[torch.Tensor, torch.Tensor]:
    last_observations = [torch.tensor(observation).unsqueeze(0).to(device) for observation in observations]
    return (last_observations, torch.tensor(instruction_indices, dtype=torch.int64).to(device))


class Model(nn.Module):
    def __init__(self, input_size: int, stack_frames: int, max_episode_length: int):
        super(Model, self).__init__()

        # Image Processing (batch_size, 4, 10, 10)
        self.img1 = nn.Conv2d(in_channels=4, out_channels=128, kernel_size=3, stride=1)
        self.img2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1)
        self.img3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        # 2304 features

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = input_size
        self.embedding = nn.Embedding(self.input_size, 128)
        self.gru = nn.GRU(128, self.gru_hidden_size, batch_first=True)

        self.linear = nn.Linear(2304, 256)
        self.stack_frames = stack_frames

        self.value_function = nn.Sequential(
            nn.Linear(256 * stack_frames + self.gru_hidden_size, 128 * stack_frames),
            nn.ReLU(),
            nn.Linear(128 * stack_frames, 64 * stack_frames),
            nn.Linear(64 * stack_frames, 4)
        )
        self.advantage_function = nn.Sequential(
            nn.Linear(256 * stack_frames + self.gru_hidden_size, 128 * stack_frames),
            nn.ReLU(),
            nn.Linear(128 * stack_frames, 64 * stack_frames),
            nn.Linear(64 * stack_frames, 4)
        )

        # Initializing weights
        #self.apply(weights_init)
        #self.critic_linear.weight.data = normalized_columns_initializer(
        #    self.critic_linear.weight.data, 1.0)
        #self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs: torch.Tensor):
        observations, input_inst = inputs
        batch_size = observations[0].size(0)
        #print(batch_size)

        # Get the instruction representation
        encoder_hidden = torch.zeros(1, batch_size, self.gru_hidden_size, requires_grad=True)
        encoder_hidden = encoder_hidden.to(device)
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[:, i]).view(batch_size, 1, -1)
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

        # Gated-attention over all frames
        state_representations = []
        for observation in observations:
            x = F.relu(self.img1(observation))
            x = F.relu(self.img2(x))
            x = F.relu(self.img3(x))
            x = x.view(x.size(0), -1)

            # State representation
            x = F.relu(self.linear(x))
            state_representations.append(x)

        state     = torch.cat(state_representations, dim=-1)
        state     = torch.cat([state, x_instr_rep], dim=-1)
        value     = self.value_function(state)
        advantage = self.advantage_function(state)

        return value + (advantage - advantage.mean(1, keepdim=True))