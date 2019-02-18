import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from experiments.dqn_ga_pe.parameters import device
from typing import Tuple


def normalized_columns_initializer(weights: torch.Tensor, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def prepare_model_input(observation, instruction_indices, step) -> Tuple[torch.Tensor, torch.Tensor]:
	return (
        torch.tensor(observation).unsqueeze(0).to(device),
        torch.tensor(instruction_indices, dtype=torch.int64).to(device),
        torch.tensor(step, dtype=torch.int64).unsqueeze(0).to(device)
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # pe - [length, d_model]
        
    def forward(self, x, steps):
        """
        x - [batch_size, embed_len]
        step - [batch_size, 1]
        """
        additive = torch.index_select(self.pe, dim=0, index=steps.view(-1))
        return x + additive

class Model(nn.Module):
    def __init__(self, input_size: int, max_episode_length: int):
        super(Model, self).__init__()

        # Image Processing (batch_size, 4, 10, 10)
        self.img1 = nn.Conv2d(in_channels=4, out_channels=128, kernel_size=3, stride=1)
        self.img2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1)
        self.img3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        # 2304 features

        self.positional_embed = PositionalEncoding(2304, max_episode_length)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size, batch_first=True)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)

        # Critic
        self.linear = nn.Linear(2304, 256)
        self.critic_linear = nn.Linear(256, 4)

        # Initializing weights
        self.apply(weights_init)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs: torch.Tensor):
        x, input_inst, steps = inputs
        batch_size = x.size(0)

        # Get the image representation
        x = F.relu(self.img1(x))
        x = F.relu(self.img2(x))
        x_image_rep = F.relu(self.img3(x))

        # Get the instruction representation
        encoder_hidden = torch.zeros(1, batch_size, self.gru_hidden_size, requires_grad=True)
        encoder_hidden = encoder_hidden.to(device)
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[:, i]).view(batch_size, 1, -1)
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

        # Get the attention vector from the instruction representation
        x_attention = F.sigmoid(self.attn_linear(x_instr_rep))
        x_attention = x_attention.unsqueeze(2).unsqueeze(3)
        x_attention = x_attention.expand(batch_size, 64, 6, 6)

        # Gated-Attention
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep*x_attention
        x = x.view(x.size(0), -1)

        # Add timing (or positional encoding)
        x = self.positional_embed.forward(x, steps)

        # State representation
        x = F.relu(self.linear(x))

        return self.critic_linear(x)