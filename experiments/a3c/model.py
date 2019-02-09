import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class A3C_LSTM_GA(nn.Module):
    def __init__(self, input_size: int, max_episode_length: int):
        super(A3C_LSTM_GA, self).__init__()

        # Image Processing (batch_size, 4, 10, 10)
        self.img1 = nn.Conv2d(in_channels=4, out_channels=128, kernel_size=3, stride=1)
        self.img2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1)
        self.img3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        # 2304 features

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                max_episode_length+1,
                self.time_emb_dim)

        # A3C layers
        self.linear = nn.Linear(2304 + self.gru_hidden_size, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs: torch.Tensor):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.img1(x))
        x = F.relu(self.img2(x))
        x = F.relu(self.img3(x))

        # Get the instruction representation
        encoder_hidden = torch.zeros(1, 1, self.gru_hidden_size, requires_grad=True)
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[0, i]).view(1, 1, -1)
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

        x = torch.cat([x.view(x.size(0), -1), x_instr_rep], dim=-1)

        # A3C
        x = F.relu(self.linear(x))
        hx, cx = torch.zeros(1, 256), torch.zeros(1, 256)#self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)