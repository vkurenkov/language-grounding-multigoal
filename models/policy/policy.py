import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Categorical
from models.utils import is_cuda
from models.utils import xavier_initialization


class GridPolicy(nn.Module):
    def __init__(self, vision, language, state, num_actions=4):
        '''
        vision - a module to encode visual modality
        language - a module to encode language modality
        state - a module to encode current state
        '''
        super(GridPolicy, self).__init__()

        self._vision = vision
        self._language = language
        self._state = state

        self._input_size = self._state.out_size + self._state._goal_space_size
        self._num_actions = num_actions

        self.action_probs = nn.Sequential(
            nn.Linear(self._input_size, self._input_size // 4),
            nn.SELU(),

            nn.Linear(self._input_size // 4, self._input_size // 8),
            nn.SELU(),

            nn.Linear(self._input_size // 8, num_actions),
            nn.Softmax(dim=-1)
        )

        self.state_value = nn.Sequential(
            nn.Linear(self._input_size, self._input_size // 2),
            nn.SELU(),

            nn.Linear(self._input_size // 2, self._input_size // 4),
            nn.SELU(),

            nn.Linear(self._input_size // 4, 1)
        )

        # Feature creator (must be freezed)
        # Feature predictor

        # Smart-initialization
        xavier_initialization(self)

    def forward(self, cur_views, cur_instructions, prev_state):
        '''
        params:
            cur_views - current visual perception of the agent [batch_size, depth, width, height]
            cur_instructions - current instruction for the agent [batch_size, max_instruction_len]
            prev_state - previous state of the agent [batch_size, *]
        return:
            action_distribution (actor) - a distribution over the set of actions
            state_value (critic) - an estimated value for current state
            cur_state - current state [batch_size, state_embedding_size]
        '''
        cur_state, cur_goal = self._state.forward(prev_state, self._vision.forward(cur_views), self._language.forward(cur_instructions))

        action_distribution = Categorical(self.action_probs(t.cat([cur_state, cur_goal], dim=-1)))
        state_value = self.state_value(t.cat([cur_state, cur_goal], dim=-1))

        return action_distribution, state_value, cur_state

    def get_initial_state(self, batch_size):
        return self._state.get_initial_state(batch_size)