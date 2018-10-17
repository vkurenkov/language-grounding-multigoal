import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from models.utils import is_cuda


class State(nn.Module):
    def __init_(self):
        super(State, self).__init__()

    def forward(self, cur_state, cur_vision, cur_language):
        '''
        params:
            cur_state - current state of the agent [batch_size, *]
            cur_vision - encoded visual modality [batch_size, *]
            cur_language - encoded language modality [batch_size, *]
        return:
            encoded - encoded current state [batch_size, *]
        '''
        pass


class GridState(State):
    def __init__(self, vision, language, size=128):
        super(GridState, self).__init__()

        self.out_size = size
        self._vision_out_size = vision.out_size
        self._language_out_size = language.out_size
        self._goal_space_size = self._language_out_size // 2
        
        # Current state to goal space (cur_state -> goal)
        self.state_goal_lin1 = nn.Linear(self.out_size, self._language_out_size)
        self.state_goal_lin2 = nn.Linear(self._language_out_size, self._goal_space_size)

        # Language to goal space (language -> goal)
        self.lang_goal_lin1 = nn.Linear(self._language_out_size, self._language_out_size)
        self.lang_goal_lin2 = nn.Linear(self._language_out_size, self._goal_space_size)

        # State encoder
        self.rnn = nn.GRUCell(self._vision_out_size, self.out_size)

    def forward(self, prev_state, cur_vision, cur_language):
        '''
        params:
            prev_state - previous state of the agent [batch_size, 128]
            cur_vision - encoded visual modality [batch_size, (depth * width * height) // 8]
            cur_language - encoded language modality [batch_size, max_instruction_len, size]
        returns:
            cur_state - current state of the agent (in terms of vision)
            cur_goal - current goal of the agent (embedding)
        '''
        cur_state = self.rnn.forward(cur_vision, prev_state)
        cur_goal = self.infere_current_goal(cur_state, cur_language)
        
        return cur_state, cur_goal

    def infere_current_goal(self, cur_state, language):
        '''
        params:
            cur_state - should have enough information to produce new goal
                        [batch_size, state_size]
            language - language modality [batch_size, length, language_embed_size]
        returns:
            cur_goal - [batch_size, goal_space_size]
        '''
        batch_size = cur_state.size(0)
        goal_space_size = self._goal_space_size
        length = language.size(1)

        state_goal = self.state_to_goal_space(cur_state).view(batch_size, goal_space_size, 1) # [batch_size, goal_space_size, 1]
        lang_goals = self.lang_to_goal_space(language).view(batch_size, length, goal_space_size) # [batch_size, length, goal_space_size]

        self.attention = t.bmm(lang_goals, state_goal) # [batch_size, length, 1]
        weighted_goals = lang_goals * self.attention # [batch_size, length, goal_space_size]
        return t.sum(weighted_goals, dim=1) # [batch_size, goal_space_size] summation over the weighted goals

    def state_to_goal_space(self, state):
        return self.state_goal_lin2(F.selu(self.state_goal_lin1(state)))

    def lang_to_goal_space(self, language):
        '''
        params:
            language - [batch_size, length, language_embedding_size]
        returns:
            lang_goal_space - [batch_size, length, goal_space_size]
        '''
        batch_size = language.size(0)
        seq_len = language.size(1)

        lang_goal_space = Variable(t.zeros(batch_size, seq_len, self._goal_space_size))
        if is_cuda(self):
            lang_goal_space.cuda()

        for i in range(seq_len):
            lang_goal_space[:, i, :] = self.lang_goal_lin2(F.selu(self.lang_goal_lin1(language[:, i, :])))

        return lang_goal_space

    def get_initial_state(self, batch_size):
        initial_state = Variable(t.ones(batch_size, self.out_size))
        if is_cuda(self):
            initial_state.cuda()
        return initial_state
