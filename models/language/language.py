import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Language(nn.Module):
    def __init__(self, vocabulary_size, max_instruction_len):
        super(Language, self).__init__()
        self._info = {"vocabulary_size": vocabulary_size, "max_instruction_len": max_instruction_len}

    def forward(self, cur_instruction):
        '''
        params:
            cur_instruction - current instruction for the agent [batch_size, max_instruction_len]
        return:
            encoded - [batch_size, *]
        '''
        pass


class InstructionLanguage(Language):
    def __init__(self, vocabulary_size, max_instruction_len, size=128):
        super(InstructionLanguage, self).__init__(vocabulary_size, max_instruction_len)

        self.size = size
        self.embed_encoder = nn.Embedding(vocabulary_size, size, padding_idx=0)
        self.rnn_encoder = nn.GRU(size, size, num_layers=1, batch_first=True)

        self.out_size = size

    def forward(self, cur_instruction):
        '''
        params:
            cur_instruction - current instruction for the agent [batch_size, max_instruction_len]
        return:
            rnn_encoding - rnn encoding for each token in the instruction [batch_size, max_instruction_len, size]
        '''
        batch_size = cur_instruction.size(0)
        max_instruction_len = self._info["max_instruction_len"]

        embed_encoding = self.embed_encoder(cur_instruction)
        rnn_encoding = self.rnn_encoder(embed_encoding)[0] # Interested in the entire sequence

        return rnn_encoding


