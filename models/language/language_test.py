import torch as t

from torch.autograd import Variable
from models.language import InstructionLanguage


def test_instruction_language_shapes():
    vocabulary_size = 5
    max_instruction_len = 20
    out_size = 128
    batch_size = 10

    instructions = Variable(t.LongTensor(batch_size, max_instruction_len).random_(0, vocabulary_size))
    instruction_language = InstructionLanguage(vocabulary_size, max_instruction_len, out_size)
    instructions_encoded = instruction_language.forward(instructions)

    assert(instructions_encoded.size() == (batch_size, max_instruction_len, out_size))