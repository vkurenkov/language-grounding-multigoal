import torch as t

from torch.autograd import Variable
from models.state import GridState
from models.vision import GridVision
from models.language import InstructionLanguage


def test_grid_state_shapes():
    # State batch size
    batch_size = 10

    # Instructions language
    vocabulary_size = 5
    max_instruction_len = 20
    lang_out_size = 128
    instructions = Variable(t.LongTensor(batch_size, max_instruction_len).random_(0, vocabulary_size))
    lang = InstructionLanguage(vocabulary_size, max_instruction_len, lang_out_size)
    instructions_encoded = lang.forward(instructions) # Expected out size is [10, 20, 128]
    
    # Grid vision
    width = 10 * 4
    height = 10
    depth = 5
    vision = GridVision(depth, width, height)
    images = Variable(t.FloatTensor(batch_size, depth, width, height).random_(0, 1))
    vision_encoded = vision.forward(images) # Expected out size is 250

    # State
    state_out_size = 128
    state = GridState(vision, lang, state_out_size)

    initial_state = state.get_initial_state(batch_size)
    cur_state, cur_goal = state.forward(initial_state, vision_encoded, instructions_encoded)

    assert(cur_state.size() == (batch_size, state_out_size)) # (10, 128)
    assert(cur_goal.size() == (batch_size, lang_out_size // 2)) # (10, 64)