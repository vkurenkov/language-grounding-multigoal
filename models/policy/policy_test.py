import torch as t

from torch.autograd import Variable
from models.policy import GridPolicy
from models.vision import GridVision
from models.state import GridState
from models.language import InstructionLanguage


def test_grid_policy_shape():
    # Policy settings
    batch_size = 10
    num_actions = 4

    # Instructions language
    vocabulary_size = 20
    max_instruction_len = 20
    lang_out_size = 128
    lang = InstructionLanguage(vocabulary_size, max_instruction_len, lang_out_size)
    
    # Grid vision
    width = 10 * 4
    height = 10
    depth = 5
    vision = GridVision(depth, width, height)

    # State
    state_out_size = 128
    state = GridState(vision, lang, state_out_size)

    # Policy
    policy = GridPolicy(vision, lang, state, num_actions)

    initial_state = state.get_initial_state(batch_size)
    instructions = Variable(t.LongTensor(batch_size, max_instruction_len).random_(0, vocabulary_size))
    images = Variable(t.FloatTensor(batch_size, depth, width, height).random_(0, 1))
    actions_distr, state_value, cur_state = policy.forward(images, instructions, initial_state)

    assert(actions_distr.probs.size() == (batch_size, num_actions))
    assert(state_value.size() == (batch_size, 1))
    assert(cur_state.size() == (batch_size, state_out_size))
