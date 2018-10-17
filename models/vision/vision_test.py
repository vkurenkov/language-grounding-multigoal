import torch as t

from torch.autograd import Variable
from models.vision import GridVision


def test_grid_vision_shapes():
    stack_size = 4
    orig_width = 10
    orig_height = 10
    orig_depth = 5
    batch_size = 10

    vision = Variable(t.FloatTensor(batch_size, orig_depth, orig_width * stack_size, orig_height).random_(0, 1))

    grid_vision = GridVision(orig_depth, orig_width * stack_size, orig_height)
    encoded_vision = grid_vision.forward(vision)

    # 250 = (orig_width * orig_height * stack_size * depth) // 8
    assert(encoded_vision.size() == (batch_size, 250))