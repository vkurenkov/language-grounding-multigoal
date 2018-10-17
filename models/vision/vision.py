import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Vision(nn.Module):
    def __init__(self, depth, width, height):
        super(Vision, self).__init__()
        self._info = {"depth": depth, "width": width, "height": height}

    def forward(self, cur_views):
        '''
        params:
            cur_views - current visual perception of the agent [batch_size, depth, width, height]
        return:
            encoded - [batch_size, *]
        '''
        raise NotImplementedError()


class GridVision(Vision):
    def __init__(self, depth, width, height):
        super(GridVision, self).__init__(depth, width, height)
        
        start_size = (width * height * depth)

        self.linear1 = nn.Linear(start_size, start_size // 2)
        self.linear2 = nn.Linear(start_size // 2, start_size // 4)
        self.linear3 = nn.Linear(start_size // 4, start_size // 8)

        self.out_size = start_size // 8

    def forward(self, cur_views):
        '''
        params:
            cur_views - current visual perception of the grid by the agent [batch_size, depth, width, height]
        return:
            encoded - [batch_size, (depth * width * height) // 8]
        '''
        batch_size = cur_views.size(0)

        lin1 = F.selu(self.linear1(cur_views.view(batch_size, -1)))
        lin2 = F.selu(self.linear2(lin1))
        lin3 = F.selu(self.linear3(lin2))

        return lin3