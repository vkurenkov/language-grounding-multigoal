import numpy as np
from torch.nn.init import xavier_normal


def is_cuda(module):
    return next(module.parameters()).is_cuda


def to_device(module, device):
    if str.startswith(device, "gpu"):
        return module.cuda()
    else:
        return module.cpu()


def xavier_initialization(module):
    for p in module.parameters():
        if len(p.size()) > 1:
            xavier_normal(p)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def num_learnable_parameters(module):
    module_parameters = filter(lambda p: p.requires_grad, module.parameters())
    return sum([np.prod(p.size()) for p in module_parameters])