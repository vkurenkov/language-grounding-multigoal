import torch
import numpy
import random


def fix_random_seeds(seed: int) -> None:
    """
    Use in the beginning of the program only.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    numpy.random.seed(seed)
    random.seed(seed)