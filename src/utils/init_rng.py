import random
import numpy as np
import torch
from torch.backends import cudnn



def init_rng(seed: int) -> None:
    '''
        Primes all sorts of random number generator seeds.

        Args:
            seed (int): random seed to prime RNGs with.
    '''
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True
