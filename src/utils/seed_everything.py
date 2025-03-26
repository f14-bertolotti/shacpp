import random
import numpy
import torch

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
