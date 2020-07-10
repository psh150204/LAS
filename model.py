import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()

    def forward(self, x):
        pass


class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()

    def forward(self, x):
        pass
