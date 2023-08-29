import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CT_GAN(nn.Module):
    def __init__(self, base_params, device=None):
        super().__init__()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        self.dtype = torch.float32
        base_params['device'] = self.device
        self.params = base_params


    def Divergence(self):
        pass


    def Generator(self):
        pass


    def Discriminator(self):
        pass


    def Reg_penalty(self):
        pass


    def Loss(self):
        pass


def Data_Class(params):
    def __init__(self, data={}, params = {}, device = None):
        pass