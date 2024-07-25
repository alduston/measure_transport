import torch
import torch.nn as nn
import os
from copy import deepcopy,copy
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime as dt


def l_func(X,y,gamma=4):
    return np.exp(-gamma * (np.linalg.norm(X-y, axis = 1)**2))


def normalize(V, ord = 1):
    return V/np.linalg.norm(V, ord=ord)


def resample(X, alpha = [], N = 10000):
    n = len(X)
    if not len(alpha):
        alpha = np.full(n, 1/n)

    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=normalize(alpha))
    X_resample = X[resample_indexes, :]
    return X_resample


