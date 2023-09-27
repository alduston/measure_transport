import numpy as np
from scipy.integrate import odeint
from scipy.stats import lognorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import pandas as pd
from scipy import integrate
from lokta_voltera import DeterministicLotkaVolterra



def replace_zeros(array, eps = 1e-5):
    for i,val in enumerate(array):
        if np.abs(val) < eps:
            array[i] = 1.0
    return array


def normalize(array, keep_axes=[], just_var = False, just_mean = False):
    normal_array = deepcopy(array)
    if len(keep_axes):
        norm_axes = np.asarray([axis for axis in range(len(array.shape)) if (axis not in keep_axes)])
        keep_array = deepcopy(normal_array)[:, keep_axes]
        normal_array = normal_array[:, norm_axes]
    if not just_var:
        normal_array = normal_array - np.mean(normal_array, axis = 0)
    std_vec = replace_zeros(np.std(normal_array, axis = 0))
    if not just_mean:
        normal_array = normal_array/std_vec
    if len(keep_axes):
        normal_array = np.concatenate([normal_array, keep_array], axis = 0)
    return normal_array


def sample_VL_prior(N):
    LV = DeterministicLotkaVolterra(T = 20)
    X = LV.sample_prior(N)
    LV.sample_data(X)
    return X.astype(float)


def derivative(X, t, alpha, beta, gamma, delta):
    x,y = X
    dotx = x * (alpha - (beta * y))
    doty = y * ((delta * x) - gamma)
    return np.array([dotx, doty])



def run_ode(params, T = 20, n = 10, X0 = np.asarray([30,1]), obs_std = np.sqrt(.1)):
    t_vec = np.linspace(0,T, num = n + 1)[:-1]
    alpha, beta, gamma, delta = params
    res = integrate.odeint(derivative, X0, t_vec, args=(alpha, beta, gamma, delta))
    x, y = res[1:].T
    res_vec = np.zeros(len(x)+len(y))
    res_vec[::2]+=x
    res_vec[1::2]+=y
    res_vec = np.abs(res_vec)

    yobs = np.array([lognorm.rvs(scale=x, s=obs_std) for x in res_vec])
    return yobs



def get_VL_data(N, X = [], normal = True, T = 20, Yd = 18, X0 = np.asarray([30,1]), n = 10):
    if not len(X):
        X = sample_VL_prior(N).astype(float)
    Y = np.asarray([run_ode(x, T = T, X0 = X0, n = n) for x in X], dtype=float)
    X = np.asarray(X, dtype=float)
    XY = np.concatenate([X,Y], axis = 1)
    if normal:
        XY = normalize(XY)
    return XY[:, :4 + Yd]



def run():
   get_VL_data(1)


   pass


if __name__=='__main__':
    run()