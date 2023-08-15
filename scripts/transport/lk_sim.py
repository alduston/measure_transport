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
        normal_array = np.concatenate([normal_array, keep_array], axis = 1)
    return normal_array


def sample_VL_prior(N):
    LV = DeterministicLotkaVolterra(T = 20)
    X = LV.sample_prior(N)
    return X


def derivative(X, t, alpha, beta, gamma, delta):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])


def run_ode(params, T = 20, n = 10, X0 = np.asarray([30,1]), obs_std = np.sqrt(1e-1)):
    t_vec = np.linspace(0,T, num = n)
    alpha, beta, delta, gamma = params
    res = integrate.odeint(derivative, X0, t_vec, args=(alpha, beta, delta, gamma))[1:]
    x, y = res.T
    res_vec = np.zeros(len(x)+len(y))
    res_vec[::2]+=x
    res_vec[1::2]+=y
    res_vec = np.abs(res_vec)
    yobs = np.array([lognorm.rvs(scale=x, s=obs_std) for x in res_vec])
    return yobs


def get_VL_data(N, X = [], normal = True, T = 20, Yd = 18):
    if not len(X):
        X = sample_VL_prior(N)
    Y = np.asarray([run_ode(x, T = T) for x in X])
    if normal:
        X_mean = np.asarray([1, 0.0564, 1, 0.0564])
        X_var = np.asarray([0.2836, 0.0009, 0.2836, 0.0009]) ** .5
        X -= X_mean
        X /= X_var
        Y = normalize(Y)[:, :Yd]
    Y = Y[::18]
    return np.concatenate([X,Y], axis = 1)



def run():
    x = np.asarray([1,1,1,1])
    X = np.asarray([x for i in range(10)])
    lv_data = get_VL_data(10, X = X)
    for xy in lv_data:
        y = xy[4:]
        prey_data = y[::2]
        predator_data = y[1::2]
        plt.plot(prey_data)
        plt.plot(predator_data)
    plt.savefig('y_vecs.png')


if __name__=='__main__':
    run()