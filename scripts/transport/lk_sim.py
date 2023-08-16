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
    return X.astype(float)


def derivative(X, t, alpha, beta, gamma, delta):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])


def run_ode(params, T = 10, n = 10, X0 = np.asarray([30,1]), obs_std = np.sqrt(1e-5)):
    t_vec = np.linspace(0,T, num = n)
    alpha, beta, delta, gamma = params
    res = integrate.odeint(derivative, X0, t_vec, args=(alpha, beta, delta, gamma))[1:]
    x, y = res.T
    plt.plot(x+ np.random.random(x.shape), color = 'red')
    plt.plot(y+np.random.random(y.shape), color='red')

    res_vec = np.zeros(len(x)+len(y))
    res_vec[::2]+=x
    res_vec[1::2]+=y
    res_vec = np.abs(res_vec)
    yobs = np.array([lognorm.rvs(scale=x, s=obs_std) for x in res_vec])
    return yobs


def get_VL_data(N, X = [], normal = True, T = 20, Yd = 18, X0 = np.asarray([1,1])):
    if not len(X):
        X = sample_VL_prior(N).astype(float)
    Y = np.asarray([run_ode(x, T = T, X0 = X0) for x in X], dtype=float)
    X = np.asarray(X, dtype=float)
    if normal:
        X_mean = np.asarray([1, 0.0564, 1, 0.0564], dtype=float)
        X_var = np.asarray([0.2836, 0.0009, 0.2836, 0.0009],dtype=float) ** .5
        X -= X_mean
        X /= X_var
        Y = normalize(Y)[:, :Yd]
    Y = Y[:, :18]
    return np.concatenate([X,Y], axis = 1)



def run():
    #lv_data = get_VL_data(500)
   # mu = np.mean(lv_data, axis = 0)
    #sigma = np.std(lv_data, axis = 0)

    #colors = ['red']#, 'blue', 'green']
    #x = sample_VL_prior(1)[0]
    #X = np.asarray([x for i in range(5)])
    #get_VL_data(N=10, X=X)
    #plt.savefig('y_vecs.png')

    data = get_VL_data(N=10)
    print(data.shape)

    '''
    for i,x in enumerate(param_vecs):
        X = np.asarray([x for i in range(3)])
        param_data = (get_VL_data(N = 10, X = X))
        prey_data = param_data[:, 4:][:, ::2]
        predator_data = param_data[:, 4:][:, ::2]

        for v_prey, v_predator in zip(prey_data, predator_data):
            plt.plot(v_prey, color =  colors[i])
            plt.plot(v_predator, color =  colors[i])

    plt.savefig('y_vecs.png')
    '''


if __name__=='__main__':
    run()