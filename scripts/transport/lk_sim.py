import numpy as np
from scipy.integrate import odeint
from scipy.stats import lognorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import pandas as pd
from scipy import integrate


class DeterministicLotkaVolterra:
    def __init__(self, T):
        # number of unknown (prior) parameters
        self.d = 4
        # prior parameters
        self.alpha_mu = -0.125
        self.alpha_std = 0.5
        self.beta_mu = -3
        self.beta_std = 0.5
        self.gamma_mu = -0.125
        self.gamma_std = 0.5
        self.delta_mu = -3
        self.delta_std = 0.5
        # initial condition
        self.x0 = [30, 1];
        # length of integration window
        self.T = T
        # observation parameters
        self.obs_std = np.sqrt(0.1)

    def sample_prior(self, N):
        # generate Normal samples
        alpha = lognorm.rvs(scale=np.exp(self.alpha_mu), s=self.alpha_std, size=(N,))
        beta = lognorm.rvs(scale=np.exp(self.beta_mu), s=self.beta_std, size=(N,))
        gamma = lognorm.rvs(scale=np.exp(self.gamma_mu), s=self.gamma_std, size=(N,))
        delta = lognorm.rvs(scale=np.exp(self.delta_mu), s=self.delta_std, size=(N,))
        # join samples
        return np.vstack((alpha, beta, gamma, delta)).T

    def ode_rhs(self, z, t, theta):
        # extract parameters
        alpha, beta, gamma, delta = theta
        # compute RHS of
        fz1 = alpha * z[0] - beta * z[0] * z[1]
        fz2 = -gamma * z[1] + delta * z[0] * z[1]
        return np.array([fz1, fz2])

    def simulate_ode(self, theta, tt):
        # check dimension of theta
        assert (theta.size == self.d)
        # numerically intergate ODE
        return odeint(self.ode_rhs, self.x0, tt, args=(theta,))

    def sample_data(self, theta):
        # check inputs
        if len(theta.shape) == 1:
            theta = theta[np.newaxis, :]
        assert (theta.shape[1] == self.d)
        # define observation locations
        tt = np.arange(0, self.T, step=2)
        nt = 2 * (len(tt) - 1)
        # define arrays to store results
        xt = np.zeros((theta.shape[0], nt))
        # run ODE for each parameter value
        for j in range(theta.shape[0]):
            yobs = self.simulate_ode(theta[j, :], tt);
            # extract observations, flatten, and add noise
            yobs = np.abs(yobs[1:, :]).ravel()
            yobs = np.array([lognorm.rvs(scale=x, s=self.obs_std) for x in yobs])
            xt[j, :] = yobs
        return (xt, tt)


    def log_prior_pdf(self, theta):
        # check dimensions of inputs
        assert (theta.shape[1] == self.d)
        # compute mean and variance
        prior_mean = [self.alpha_mu, self.beta_mu, self.gamma_mu, self.delta_mu]
        prior_std = [self.alpha_std, self.beta_std, self.gamma_std, self.delta_std]
        # evaluate product of PDFs for independent variables
        return np.sum(lognorm.logpdf(theta, scale=np.exp(prior_mean), s=prior_std), axis=1)

    def prior_pdf(self, theta):
        return np.exp(self.log_prior_pdf(theta))

    def log_likelihood(self, theta, yobs):
        # check dimension of inputs
        assert (theta.shape[1] == self.d)
        assert (yobs.size == (self.T - 2))
        # define observation locations
        tt = np.arange(0, self.T, step=2)
        # define array to store log-likelihood
        loglik = np.zeros(theta.shape[0], )
        # simulate dynamics for each theta
        for j in range(theta.shape[0]):
            xt = self.simulate_ode(theta[j, :], tt)
            xt = np.abs(xt[1:, :]).ravel()
            # compare observations under LogNormal(G(theta),obs_var)
            loglik[j] = np.sum([lognorm.logpdf(yobs, scale=xt, s=self.obs_std)])
        return loglik


    def likelihood(self, theta, yobs):
        return np.exp(self.log_likelihood(theta, yobs))


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