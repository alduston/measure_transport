import numpy as np
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
import os


def prior_prob(x,r):
    if r > 1.25 or r < .75:
        return 0
    if x > .8*r or x < -.8*r:
        return 0
    return 1/(.8 * r)


def get_v(x,r,n = 20):
    thetas = np.linspace(start=0, stop=2 * np.pi, num=n + 1)[:-1]
    r = np.sqrt(r - (x ** 2))
    z = r * np.sin(thetas)
    y = r * np.cos(thetas)
    v = np.zeros(2 * len(y))
    v[::2] += z
    v[1::2] += y
    return v


def Log_likehood(Y, v, cov):
    residuals = Y - v
    def calc_loglikelihood(residuals):
        return -0.5 * (np.log(np.linalg.det(cov)) + residuals.T.dot(np.linalg.inv(cov)).dot(residuals)
                       + 2 * np.log(2 * np.pi))
    loglikelihood = np.apply_along_axis(calc_loglikelihood, 1, residuals)
    loglikelihoodsum = loglikelihood.sum()
    return loglikelihoodsum


def get_log_likelehood(Y, x, r, n = 20, scale_eps = .1):
    covar = scale_eps * np.indentity(n)
    v = get_v(x,r,n)
    log_prior = np.log(prior_prob(x,r))
    log_likelehood = log_prior +  Log_likehood(Y, v, covar)
    return log_likelehood








