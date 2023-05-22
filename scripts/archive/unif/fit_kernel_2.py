import torch.nn as nn
import torch
import numpy as np
from transport_kernel import  TransportKernel, k_matrix, l_scale, normalize
import matplotlib.pyplot as plt
import random
from ellipse import rand_ellipse
import os
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def sample_normal(N = 100, d = 2):
    mu = np.zeros(d)
    sigma = np.identity(d)
    X_sample = np.random.multivariate_normal(mu, sigma, N)
    X_sample = X_sample.reshape(N,d)
    return X_sample

def unif_square(N = 200):
    o1_vals = np.random.choice([-1.0, 1.0], size = N)
    unifs_vals = np.random.uniform(low = -1, high = 1, size = N)
    o1_indexes = np.random.choice([0, 1], size = N)
    unif_indexes = np.abs(1 - o1_indexes).astype(int)
    samples = np.zeros((N,2))
    for i, sample in enumerate(samples):
        samples[i,o1_indexes[i]] = o1_vals[i]
        samples[i, unif_indexes[i]] = unifs_vals[i]
    return samples

def unif_circle(N = 200):
    theta = np.random.uniform(low = -np.pi, high = np.pi, size = N)
    X = np.cos(theta)
    Y = np.sin(theta)
    sample = np.asarray([[x,y] for x,y in zip(X,Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample


def unif_elipse(N = 1000, a = .5, b = 2):
    X, Y = rand_ellipse(a = a, b = b, size=N)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    return sample


def one_normalize(vec):
    return vec/np.linalg.norm(vec, ord = 1)


def resample(Y, alpha = [], N = 10000):
    n = len(Y.T)
    if not len(alpha):
        alpha = np.full(n, 1/n)
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample


def normal_theta_circle(N = 500):
    thetas = np.linspace(-np.pi, np.pi, N)
    theta_probs = one_normalize(np.exp(-2 * np.abs(thetas)) + .01)
    thetas = thetas.reshape((1,len(thetas)))
    thetas = resample(thetas, theta_probs, N).reshape(thetas.shape[1])
    X = np.cos(thetas)
    Y = np.sin(thetas)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample.T


def sample_2normal(N = 100, d = 2, mu_1 = 1, mu_2 = -1, sigma = .5):
    mu_1 = mu_1 * np.ones(d)
    mu_2 =  mu_2  * np.ones(d)
    sigma =  sigma * np.identity(d)
    X_1 = np.random.multivariate_normal(mu_1, sigma, N)
    X_2 = np.random.multivariate_normal(mu_2, sigma, N)
    X_sample =  np.concatenate([X_1, X_2]).reshape(2*N)
    return X_sample


def sample_uniform(N = 100,  d = 2, l = -1.5, h = 1.5):
    Y = []
    for i in range(d):
        yi = np.random.uniform(l,h, N)
        Y.append(yi)
    Y_sample = np.stack(Y).reshape((N,d))
    return Y_sample


def sample_Y(N = 100, sigma = .1):
    mu = 0
    Sigma = 1
    w = np.random.normal(mu, Sigma, N)
    xi = np.random.normal(mu, sigma, N)
    Y_1 = w**2 + xi
    Y_2 = np.random.normal(mu, Sigma, N)
    Y_sample = np.stack([Y_1, Y_2]).reshape((N,2))
    return Y_sample


def update_list_dict(Dict, update):
    for key, val in update.items():
        Dict[key].append(val)
    return Dict


def train_kernel_transport(kernel_model, n_iter = 100, save_dir = '', d = 1,plt_range = [[-3,3]]):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'total': []}
    kernel_model.train()
    for i in range(n_iter):
        if not i % 10:
            Y_pred = kernel_model.map(kernel_model.X).detach().cpu().numpy().T
            sample_hmap(Y_pred, f'{save_dir}/Y_in_progress{0 if i==0 else ""}.png', d=d, bins=20, range= plt_range)
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not i % kernel_model.params['print_freq']:
            print(f'At step {i}: fit_loss = {round(float(loss_dict["fit"]),4)},'
                  f' reg_loss = {round(float(loss_dict["reg"]),4)}')
            Loss_dict = update_list_dict(Loss_dict, loss_dict)
    return kernel_model, Loss_dict


def train_step(kernel_model, optimizer):
    optimizer.zero_grad()
    loss, loss_dict = kernel_model.loss()
    loss.backward()
    optimizer.step()

    kernel_model.iters += 1
    loss_dict['n_iter'] = kernel_model.iters
    return loss, loss_dict



def sample_hmap(sample, save_loc, bins = 20, d = 2, range = None):
    if d == 2:
        x, y = sample.T
        x = np.asarray(x)
        y = np.asarray(y)
        plt.hist2d(x,y, density=True, bins = bins, range = range)
    elif d == 1:
        x =  sample
        x = np.asarray(x)
        plt.hist(x, bins = bins, range = range)
    plt.savefig(save_loc)
    clear_plt()
    return True


def sample_scatter(sample, save_loc, bins = 20, d = 2, range = []):
    x, y = sample.T
    x = np.asarray(x)
    y = np.asarray(y)
    plt.scatter(x,y)
    
    if len(range):
        x_left, x_right = range[0]
        y_bottom, y_top = range[1]
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
    plt.savefig(save_loc)
    clear_plt()
    return True


def run():
    pass

if __name__=='__main__':
    run()
