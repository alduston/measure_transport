import torch.nn as nn
import torch
import numpy as np
from transport_kernel import TransportKernel, get_kernel
import matplotlib.pyplot as plt
import random


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


def sample_uniform(N = 100,  d = 2, l = 0, h = 1):
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


def train_kernel_transport(kernel_model, n_iters = 400):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'total': []}
    kernel_model.train()
    for i in range(n_iters):
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not i % kernel_model.params['print_freq']:
            print(f'At step {i}: fit_loss = {round(float(loss_dict["fit"]),2)}, reg_loss = {round(float(loss_dict["reg"]),2)}')
            Loss_dict = update_list_dict(Loss_dict, loss_dict)
        if not i % 5:
            sample_hmap(kernel_model.Z.detach().T,'../data/Y_tilde_hmap.png', d = 1)

    return kernel_model, Loss_dict


def train_step(kernel_model, optimizer):
    optimizer.zero_grad()
    #loss, loss_dict = kernel_model.loss()
    loss, loss_dict = kernel_model.loss_z()
    loss.backward()
    optimizer.step()

    kernel_model.iters += 1
    loss_dict['n_iter'] = kernel_model.iters
    return loss, loss_dict


def sample_hmap(sample, save_loc, bins = 10, d = 2):
    if d == 2:
        x, y = sample.T
        plt.hist2d(x,y, density=True, bins = bins)
    elif d == 1:
        x =  sample
        plt.hist(x, bins = bins)
    plt.savefig(save_loc)
    clear_plt()
    return True


def run():
    X = sample_uniform(N = 1000, d = 1)
    Y = sample_normal(N = 1000, d = 1)

    sample_hmap(X, '../data/X_hmap.png', d = 1)
    sample_hmap(Y, '../data/Y_hmap.png', d = 1)

    fit_kernel_params = {'name': 'radial', 'l': .3, 'sigma': 1}
    mmd_kernel_params = {'name': 'radial', 'l': .02, 'sigma': 1}

    model_params = {'X': X, 'Y': Y, 'fit_kernel_params': fit_kernel_params, 'mmd_kernel_params': mmd_kernel_params,
                    'reg_lambda': 1, 'print_freq': 1, 'learning_rate':.15, 'nugget': .01}

    kernel_model = TransportKernel(model_params)
    train_kernel_transport(kernel_model)

    X_tilde = sample_uniform(1000, d = 1)
    Y_tilde = kernel_model.map_z(X_tilde).detach().cpu().numpy()
    sample_hmap(Y_tilde.T, '../data/Y_tilde_hmap_2.png', d = 1)





if __name__=='__main__':
    run()
