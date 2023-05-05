import torch.nn as nn
import torch
import numpy as np
from transport_kernel import TransportKernel, get_kernel, k_matrix, l_scale, normalize
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


def sample_2normal(N = 100, d = 2):
    mu_1 = 1 * np.ones(d)
    mu_2 =  -1 * np.ones(d)
    sigma =  .1 * np.identity(d)
    X_1 = np.random.multivariate_normal(mu_1, sigma, N)
    X_2 = np.random.multivariate_normal(mu_2, sigma, N)
    X_sample =  np.stack([X_1, X_2]).reshape((2*N))
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


def train_kernel_transport(kernel_model, n_iters = 100):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'var': [], 'total': []}
    kernel_model.train()
    for i in range(n_iters):
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not i % kernel_model.params['print_freq']:
            print(f'At step {i}: fit_loss = {round(float(loss_dict["fit"]),2)}, reg_loss = {round(float(loss_dict["reg"]),2)}')
            Loss_dict = update_list_dict(Loss_dict, loss_dict)
        if not i % 10:
            Y_pred = kernel_model.Z.detach().cpu().numpy() + kernel_model.X.detach().cpu().numpy()
            sample_hmap(Y_pred,'../data/Y_in_progress.png', d = 1, range = (-3,3))

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


def get_U_y(Y, U):
    norm_diffs = k_matrix(Y,U)
    min_idxs  = torch.argmin(norm_diffs,  dim=1)
    U_y = U[min_idxs]
    return U_y


def run():
    X = sample_normal(N = 10000, d = 1)
    Y = sample_uniform(N = 1000, d = 1)
    #X_tilde = sample_normal(10000, d= 1)

    X = torch.tes
    U_y = get_U_y(torch.tensor(Y), torch.tensor(X))
    sample_hmap(U_y.T, '../data/Y_u.png', d=1, bins=20, range=(-3, 3))



    #XY = torch.concat([normalize(torch.tensor(X)),normalize(torch.tensor(Y))])
    #l = l_scale(XY)



    #sample_hmap(X, '../data/Xtrain.png', d = 1)
    #sample_hmap(Y, '../data/Ytrain.png', d = 1)

    #fit_kernel_params = {'name': 'radial', 'l':  l, 'sigma': 1}
    #mmd_kernel_params = {'name': 'radial', 'l': l, 'sigma': 1}


    #model_params = {'X': X, 'Y': Y, 'fit_kernel_params': fit_kernel_params, 'mmd_kernel_params': mmd_kernel_params,
                    #'reg_lambda': 5e-2, 'print_freq': 1, 'learning_rate':.05, 'nugget': .3, 'X_tilde': X_tilde}

    #kernel_model = TransportKernel(model_params)
    #train_kernel_transport(kernel_model)

    #Y_tilde = kernel_model.map(X_tilde).detach().cpu().numpy()
    #sample_hmap(Y_tilde.T, '../data/Ypred.png', d=1, bins=20, range=(-3, 3))


if __name__=='__main__':
    run()
