import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True

def l_scale(X):
    return torch.quantile(k_matrix(X,X), q = .25)


def k_matrix(X,X_tilde):
    return  torch.norm(X.unsqueeze(1)-X_tilde, dim=2, p=2)


def radial_kernel(X, X_tilde, kern_params):
    norm_diffs = k_matrix(X, X_tilde)
    sigma = kern_params['sigma']
    l = kern_params['l']
    res =  sigma * torch.exp(-((norm_diffs ** 2) / (2 * (l ** 2))))
    return res


def linear_kernel(X, X_tilde, kern_params):
    sig_b = kern_params['sig_b']
    sig_v = kern_params['sig_v']
    c = kern_params['c']
    return sig_b**2 + (sig_v**2)*torch.matmul(X-c, (X_tilde-c).T)


def poly_kernel(X, X_tilde, kern_params):
    c = kern_params['c']
    alpha = kern_params['alpha']
    return (c + torch.matmul(X, X_tilde.T))**alpha


def geo_kernel(X, X_tilde, kern_params):
    Y = kern_params['Y']
    S_yy = kern_params['S_yy']
    mmd_kernel = kern_params['mmd_kernel']
    P_xy = mmd_kernel(X, Y.T)
    P_x_tilde_y = mmd_kernel(X_tilde, Y.T)
    return  P_xy.T @ S_yy @ P_x_tilde_y


def get_kernel(kernel_params, device, dtype = torch.float32):
    kernel_name = kernel_params['name']
    for key,val in kernel_params.items():
        if key not in  ['name','mmd_kernel','diff_map']:
            kernel_params[key] = torch.tensor(val, device = device, dtype = dtype)

    if kernel_name == 'radial':
        return lambda x,x_tilde: radial_kernel(x,x_tilde, kernel_params)

    elif kernel_name == 'poly':
        return lambda x, x_tilde: poly_kernel(x, x_tilde, kernel_params)

    elif  kernel_name == 'linear':
        return lambda x, x_tilde: linear_kernel(x, x_tilde, kernel_params)

    elif kernel_name == 'geo':
        Y = kernel_params['Y']
        kernel_params['S_yy']= torch.exp(-1 * kernel_params['diff_map'](Y,Y)[0])
        kernel_params['mmd_kernel'] = lambda x,x_tilde: radial_kernel(x,x_tilde, kernel_params)
        return  lambda x, x_tilde: geo_kernel(x, x_tilde, kernel_params)



def normalize(tensor, revar = False):
    m_dim = int(torch.argmax(torch.tensor(tensor.shape)))
    normal_tensor = tensor
    try:
        normal_tensor = normal_tensor - torch.mean(normal_tensor, dim= m_dim)
        if revar:
            normal_tensor = normal_tensor/torch.mean(torch.var(normal_tensor, dim = m_dim))
    except RuntimeError:
        normal_tensor = (normal_tensor.T - torch.mean(normal_tensor, dim= m_dim)).T
        if revar:
            normal_tensor = (normal_tensor.T / torch.mean(torch.var(normal_tensor, dim=m_dim))).T
    return normal_tensor


class TransportKernel(nn.Module):
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

        self.X = torch.tensor(base_params['X'], device=self.device, dtype=self.dtype)
        self.Y = torch.tensor(base_params['Y'], device = self.device, dtype = self.dtype)
        self.N = len(self.X)

        if self.params['normalize']:
            self.X = normalize(self.X)
            self.Y = normalize(self.Y)

        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)
        self.nugget_matrix = self.params['nugget'] * torch.eye(self.N, device=self.device, dtype=self.dtype)

        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)
        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.iters = 0
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.ones = torch.ones(self.X.shape)
        self.mmd_YY_mean = torch.mean(self.mmd_kernel(self.Y, self.Y))


    def init_alpha(self):
        return torch.tensor([0], device = self.device, dtype = self.dtype)


    def init_Z(self):
        return torch.zeros(self.X.shape, device = self.device, dtype = self.dtype)


    def init_Lambda(self):
        return torch.zeros(self.X.shape, device = self.device, dtype = self.dtype)


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def map(self, x):
        x = torch.tensor(x, device=self.device, dtype=self.dtype)
        Lambda = self.get_Lambda()
        res =  (Lambda.T @ self.fit_kernel(self.X, x) + x.T)
        return res


    def loss_mmd(self):
        map_vec = self.Z + self.X
        Y = self.Y
        normalization = self.N / (self.N - 1)

        k_YY_mean = self.mmd_YY_mean
        k_ZZ = self.mmd_kernel(map_vec, map_vec)
        k_ZZ = k_ZZ - torch.diag(torch.diag(k_ZZ))
        k_ZY = self.mmd_kernel(map_vec, Y)
        return normalization * (torch.mean(k_ZZ)) - 2 * torch.mean(k_ZY) + k_YY_mean


    def mmd(self, map_vec, target):
        Y = target
        normalization = self.N/(self.N-1)

        k_YY_mean = torch.mean(self.mmd_kernel(Y,Y))
        k_ZZ = self.mmd_kernel(map_vec, map_vec)
        k_ZZ = k_ZZ - torch.diag(torch.diag(k_ZZ))
        k_ZY =  self.mmd_kernel(Y, map_vec)
        return normalization * (torch.mean(k_ZZ)) - 2 * torch.mean(k_ZY) + k_YY_mean


    def loss_reg(self, Z = []):
        if not len(Z):
            Z = self.Z
        return self.params['reg_lambda'] * torch.trace(Z.T @ self.fit_kXX_inv @ Z)


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg  = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def run():
    pass


if __name__=='__main__':
    run()