import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time


def l_scale(X):
    if X.shape[1] > 1:
        return torch.quantile(k_matrix(X,X), q = .25, dim= 0)
    return torch.quantile(k_matrix(X, X), q=.25)
    #return torch.median(k_matrix(X,X))


def k_matrix(X,X_tilde):
    return torch.norm(X.unsqueeze(1) - X_tilde, dim=2, p=1)


def radial_kernel(X, X_tilde, kern_params):
    norm_diffs = k_matrix(X, X_tilde)
    sigma = kern_params['sigma']
    l = kern_params['l']
    return (sigma ** 2) * torch.exp(-norm_diffs ** 2 / (2 * (l ** 2)))


def linear_kernel(X, X_tilde, kern_params):
    sig_b = kern_params['sig_b']
    sig_v = kern_params['sig_v']
    c = kern_params['c']
    return sig_b**2 + (sig_v**2)*torch.matmul(X-c, (X_tilde-c).T)


def poly_kernel(X, X_tilde, kern_params):
    c = kern_params['c']
    alpha = kern_params['alpha']
    return (c + torch.matmul(X, X_tilde.T))**alpha


def get_kernel(kernel_params, device, dtype = torch.float32):
    kernel_name = kernel_params['name']
    for key,val in kernel_params.items():
        if key!= 'name':
            kernel_params[key] = torch.tensor(val, device = device, dtype = dtype)

    if kernel_name == 'radial':
        return lambda x,x_tilde: radial_kernel(x,x_tilde, kernel_params)

    elif kernel_name == 'poly':
        return lambda x, x_tilde: poly_kernel(x, x_tilde, kernel_params)

    elif  kernel_name == 'linear':
        return lambda x, x_tilde: linear_kernel(x, x_tilde, kernel_params)

def normalize(tensor):
    normal_tensor = tensor - torch.mean(tensor, dim=0)
    normal_tensor = normal_tensor/torch.var(normal_tensor, dim = 0)
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

        self.X = normalize(torch.tensor(base_params['X'], device=self.device, dtype = self.dtype))
        self.Y = normalize(torch.tensor(base_params['Y'], device = self.device, dtype = self.dtype))

        self.N = len(self.X)


        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)
        self.nugget_matrix = self.params['nugget'] * torch.eye(self.N ,device=self.device, dtype = self.dtype)

        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)
        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.mmd_kXX = self.mmd_kernel(self.X, self.X)

        self.iters = 0
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)


    def init_alpha(self):
        return torch.tensor([0], device = self.device, dtype = self.dtype)

    def init_Z(self):
        return torch.zeros(self.X.shape)


    def map(self, x):
        x = torch.tensor(x, device=self.device, dtype=self.dtype)
        Lambda =  self.fit_kXX_inv @ self.Z
        return (Lambda.T @ self.fit_kernel(self.X, x) + x.T)


    def loss_fit(self):
        mapp_vec = self.Z + self.X
        k_ZZ = self.mmd_kernel(mapp_vec, mapp_vec)
        k_ZZ = k_ZZ - torch.diag(torch.diag(k_ZZ))
        k_ZY =  self.mmd_kernel(mapp_vec, self.Y)
        k_YY = self.mmd_kernel(self.Y, self.Y)
        normal_factor = self.N/(self.N-1)
        return torch.log(1 + (normal_factor * torch.mean(k_ZZ + k_YY) - 2*torch.mean(k_ZY)))


    def loss_reg(self):
        fit_kXX_inv =  self.fit_kXX_inv
        Z = self.Z
        return self.params['reg_lambda'] * torch.log(1 + torch.trace(Z.T @ fit_kXX_inv @ Z))


    def loss(self):
        loss_fit = self.loss_fit()
        loss_reg  = self.loss_reg()
        loss = loss_fit + loss_reg
        loss_dict = {'fit': loss_fit.detach().cpu(),'reg': loss_reg.detach().cpu(),'total': loss.detach().cpu()}
        return loss, loss_dict


def run():
    pass


if __name__=='__main__':
    run()