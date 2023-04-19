import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def radial_kernel(x,x_tilde, l, sigma):
    return (sigma**2) * torch.exp(-torch.linalg.norm(x-x_tilde)**2/(2*(l**2)))


def linear_kernel(x, x_tilde, sig_b, sig_v, c):
    return sig_b**2 + (sig_v**2)*torch.inner(x-c, x_tilde-c)


def poly_kernel(x, x_tilde, l, sigma, alpha):
    inner_val = 1 + (torch.linalg.norm(x-x_tilde)**2)/(2*alpha*(l**2))
    return (sigma**2)*torch.exp(torch.log(inner_val) * -alpha)


def get_kernel(kernel_params, device, dtype = torch.float32):
    kernel_name = kernel_params['name']
    for key,val in kernel_params.items():
        if key!= 'name':
            kernel_params[key] = torch.tensor(val, device = device, dtype = dtype)

    if kernel_name == 'radial':
        sigma = kernel_params['sigma']
        l = kernel_params['l']
        return lambda x,x_tilde: radial_kernel(x,x_tilde, l, sigma)

    elif kernel_name == 'linear':
        sig_b = kernel_params['sig_b']
        sig_v = kernel_params['sig_v']
        c = kernel_params['c']
        return lambda x,x_tilde: linear_kernel(x, x_tilde, sig_b, sig_v, c)


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
        self.X = torch.tensor(base_params['X'], device=self.device, dtype = self.dtype)
        self.Y = torch.tensor(base_params['Y'], device = self.device, dtype = self.dtype)
        self.N = len(self.X)
        self.nugget = self.params['nugget']

        self.fit_kernel = self.get_fit_kernel()
        self.fit_kXX = self.get_kXX(self.fit_kernel)
        nugget_matrix = self.nugget * torch.eye(self.N ,device=self.device, dtype = self.dtype)
        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + nugget_matrix)

        self.mmd_kernel = self.get_mmd_kernel()
        self.mmd_kXX = self.get_kXX(self.mmd_kernel)

        self.iters = 0
        #self.Lambda = nn.Parameter(self.init_Lambda(), requires_grad=True)
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)


    def get_fit_kernel(self):
        kernel_params = self.params['fit_kernel_params']
        return get_kernel(kernel_params, device = self.device, dtype= self.dtype)


    def get_mmd_kernel(self):
        kernel_params = self.params['mmd_kernel_params']
        return get_kernel(kernel_params, device = self.device, dtype= self.dtype)


    def init_Z(self):
        return deepcopy(self.Y)


    def init_Lambda(self):
        N,d = self.X.shape
        Lambda_0 = torch.zeros(N,d, device = self.device, dtype= self.dtype)
        return nn.init.normal_(Lambda_0)


    def get_kX1X2(self, kernel, X_1, X_2):
        N_1 = len(X_1)
        N_2 = len(X_2)
        k_X1X2 = torch.zeros((N_1,N_2), device = self.device, dtype= self.dtype)
        for i,xi in enumerate(X_1):
            for j,xj in enumerate(X_2):
                k_X1X2[i,j] += kernel(xi, xj)
        return k_X1X2

    def get_kXX(self, kernel):
        X = self.X
        return self.get_kX1X2(kernel, X, X)


    def get_kXx(self, kernel, X_tilde):
        X = self.X
        return self.get_kX1X2(kernel, X, X_tilde)


    def map(self, x):
        x = torch.tensor(x, device = self.device, dtype = self.dtype)
        return self.get_kXx(self.fit_kernel, x) @ self.Lambda

    def map_z(self, x):
        Lambda = self.Z @ self.self.fit_kXX_inv
        return self.get_kXx(self.fit_kernel, x) @ Lambda


    def loss_fit(self):
        fit_kXX = self.fit_kXX
        mmd_kXX = self.mmd_kXX
        Y = self.Y
        Lambda  = self.Lambda
        diff_vec =  fit_kXX @ Lambda - Y
        return torch.trace(diff_vec.T @ mmd_kXX @ diff_vec)


    def loss_fit_z(self):
        k_ZZ = self.get_kX1X2(self.mmd_kernel, self.Z, self.Z)
        k_ZY = self.get_kX1X2(self.mmd_kernel, self.Z, self.Y)
        return torch.linalg.norm(k_ZZ)**2 - 2*(torch.linalg.norm(k_ZY)**2)


    def loss_fit2(self):
        Y = self.Y
        Lambda  = self.Lambda
        fit_kXX = self.fit_kXX
        N = self.N
        mmd_kernel = self.mmd_kernel

        fit_loss = 0
        V = fit_kXX @ Lambda

        for i in range(N):
            for j in range(N):
                v_i = V[i]
                v_j = V[j]
                y_j = Y[j]
                fit_loss += mmd_kernel(v_i,v_j) - 2*mmd_kernel(v_i, y_j)
        return fit_loss


    def loss_reg(self):
        fit_kXX = self.fit_kXX
        Lambda = self.Lambda
        return self.params['reg_lambda'] * torch.trace(Lambda.T @ fit_kXX @ Lambda)


    def loss_reg_z(self):
        fit_kXX_inv =  self.fit_kXX_inv
        Z = self.Z
        return self.params['reg_lambda'] * torch.trace(Z.T @ fit_kXX_inv @ Z)


    def loss(self):
        loss_fit = self.loss_fit()
        loss_reg  = self.loss_reg()

        loss = self.loss_fit2() + self.loss_reg()
        loss_dict = {'fit': loss_fit.detach().cpu(), 'reg': loss_reg.detach().cpu(), 'total': loss.detach().cpu()}
        return loss, loss_dict

    def loss_z(self):
        loss_fit = self.loss_fit_z()
        loss_reg  = self.loss_reg_z()

        loss = loss_fit + loss_reg
        loss_dict = {'fit': loss_fit.detach().cpu(), 'reg': loss_reg.detach().cpu(), 'total': loss.detach().cpu()}
        return loss, loss_dict


def run():
    pass


if __name__=='__main__':
    run()