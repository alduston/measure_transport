import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
from transport_kernel import get_kernel
from unif_transport import one_normalize



def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def sample_hmap(sample, save_loc, bins = 20, d = 2, range = None, vmax= None, cmap = None):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    if d == 2:
        x, y = sample.T
        x = np.asarray(x)
        y = np.asarray(y)
        plt.hist2d(x,y, density=True, bins = bins, range = range, cmin = 0, vmin=0, vmax = vmax, cmap = cmap)
        plt.colorbar()
    elif d == 1:
        x =  sample
        x = np.asarray(x)
        plt.hist(x, bins = bins, range = range)
    plt.savefig(save_loc)
    clear_plt()
    return True



class RegressionKernel(nn.Module):
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

        self.Y = torch.tensor(base_params['Y'], device=self.device, dtype=self.dtype)
        self.X = torch.tensor(base_params['Y_unif'], device=self.device, dtype=self.dtype)
        self.N = len(self.X)
        self.N_Y = len(self.Y)

        self.diff_map = self.params['diff_map']
        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        if self.params['use_geo']:
            self.WXX,self.WXY,self.WYY =  self.diff_map(self.X.T.detach().cpu().numpy(),self.Y.T.detach().cpu().numpy())


            self.WXX = torch.tensor(self.WXX, device=self.device, dtype=self.dtype)/np.linalg.norm(self.WXX)
            self.WXY = torch.tensor(self.WXY, device=self.device, dtype=self.dtype)/np.linalg.norm(self.WXY)
            self.WYY = torch.tensor(self.WYY, device=self.device, dtype=self.dtype)/np.linalg.norm(self.WYY)

            self.fit_kXX = self.fit_kernel(self.WXX)
            self.mmd_XX = self.mmd_kernel(self.WXX)
            self.mmd_XY = self.mmd_kernel(self.WXY)
            self.mmd_YY = self.mmd_kernel(self.WYY)

        else:
            self.fit_kXX = self.fit_kernel(self.X, self.X)

            self.mmd_XX = self.mmd_kernel(self.X, self.X)
            self.mmd_XY = self.mmd_kernel(self.X, self.Y)
            self.mmd_YY = self.mmd_kernel(self.Y, self.Y)

        self.alpha_Y = torch.ones(len(self.Y), device = self.device, dtype = self.dtype)/len(self.Y)
        self.E_mmd_YY = self.alpha_Y.T @ self.mmd_YY @ self.alpha_Y

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.N, device=self.device, dtype=self.dtype)

        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)
        self.iters = 0
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.target = torch.ones(self.Z.shape, device = self.device, dtype = self.dtype)
        self.W_inf = torch.tensor(base_params['W_inf'], device = self.device, dtype = self.dtype)


    def init_Z(self):
        return torch.zeros(self.N, device = self.device, dtype = self.dtype)


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def map(self, y, Z_y = []):
        if not len(Z_y):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)
            Lambda = self.get_Lambda()
            Z_y =  (Lambda.T @ self.fit_kernel(self.X, y))
        alpha_y_inv = one_normalize(torch.exp(Z_y).detach().cpu().numpy())
        return alpha_y_inv


    def mmd(self, alpha, X, Y):
        alpha = torch.tensor(alpha, device=self.device, dtype=self.dtype)
        X = torch.tensor(X, device=self.device, dtype=self.dtype)
        Y = torch.tensor(Y, device=self.device, dtype=self.dtype)

        if self.params['use_geo']:
            WXX, WXY, WYY = self.diff_map(Y,X)
            mmd_XX = self.mmd_kernel(WXX)
            mmd_XY = self.mmd_kernel(WXY)
            mmd_YY = self.mmd_kernel(WYY)
        else:
            mmd_XX = self.mmd_kernel(X,X)
            mmd_XY = self.mmd_kernel(X,Y)
            mmd_YY = self.mmd_kernel(Y,Y)

        n_y = len(Y)
        alpha_Y = (1 / n_y) * torch.ones(n_y, device=self.device, dtype=self.dtype)

        Ek_XX = alpha.T @ mmd_XX @ alpha
        Ek_XY = alpha.T @ mmd_XY @ alpha_Y
        Ek_YY = alpha_Y.T @ mmd_YY @ alpha_Y
        return Ek_XX - 2 * Ek_XY + Ek_YY


    def loss_mmd(self):
        alpha = 1/self.N * torch.exp(self.Z)
        c = torch.linalg.norm(alpha,1) ** -1
        Ek_XX = (c**2) * alpha.T @ self.mmd_XX @ alpha
        Ek_XY = c * alpha.T @ self.mmd_XY @  self.alpha_Y
        Ek_YY =  self.E_mmd_YY
        return Ek_XX - 2 * Ek_XY + Ek_YY


    def loss_reg(self):
        Z = self.Z
        return self.params['reg_lambda'] * (Z.T @ self.fit_kXX_inv @ Z)


    def loss_one(self):
        Z = self.Z
        alpha = (1/self.N)*torch.exp(Z)
        return self.params['one_lambda'] * torch.exp(1 + (1 - torch.linalg.norm(alpha, ord = 1))**2)


    def loss(self):
        loss_fit = self.loss_mmd()
        loss_reg  = self.loss_reg()
        loss_one = self.loss_one()
        loss = loss_fit + loss_reg + loss_one
        loss_dict = {'fit': loss_fit.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def run():
    pass