import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time

def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1,1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor


def t_one_normalize(vec):
    return vec/torch.linalg.norm(vec, ord = 1)


def t_resample(Y, alpha = [], N =None):
    n = len(Y.T)
    if N == None:
        N = n
    if not len(alpha):
        alpha = torch.full(n, 1/n)
    alpha = alpha.detach().cpu().numpy()
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True

def l_scale(X):
    return torch.quantile(k_matrix(X[:3300],X[:3300]), q = .25)


def k_matrix(X,X_tilde,  D_inv = []):
    diff_tensor = X.unsqueeze(1) - X_tilde
    if len(X.shape) == 1:
        diff_tensor = diff_tensor.reshape(list(diff_tensor.shape) + [1])
    if len(D_inv):
        return torch.sqrt(diff_tensor.T @ D_inv @ diff_tensor)
    return torch.norm( diff_tensor, dim=2, p=2)



def radial_kernel(X, X_tilde, kern_params, diff_matrix = []):
    if not len(diff_matrix):
        diff_matrix = k_matrix(X, X_tilde)
    sigma = kern_params['sigma']
    l = kern_params['l']
    res =  sigma * torch.exp(-((diff_matrix ** 2) / (2 * (l ** 2))))
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


def laplace_kernel(X, X_tilde, kern_params, diff_matrix = []):
    if not len(diff_matrix):
        diff_matrix = k_matrix(X, X_tilde)
    sigma = kern_params['sigma']
    l = kern_params['l']
    res = sigma * torch.exp(-(diff_matrix / (l ** 2)))
    return res


def r_quadratic_kernel(X, X_tilde, kern_params, diff_matrix = []):
    if not len(diff_matrix):
        diff_matrix = k_matrix(X, X_tilde)
    alpha = kern_params['alpha']
    l = kern_params['l']
    denom = 2*alpha*(l**2)
    res = (1 + (diff_matrix**2)/(denom))**(-alpha)
    return res



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

    elif  kernel_name == 'laplace':
        return lambda x, x_tilde: laplace_kernel(x, x_tilde, kernel_params)

    elif kernel_name == 'r_quadratic':
        return lambda x, x_tilde: r_quadratic_kernel(x, x_tilde, kernel_params)



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

        self.test = False
        if 'Y_eta_test' in base_params.keys():
            self.test = True
            self.Y_eta_test = torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype)

        self.N = len(self.X)
        self.n = len(self.Y)


        if self.params['normalize']:
            self.X = normalize(self.X)
            self.Y = normalize(self.Y)

        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)
        self.fit_kYY = self.fit_kernel(self.X, self.X)

        self.Xnugget_matrix = self.params['nugget'] * torch.eye(self.N, device=self.device, dtype=self.dtype)
        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.Xnugget_matrix)

        self.Ynugget_matrix = self.params['nugget'] * torch.eye(self.N, device=self.device, dtype=self.dtype)
        self.fit_kYY_inv = torch.linalg.inv(self.fit_kYY+ self.Ynugget_matrix)

        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.iters = 0
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.mmd_YY = self.mmd_kernel(self.Y, self.Y)


        self.alpha_x = self.p_vec(self.N)
        self.alpha_y =self.p_vec(self.n)

        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y


    def p_vec(self, n):
        return torch.full([n], 1 / n, device=self.device, dtype=self.dtype)


    def init_Z(self):
        return torch.zeros(self.X.shape, device = self.device, dtype = self.dtype)

    def init_alpha_x(self):
        return torch.zeros(self.alpha_u.shape , device=self.device, dtype=self.dtype)


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def map(self, x):
        x = torch.tensor(x, device=self.device, dtype=self.dtype)
        Lambda = self.get_Lambda()
        res =  self.fit_kernel(self.X, x).T @ Lambda + x
        return res


    def loss_mmd_resample(self):
        alpha_x = self.alpha_x
        alpha_x_p = 1/self.N * torch.exp(-alpha_x)
        c = torch.linalg.norm(alpha_x_p, ord = 1)**-1

        map_vec = self.Z + self.X
        Y = self.Y
        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, Y)

        alpha_y = self.alpha_y
        Ek_ZZ = (c**2) * alpha_x_p @ mmd_ZZ @ alpha_x_p
        Ek_ZY = c * alpha_x_p @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        return Ek_ZZ - 2 * Ek_ZY + Ek_YY


    def loss_mmd(self):
        map_vec = self.Z + self.X
        Y = self.Y
        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, Y)

        alpha_y = self.alpha_y
        alpha_u = self.alpha_x

        Ek_ZZ = alpha_u @ mmd_ZZ @ alpha_u
        Ek_ZY = alpha_u @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        return Ek_ZZ - 2 * Ek_ZY + Ek_YY


    def mmd(self, map_vec, target):
        Y = target.reshape(len(target))
        map_vec = map_vec.reshape(len(map_vec))

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, Y)
        mmd_YY = self.mmd_kernel(Y, Y)

        alpha_y = self.p_vec(len(target))
        alpha_z = self.p_vec(len(map_vec))

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = alpha_y @ mmd_YY @ alpha_y
        return Ek_ZZ - 2 * Ek_ZY + Ek_YY


    def loss_reg(self, Z = []):
        if not len(Z):
            Z = self.Z
            if len(Z.shape)==1:
                Z = Z.reshape(len(Z),1)
        return self.params['reg_lambda'] * torch.trace(Z.T @ self.fit_kXX_inv @ Z)


    def loss_reg_alpha(self, alpha=[]):
        if not len(alpha):
            alpha_x = self.alpha_x
        return self.params['reg_lambda_alpha'] * alpha_x.T @ self.fit_kXX_inv @ alpha_x


    def loss_test(self):
        y_eta = self.Y_eta_test
        target = self.Y
        map_vec = self.map(y_eta)
        plt.hist(map_vec.detach().cpu().numpy(), bins=40, range=[-1, 5])
        plt.savefig('output_hist.png')
        clear_plt()
        if self.iters < 50:
            plt.hist(target.detach().cpu().numpy(), bins=40, range=[-1, 5])
            plt.savefig('target_hist.png')
            clear_plt()
        return self.mmd(map_vec, target)


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