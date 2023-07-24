import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from transport_kernel import get_kernel, normalize, TransportKernel, l_scale, clear_plt
from fit_kernel import train_kernel,sample_scatter, sample_hmap
import os
from get_data import sample_normal, sample_banana, mgan2, sample_swiss_roll
from picture_to_dist import sample_elden_ring
import time as TIME



class VAETransportKernel(nn.Module):
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

        self.eps = self.get_eps(self.X)


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
        nx = len(self.X[0])
        self.t_idx = torch.tril_indices(row=nx, col=nx, offset=0)

        self.alpha_x = (1/self.N) * torch.ones(self.N, device = self.device, dtype = self.dtype)

        self.mmd_YY = self.mmd_kernel(self.Y, self.Y)
        self.alpha_y = (1 / self.n) * torch.ones(self.n, device=self.device, dtype=self.dtype)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y


    def init_Z(self):
        n = len(self.X[0])
        N = len(self.X)
        #m =  int(n + (n *(n+1))//2)
        return torch.randn([N, n], device=self.device, dtype=self.dtype)
        #return torch.randn([N,2 *n], device = self.device, dtype = self.dtype)


    def v_to_lt(self, V, n = 0, t_idx = []):
        N = len(V)
        if not n:
            n = V.shape[1]
        if not len(t_idx):
            t_idx = torch.tril_indices(row=n, col=n, offset=0)
        m = torch.zeros((N, n, n), device = self.device, dtype = self.dtype)
        m[:, t_idx[0], t_idx[1]] = V
        return m


    def get_mu_sig(self, Z = []):
        n = len(self.X[0])
        if not len(Z):
            Z = self.Z
        mu = Z[:, :n]
        return mu, mu

        #sig = Z[:, n:]**2
        #return mu, sig

        #sig_vs = Z[:, n:]

        #t_idx = self.t_idx
        #sig_ltms = self.v_to_lt(sig_vs,n,t_idx)
        #sig_ltms_T = torch.transpose(sig_ltms,1,2)

        #sig_ms = torch.matmul(sig_ltms, sig_ltms_T)
        #return mu, sig_ms
        #return mu, sig


    def get_sample(self, params = {}):
        if not len(params):
            mu,sig = self.get_mu_sig()
            params = {'mu': mu, 'sig': sig, 'eps': self.eps}

        #eps = torch.unsqueeze(self.get_eps(self.X),2) #torch.unsqueeze(params['eps'],2)
        #eps = self.get_eps(self.X)
        #diffs = torch.matmul(params['sig'], eps)
        Z_sample = params['mu'] #+ params['sig'] * eps #diffs.reshape(diffs.shape[:-1])
        return Z_sample


    def get_Lambda(self):
        mu,sig = self.get_mu_sig()
        return self.fit_kXX_inv @ mu
        #return self.fit_kXX_inv @ self.Z


    def get_eps(self, x):
        eps_shape = list(x.shape)
        return torch.randn(eps_shape, device=self.device, dtype=self.dtype)


    def map(self, x, just_mu = False):
        x = torch.tensor(x, device=self.device, dtype=self.dtype)
        Lambda = self.get_Lambda()
        z =  self.fit_kernel(self.X, x) @ Lambda
        return z + x
        mu, sig = self.get_mu_sig(z)
        if just_mu:
            return mu + x
        eps = self.get_eps(x)
        z_sample = self.get_sample({'mu': mu, 'sig': sig, 'eps': eps})
        return z_sample + x


    def loss_mmd(self):
        map_vec = self.get_sample() + self.X

        Y = self.Y
        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, Y)

        alpha_y = self.alpha_y
        alpha_z = self.alpha_x

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY

        return Ek_ZZ - 2 * Ek_ZY + Ek_YY


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
            mu, sig = self.get_mu_sig(Z)
            if len(Z.shape)==1:
                Z = Z.reshape(len(Z),1)
        if len(mu.shape) == 1:
            mu = mu.reshape(len(mu),1)
            #sig = sig.reshape(len(sig),1)
        mu_reg = torch.trace(mu.T @ self.fit_kXX_inv @ mu)
        #sig_reg = torch.trace(sig.T @ self.fit_kXX_inv @ sig)
        return self.params['reg_lambda'] * (mu_reg)


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg  = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def VAE_transport_exp(ref_gen, target_gen, N, params, t_iter = 801, exp_name= 'exp', plt_range = None):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    ref_sample = torch.tensor(ref_gen(N))
    test_sample = torch.tensor(ref_gen(N))
    target_sample = torch.tensor(target_gen(N)).T

    if target_sample.shape[0] != max(target_sample.shape):
        target_sample = target_sample.T

    transport_params = {'X': ref_sample, 'Y': target_sample, 'fit_kernel_params': params['fit'],
                        'mmd_kernel_params': params['mmd'], 'normalize': False,
                        'reg_lambda': 1e-3, 'print_freq': 100, 'learning_rate': .1,
                        'nugget': 1e-4, 'X_tilde': test_sample }
                         #,'alpha_x': [], 'alpha_y': []}

    VAET_kernel = VAETransportKernel(transport_params)
    #VAET_kernel = TransportKernel(transport_params)
    train_kernel(VAET_kernel, n_iter=t_iter)

    gen_sample_mu = VAET_kernel.map(test_sample, just_mu = True)
    gen_sample = VAET_kernel.map(test_sample) #VAET_kernel.get_sample() + VAET_kernel.X #
    #gen_sample = VAET_kernel.Z +  VAET_kernel.X

    sample_scatter(gen_sample_mu, f'{save_dir}/cond_sample_mean.png', bins=25, d=2, range=plt_range)
    sample_hmap(gen_sample_mu, f'{save_dir}/cond_sample_mean_map.png', bins=25, d=2, range=plt_range)

    sample_scatter(gen_sample, f'{save_dir}/cond_sample.png', bins=25, d=2, range=plt_range)
    sample_hmap(gen_sample, f'{save_dir}/cond_sample_map.png', bins=25, d=2, range=plt_range)

    sample_scatter(target_sample, f'{save_dir}/target_sample.png', bins=25, d=2, range=plt_range)
    sample_hmap(target_sample, f'{save_dir}/target_sample_map.png', bins=25, d=2, range=plt_range)

    test_mmd = float(VAET_kernel.mmd(target_sample.to(VAET_kernel.device), gen_sample.to(VAET_kernel.device)).detach().cpu())
    print(f'test_mmd was {test_mmd}')


def run():
    ref_gen = sample_normal
    target_gen = sample_swiss_roll

    l = l_scale(torch.tensor(ref_gen(2000)))

    mmd_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    fit_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    exp_params = {'fit': mmd_params, 'mmd': fit_params}

    range = [[-3, 3], [-3, 3]]

    VAE_transport_exp(ref_gen, target_gen, N=1000, t_iter=3000,
                              exp_name='swiss_VAE_exp', params=exp_params, plt_range=range)

#At step 9900: fit_loss = 0.000112, reg_loss = 0.006806

if __name__=='__main__':
    run()