import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap
import os
from copy import deepcopy
from get_data import sample_banana, sample_normal, mgan2, sample_spirals, sample_pinweel
from K_VAE import VAETransportKernel
import matplotlib.pyplot as plt
import numpy as np
import random


def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1,1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor

def flip_2tensor(tensor):
    Ttensor = torch.zeros(tensor.T.shape)
    Ttensor[0] += tensor.T[1]
    Ttensor[1] += tensor.T[0]
    return Ttensor.T


class Comp_transport_model:
    def __init__(self, submodels, cond = False, device = None):
        self.submodels = submodels
        self.cond = cond
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

    def base_map(self, z):
        for submodel in self.submodels:
            z = submodel.map(z)
        return z

    def c_map(self, x, z):
        x = geq_1d(torch.tensor(x, device = self.device))
        z = geq_1d(torch.tensor(z, device = self.device))
        for submodel in self.submodels:
            z = submodel.map(x,z, no_x = True)
        return torch.concat([x, z], dim = 1)

    def map(self, z , x = []):
        if self.cond:
            return self.c_map(x,z)
        return self.base_map(z)



class CondTransportKernel(nn.Module):
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
        self.params = base_params
        base_params['device'] = self.device

        self.Y_eta = geq_1d(torch.tensor(base_params['Y_eta'], device=self.device, dtype=self.dtype))
        self.X_mu =  geq_1d(torch.tensor(base_params['X_mu'], device=self.device, dtype=self.dtype))
        self.X = torch.concat([self.X_mu, self.Y_eta], dim=1)
        self.Nx = len(self.X)

        self.Y_mu = geq_1d(torch.tensor(base_params['Y_mu'], device=self.device, dtype=self.dtype))
        self.Y = torch.concat([self.X_mu, self.Y_mu], dim=1)
        self.Ny = len(self.Y)

        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.Nx, device=self.device, dtype=self.dtype)
        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)

        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.mmd_YY = self.mmd_kernel(self.Y, self.Y)

        self.test = False
        if 'Y_eta_test' in base_params.keys():
            self.test = True
            self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))
            self.X_mu_test = geq_1d(torch.tensor(base_params['X_mu_test'], device=self.device, dtype=self.dtype))
            self.Y_mu_test = geq_1d(torch.tensor(base_params['Y_mu_test'], device=self.device, dtype=self.dtype))
            self.Y_test = torch.concat([self.X_mu_test, self.Y_mu_test], dim=1)

        self.alpha_z = self.p_vec(self.Nx)
        self.alpha_y = self.p_vec(self.Ny)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y
        self.iters = 0


    def p_vec(self, n):
        return torch.full([n], 1/n, device=self.device, dtype=self.dtype)


    def init_Z(self):
        return torch.zeros(self.Y_mu.shape, device=self.device, dtype=self.dtype)


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def expand(self, tensor, N):
        n = len(tensor)
        indexes = torch.tensor(np.random.choice(list(range(n)), size = N), device = self.device).long()
        return tensor[indexes]


    def map(self, x_mu, y_eta, no_x = False):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        w = torch.concat([x_mu, y_eta], dim=1)
        Lambda = self.get_Lambda()
        z = self.fit_kernel(self.X, w).T @ Lambda
        if no_x:
            return z + y_eta
        return torch.concat([x_mu, z + y_eta], dim = 1)


    def mmd(self, map_vec, target):
        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, target)
        mmd_YY = self.mmd_kernel(target, target)

        alpha_z = self.p_vec(len(map_vec))
        alpha_y = self.p_vec(len(target))

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = alpha_y @ mmd_YY @ alpha_y

        return Ek_ZZ - (2 * Ek_ZY) + Ek_YY


    def loss_mmd(self):
        map_vec = torch.concat([self.X_mu, self.Y_eta + self.Z], dim=1)
        target = self.Y

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, target)

        alpha_z = self.alpha_z
        alpha_y = self.alpha_y

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        return Ek_ZZ - (2 * Ek_ZY) + Ek_YY


    def loss_reg(self):
        Z = self.Z
        return  self.params['reg_lambda'] * torch.trace(Z.T @ self.fit_kXX_inv @ Z)


    def prob_add(self, t_1, t_2, p = .001):
        T = []
        for i in range(len(t_1)):
            if random.random() < p:
                T.append(t_2[i])
            else:
                T.append(t_1[i])
        return torch.tensor(T, device= self.device).reshape(t_1.shape)


    def loss_test(self):
        x_mu = self.X_mu_test
        y_eta = self.Y_eta_test
        target = self.Y_test
        map_vec = self.map(x_mu, y_eta)
        return self.mmd(map_vec, target)


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def base_kernel_transport(Y_eta, Y_mu, params, n_iter = 1001, Y_eta_test = []):
    transport_params = {'X': Y_eta, 'Y': Y_mu, 'reg_lambda': 1e-5,'normalize': False,
                   'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                   'print_freq':  100, 'learning_rate': .01, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=n_iter)
    return transport_kernel


def comp_base_kernel_transport(Y_eta, Y_mu, params, n_iter = 1001, Y_eta_test = [], n =3, f = .5):
    models = []
    for i in range(n):
        model = base_kernel_transport(Y_eta, Y_mu, params, n_iter, Y_eta_test)
        n_iter = int(n_iter * f)
        Y_eta = model.map(model.X)
        Y_eta_test = model.map(model.Y_eta_test)
        models.append(model)
    return Comp_transport_model(models, cond=False)


def cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 10001, Y_eta_test = [], X_mu_test = [],
                          Y_mu_test = []):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 1e-5,
                        'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                        'print_freq': 100, 'learning_rate': .01, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    if len(X_mu_test):
        transport_params['X_mu_test'] = X_mu_test
    if len(Y_mu_test):
        transport_params['Y_mu_test'] = Y_mu_test

    ctransport_kernel = CondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel



def comp_cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 1001, Y_eta_test = [],
                               X_mu_test = [],Y_mu_test = [], n = 3, f = .5):
    models = []
    for i in range(n):
        model = cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test,
                                                    X_mu_test = X_mu_test, Y_mu_test = Y_mu_test)
        n_iter = int(n_iter * f)
        Y_eta = model.map(model.X_mu, model.Y_eta, no_x = True)
        Y_eta_test = model.map(model.X_mu_test, model.Y_eta_test, no_x = True)
        models.append(model)
    return Comp_transport_model(models, cond=True)



def train_cond_transport(ref_gen, target_gen, params, N = 1000, n_iter = 1001,process_funcs = [],
                         base_model_trainer = base_kernel_transport, cond_model_trainer = cond_kernel_transport):

    ref_sample = ref_gen(N)
    target_sample = target_gen(N)

    test_sample = ref_gen(5 * N)
    test_target_sample = target_gen(5 * N)

    trained_models = []

    if len(process_funcs):
        forward = process_funcs[0]
        target_sample = forward(target_sample)


    Y_eta = ref_sample[:, 0]
    Y_eta_test = test_sample[:, 0]
    Y_mu = target_sample[:, 0]
    trained_models.append(base_model_trainer(Y_eta, Y_mu, params, n_iter, Y_eta_test))

    for i in range(1, len(target_sample[0])):
        X_mu = target_sample[:, :i]
        X_mu_test = test_target_sample[:, :i]

        Y_mu = target_sample[:, i]
        Y_mu_test = test_target_sample[:, i]

        Y_eta = ref_sample[:,i]
        Y_eta_test = test_sample[:, i]

        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test,
                                                 Y_mu_test = Y_mu_test, X_mu_test = X_mu_test))

    return trained_models


def compositional_gen(trained_models, ref_sample):
    ref_sample = geq_1d(ref_sample)
    Y_eta =  ref_sample[:, 0]
    base_model = trained_models[0]
    X = base_model.map(Y_eta)

    for i in range(1, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, i]
        X = model.map(X, Y_eta)
    return X


def conditional_gen(trained_models, ref_sample, cond_sample):
    X = geq_1d(cond_sample)
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, i]
        X = model.map(X, Y_eta)
    return X



def comp_gen_exp(ref_gen, target_gen, N = 1000, n_iter = 1001, exp_name= 'exp', plt_range = None, vmax = None):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    l = l_scale(torch.tensor(ref_gen(N)[:, 1]))
    mmd_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    fit_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    exp_params = {'fit': mmd_params, 'mmd': fit_params}

    Y_eta = ref_gen(N)
    Y_eta_test = ref_gen(5 * N)
    Y_mu = target_gen(N)

    comp_model = comp_base_kernel_transport(Y_eta, Y_mu, exp_params, n_iter, Y_eta_test=Y_eta_test, n=3, f=.5)

    Y_eta_plot = ref_gen(25 * N)
    Y_mu_plot = target_gen(25 * N)
    gen_sample = comp_model.map(torch.tensor(Y_eta_plot, device=comp_model.device))


    sample_scatter(gen_sample, f'{save_dir}/gen_scatter.png', bins=25, d=2, range=plt_range)
    sample_hmap(gen_sample, f'{save_dir}/gen_map.png', bins=50, d=2, range=plt_range, vmax= vmax)

    sample_scatter(Y_mu_plot, f'{save_dir}/target.png', bins=25, d=2, range=plt_range)
    sample_hmap(Y_mu_plot, f'{save_dir}/target_map.png', bins=50, d=2, range=plt_range, vmax= vmax)

    return True



def conditional_transport_exp(ref_gen, target_gen, N = 1000, n_iter = 1001, slice_vals = [],
                           exp_name= 'exp', plt_range = None, slice_range = None, process_funcs = []):
     save_dir = f'../../data/kernel_transport/{exp_name}'
     try:
         os.mkdir(save_dir)
     except OSError:
         pass

     l = l_scale(torch.tensor(ref_gen(N)[:, 1]))
     mmd_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
     fit_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
     exp_params = {'fit': mmd_params, 'mmd': fit_params}

     trained_models = train_cond_transport(ref_gen, target_gen, exp_params, N, n_iter, process_funcs
                                           ,base_model_trainer=comp_base_kernel_transport
                                           ,cond_model_trainer=comp_cond_kernel_transport)

     gen_sample = compositional_gen(trained_models, ref_gen(10 * N))

     if len(slice_vals):
         for slice_val in slice_vals:
             ref_slice_sample = torch.full([N],  slice_val, device = trained_models[0].device)
             slice_sample = conditional_gen([trained_models[-1]], ref_gen(N), ref_slice_sample)
             plt.hist(slice_sample[:, 1].detach().cpu().numpy(), label = f'z  = {slice_val}', bins = 60, range=slice_range)
         plt.legend()
         plt.savefig(f'{save_dir}/conditional_hists.png')
         clear_plt()

     if len(process_funcs):
         backward = process_funcs[1]
         gen_sample = backward(gen_sample.cpu())

     d = len(gen_sample[0])
     if d <=2:
         sample_scatter(gen_sample, f'{save_dir}/gen_scatter.png', bins=25, d = d, range = plt_range)
         sample_scatter(target_gen(10 * N), f'{save_dir}/target.png', bins=25, d=d, range=plt_range)
     return True

#003641

#Adam:
#0.000937
#0.000562

def run():
    ref_gen = sample_normal
    target_gen = sample_spirals

    range = [[-3,3],[-3,3]]

    conditional_transport_exp(ref_gen, target_gen, N=1000, n_iter=101, slice_vals=[0],
                              exp_name='spiral_composed', plt_range=range, slice_range=[-3,3], process_funcs=[])

    #slice_range = [-2.5,2.5]
    #process_funcs = []
    #process_funcs = [flip_2tensor, flip_2tensor]
    #conditional_transport_exp(ref_gen, target_gen, exp_name= 'mgan2_composed', N = 2000, n_iter = 8000,
                              #plt_range=range, slice_range= slice_range, process_funcs=process_funcs,
                              #slice_vals=[-1.1, 0, 1.1])


if __name__=='__main__':
    run()







