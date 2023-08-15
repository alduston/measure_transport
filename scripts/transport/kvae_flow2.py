import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap,seaborne_hmap
import os
from copy import deepcopy
from get_data import sample_banana, sample_normal, mgan2, sample_spirals, sample_pinweel, mgan1, sample_rings, \
    rand_covar, sample_torus, sample_x_torus, sample_sphere, sample_base_mixtures, sample_spheres, sample_swiss_roll

import matplotlib.pyplot as plt
import numpy as np
import random
from lokta_voltera import get_VL_data,sample_VL_prior
from picture_to_dist import sample_elden_ring
from datetime import datetime as dt
from seaborn import kdeplot


def shuffle(tensor):
    if geq_1d(tensor).shape[0] <=1:
        return tensor
    else:
        return tensor[torch.randperm(len(tensor))]


def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor


def replace_zeros(array, eps = 1e-5):
    for i,val in enumerate(array):
        if np.abs(val) < eps:
            array[i] = 1.0
    return array


def normalize(array, keep_axes=[], just_var = False, just_mean = False):
    normal_array = deepcopy(array)
    if len(keep_axes):
        norm_axes = np.asarray([axis for axis in range(len(array.shape)) if (axis not in keep_axes)])
        keep_array = deepcopy(normal_array)[:, keep_axes]
        normal_array = normal_array[:, norm_axes]
    if not just_var:
        normal_array = normal_array - np.mean(normal_array, axis = 0)
    std_vec = replace_zeros(np.std(normal_array, axis = 0))
    if not just_mean:
        normal_array = normal_array/std_vec
    if len(keep_axes):
        normal_array = np.concatenate([normal_array, keep_array], axis = 1)
    return normal_array


def flip_2tensor(tensor):
    Ttensor = 0 * tensor.T
    Ttensor[0] += tensor.T[1]
    Ttensor[1] += tensor.T[0]
    return Ttensor.T


class Comp_transport_model:
    def __init__(self, submodels_params, device = None):
        self.submodel_params = submodels_params
        self.dtype = torch.float32
        self.plot_steps = False

        n = len(self.submodel_params['Lambda_mean'])
        eps = 1e-3
        self.noise_shrink_c = np.exp(np.log(eps)/(n))


        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

    def mmd(self, map_vec, target):
        map_vec = torch.tensor(map_vec, device = self.device, dtype=self.dtype)
        target = torch.tensor(target, device = self.device, dtype=self.dtype)
        return self.submodel_params['mmd_func'](map_vec, target)


    def map_mean(self, x_mu, y_mean, y_var, Lambda_mean, X_mean, fit_kernel):
        x_mean = torch.concat([x_mu, y_mean + y_var], dim=1)
        z_mean = fit_kernel(X_mean, x_mean).T @ Lambda_mean
        return z_mean


    def map_var(self, x_mu, y_eta, y_mean, Lambda_var, X_var, y_var, fit_kernel):
        if not self.approx:
            y_eta = shuffle(y_eta)
        x_var = torch.concat([x_mu, y_eta, y_mean + y_var], dim=1)
        Lambda_var = Lambda_var

        z_var = fit_kernel(X_var, x_var).T @ Lambda_var
        return z_var


    def param_map(self, step_idx, param_dict):
        Lambda_mean = torch.tensor(self.submodel_params['Lambda_mean'][step_idx],
                                   device=self.device, dtype=self.dtype)
        Lambda_var = torch.tensor(self.submodel_params['Lambda_var'][step_idx],
                                  device=self.device, dtype=self.dtype)
        fit_kernel = self.submodel_params['fit_kernel'][step_idx]
        X_mean = torch.tensor(self.submodel_params['X_mean'][step_idx],device=self.device, dtype=self.dtype)
        X_var = torch.tensor(self.submodel_params['X_var'][step_idx], device=self.device, dtype=self.dtype)

        y_eta = geq_1d(torch.tensor(param_dict['y_eta'], device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(param_dict['x_mu'], device=self.device, dtype=self.dtype))
        y_mean = geq_1d(torch.tensor(param_dict['y_mean'], device=self.device, dtype=self.dtype))
        y_var = geq_1d(torch.tensor(param_dict['y_var'], device=self.device, dtype=self.dtype))

        if not self.approx:
            y_mean = deepcopy(y_eta)
            y_var = 0 * y_mean

        z_mean = self.map_mean(x_mu, y_mean, y_var, Lambda_mean, X_mean, fit_kernel)
        z_var = self.map_var(x_mu, y_eta, y_mean, Lambda_var, X_var,  y_var, fit_kernel)
        z = z_mean + z_var

        y_approx = y_mean + y_var
        y_eta = self.noise_shrink_c * shuffle(y_eta)

        param_dict = {'y_eta': y_eta, 'y_mean': y_mean + z_mean, 'y_var': y_var + z_var, 'x_mu': x_mu,
                       'y_approx': y_approx + z, 'y': torch.concat([x_mu, y_approx + z], dim=1)}


        if self.plot_steps:
            save_loc = f'../../data/kernel_transport/spiral_kflow/gen_map{step_idx}.png'
            map_vec = param_dict['y'].detach().cpu().numpy()
            sample_hmap(map_vec, save_loc, bins=75, bw_adjust= 0.25,
                    d=2, range=[[-3, 3], [-3, 3]])
        return param_dict


    def c_map(self, x, y, no_x = False):
        param_dict = {'y_eta': y, 'y_mean': 0 , 'y_var': 0,
                       'x_mu': x, 'y_approx': 0, 'y': 0}
        self.approx = False
        for step_idx in range(len(self.submodel_params['Lambda_mean'])):
            param_dict = self.param_map(step_idx, param_dict)
            self.approx = True
        if no_x:
            return param_dict['y_approx']
        return param_dict['y']


    def map(self, x = [], y = [], no_x = False):
        return self.c_map(x,y, no_x = no_x)



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


        self.Y_mean = deepcopy(self.Y_eta)
        self.Y_var =  0 * self.Y_mean
        self.X_var = torch.concat([self.X_mu, shuffle(self.Y_eta), self.Y_mean + self.Y_var], dim=1)
        self.approx = self.params['approx']
        if self.approx:
            self.Y_mean = geq_1d(torch.tensor(base_params['Y_mean'], device=self.device, dtype=self.dtype))
            self.Y_var = geq_1d(torch.tensor(base_params['Y_var'], device=self.device, dtype=self.dtype))
            self.X_var = torch.concat([self.X_mu, self.Y_eta, self.Y_mean + self.Y_var], dim=1)

        self.X_mean = torch.concat([self.X_mu, self.Y_mean+ self.Y_var], dim=1)
        self.Y_approx = self.Y_mean + self.Y_var

        self.Y_mu = geq_1d(torch.tensor(base_params['Y_mu'], device=self.device, dtype=self.dtype))
        self.Y = torch.concat([self.X_mu, self.Y_mu], dim=1)

        self.Nx = len(self.X_mean)
        self.Ny = len(self.Y)

        self.params['fit_kernel_params']['l'] *= l_scale(self.X_mean).cpu()
        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.Nx, device=self.device, dtype=self.dtype)
        self.fit_kXXmean_inv = torch.linalg.inv(self.fit_kernel(self.X_mean, self.X_mean) + self.nugget_matrix)
        self.fit_kXXvar_inv = torch.linalg.inv(self.fit_kernel(self.X_var, self.X_var) + self.nugget_matrix)

        self.params['mmd_kernel_params']['l'] *= l_scale(self.Y_mu).cpu()
        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)

        self.Z_mean = nn.Parameter(self.init_Z(), requires_grad=True)
        self.Z_var = nn.Parameter(self.init_Z(), requires_grad=True)
        self.mmd_YY = self.mmd_kernel(self.Y, self.Y)

        self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))
        self.Y_mean_test = deepcopy(self.Y_eta_test)
        self.Y_var_test = 0 * self.Y_eta_test

        if self.approx:
            self.Y_mean_test = geq_1d(torch.tensor(base_params['Y_mean_test'], device=self.device, dtype=self.dtype))
            self.Y_var_test = geq_1d(torch.tensor(base_params['Y_var_test'], device=self.device, dtype=self.dtype))

        self.X_mu_test = geq_1d(torch.tensor(base_params['X_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_mu_test = geq_1d(torch.tensor(base_params['Y_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_test = torch.concat([self.X_mu_test, self.Y_mu_test], dim=1)


        self.alpha_z = self.p_vec(self.Nx)
        self.alpha_y = self.p_vec(self.Ny)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y
        self.iters = deepcopy(self.params['iters'])


    def p_vec(self, n):
        return torch.full([n], 1/n, device=self.device, dtype=self.dtype)


    def prob_add(self, t_1, t_2, p = .001):
        T = []
        for i in range(len(t_1)):
            if random.random() < p:
                T.append(t_2[i])
            else:
                T.append(t_1[i])
        return torch.tensor(T, device= self.device).reshape(t_1.shape)


    def init_Z(self):
        Z = torch.zeros(self.Y_mu.shape, device=self.device, dtype=self.dtype)
        return Z


    def get_Lambda_mean(self):
        return self.fit_kXXmean_inv @ self.Z_mean


    def get_Lambda_var(self):
        return self.fit_kXXvar_inv @ self.Z_var


    def map_mean(self, x_mu, y_eta, y_mean, y_var):
        if self.approx:
            y_mean = geq_1d(torch.tensor(y_mean, device=self.device, dtype=self.dtype))
        else:
            y_mean = deepcopy(y_eta)
        x_mean = torch.concat([x_mu, y_mean + y_var], dim=1)
        Lambda_mean = self.get_Lambda_mean()
        z_mean = self.fit_kernel(self.X_mean, x_mean).T @ Lambda_mean
        return z_mean


    def map_var(self, x_mu, y_eta, y_mean, y_var):
        if not self.approx:
            y_eta = shuffle(y_eta)

        y_mean = geq_1d(torch.tensor(y_mean, device=self.device, dtype=self.dtype))

        x_var = torch.concat([x_mu, y_eta, y_mean + y_var], dim=1)
        Lambda_var = self.get_Lambda_var()

        z_var = self.fit_kernel(self.X_var, x_var).T @ Lambda_var
        return z_var


    def map(self, x_mu, y_eta, y_mean = 0, y_var = 0):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        y_mean = geq_1d(torch.tensor(y_mean, device=self.device, dtype=self.dtype))
        y_var = geq_1d(torch.tensor(y_var, device=self.device, dtype=self.dtype))

        if not self.approx:
            y_mean = deepcopy(y_eta)
            y_var = 0 * y_mean

        z_mean = self.map_mean(x_mu, y_eta, y_mean, y_var)
        z_var = self.map_var(x_mu, y_eta, y_mean, y_var)
        z = z_mean + z_var

        y_approx = y_mean + y_var
        y_eta = shuffle(y_eta)
        return_dict = {'y_eta': y_eta, 'y_mean': y_mean + z_mean, 'y_var': y_var + z_var,
                       'y_approx': y_approx + z, 'y': torch.concat([x_mu, z + y_approx], dim = 1)}

        return return_dict


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
        map_vec = torch.concat([self.X_mu, self.Y_approx  + self.Z_mean + self.Z_var], dim=1)
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
        Z_mean = self.Z_mean
        Z_var = self.Z_var * float(self.approx)

        reg_1 = self.params['reg_lambda'] * torch.trace(Z_mean.T @ self.fit_kXXmean_inv @ Z_mean)
        reg_2 = self.params['reg_lambda'] * torch.trace(Z_var.T @ self.fit_kXXvar_inv @ Z_var)
        return reg_1 + reg_2


    def loss_test(self):
        x_mu = self.X_mu_test
        y_eta = self.Y_eta_test
        y_mean = self.Y_mean_test
        y_var = self.Y_var_test
        target = self.Y_test
        map_vec = self.map(x_mu, y_eta, y_mean, y_var)['y']
        return self.mmd(map_vec, target)


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 101,  iters = -1,  approx = False, X_mu_test = [],
                          Y_mean = [],Y_var = [], Y_eta_test = [],Y_mu_test = [], Y_mean_test = [], Y_var_test =[]):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 1e-5, 'Y_mean': Y_mean, 'Y_var': Y_var,
                        'fit_kernel_params': deepcopy(params['mmd']), 'mmd_kernel_params': deepcopy(params['fit']),
                        'print_freq': 100, 'learning_rate': .001, 'nugget': 1e-4, 'Y_eta_test': Y_eta_test,
                        'X_mu_test': X_mu_test, 'Y_mu_test': Y_mu_test, 'Y_mean_test': Y_mean_test,
                        'Y_var_test': Y_var_test, 'iters': iters, 'approx': approx}
    ctransport_kernel = CondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel


def comp_cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 101, n = 50,
                               Y_eta_test = [], X_mu_test = [],Y_mu_test = [], f = 1):
    model_params = {'fit_kernel': [], 'Lambda_mean': [], 'X_mean': [], 'Lambda_var': [], 'X_var': []}
    iters = -1
    eps = 1e-3
    noise_shrink_c = np.exp(np.log(eps)/(n))

    Y_mean = 0
    Y_mean_test = 0
    Y_var = 0
    Y_var_test = 0
    approx = False
    for i in range(n):
        model = cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test, approx =approx,
                                      Y_mean = Y_mean , Y_var = Y_var, X_mu_test = X_mu_test,Y_mu_test = Y_mu_test,
                                      Y_mean_test =  Y_mean_test, Y_var_test = Y_var_test, iters = iters)

        model_params['Lambda_mean'].append(model.get_Lambda_mean().detach().cpu().numpy())
        model_params['Lambda_var'].append(model.get_Lambda_var().detach().cpu().numpy())
        model_params['fit_kernel'].append(model.fit_kernel)
        model_params['X_mean'].append(model.X_mean.detach().cpu().numpy())
        model_params['X_var'].append(model.X_var.detach().cpu().numpy())

        if i==0:
            model_params['mmd_func'] = model.mmd

        n_iter = max(int(n_iter * f), 31)

        map_dict = model.map(model.X_mu, model.Y_eta, model.Y_mean, model.Y_var)
        Y_eta, Y_mean, Y_var = map_dict['y_eta'], map_dict['y_mean'], map_dict['y_var']

        test_map_dict = model.map(model.X_mu_test, model.Y_eta_test, model.Y_mean_test, model.Y_var_test)
        Y_eta_test, Y_mean_test, Y_var_test = test_map_dict['y_eta'], test_map_dict['y_mean'], test_map_dict['y_var']

        Y_eta *= noise_shrink_c
        Y_eta_test *= noise_shrink_c
        iters = model.iters
        approx = True
    return Comp_transport_model(model_params)


def get_idx_tensors(idx_lists):
    return [torch.tensor(idx_list).long() for idx_list in idx_lists]


def zero_pad(array):
    zero_array = np.zeros([len(array),1])
    return np.concatenate([zero_array, array], axis = 1)


def train_cond_transport(ref_gen, target_gen, params, N, n_iter = 101, process_funcs = [],
                         cond_model_trainer = cond_kernel_transport, idx_dict = {},  n_transports = 100):
    ref_sample = ref_gen(N)
    target_sample = target_gen(N)

    N_test = min(10 * N, 7000)
    test_sample = ref_gen(N_test)
    test_target_sample = target_gen(N_test)

    if len(process_funcs):
        forward = process_funcs[0]
        target_sample = forward(target_sample)

    ref_idx_tensors = idx_dict['ref']
    target_idx_tensors = idx_dict['target']
    cond_idx_tensors = idx_dict['cond']
    trained_models = []

    for i in range(len(ref_idx_tensors)):
        X_mu = target_sample[:,  cond_idx_tensors[i]]
        X_mu_test = test_target_sample[:, cond_idx_tensors[i]]

        Y_mu = target_sample[:, target_idx_tensors[i]]
        Y_mu_test = test_target_sample[:, target_idx_tensors[i]]

        Y_eta = ref_sample[:,ref_idx_tensors[i]]
        Y_eta_test = test_sample[:, ref_idx_tensors[i]]

        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test,
                                                 Y_mu_test = Y_mu_test, X_mu_test = X_mu_test, n = n_transports))
    return trained_models


def compositional_gen(trained_models, ref_sample, target_sample, idx_dict):
    ref_indexes = idx_dict['ref']
    cond_indexes = idx_dict['cond']
    target_indexes = idx_dict['target']

    X =  geq_1d(0 * deepcopy(target_sample))
    X[:, cond_indexes[0]] += deepcopy(target_sample)[:, cond_indexes[0]]
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, ref_indexes[i]]
        target_shape =  X[:, target_indexes[i]].shape
        X[:, target_indexes[i]] = model.map(X[:, cond_indexes[i]], Y_eta, no_x = True)\
            .detach().cpu().numpy().reshape(target_shape)

    return X


def sode_hist(trajectories, savedir, save_name = 'traj_hist', n = 4):
    trajectories = torch.tensor(trajectories)
    trajectories = trajectories[:,:n]
    N,n = trajectories.shape
    fig, axs = plt.subplots(n)
    for i in range(n):
        hist_data = trajectories[:, i]
        axs[i].hist(hist_data.detach().cpu().numpy(), label=f't = {i}', bins = 50, range = [-6,6])
    for ax in fig.get_axes():
        ax.label_outer()
    plt.savefig(f'{savedir}/{save_name}.png')
    clear_plt()


def conditional_transport_exp(ref_gen, target_gen, N = 1000, n_iter = 1001, vmax = None,
                           exp_name= 'exp', plt_range = None,  process_funcs = [],
                           cond_model_trainer= comp_cond_kernel_transport,idx_dict = {}, bins = 70,
                           skip_idx = 0, plot_idx = [], plots_hists = False, n_transports = 40):
     save_dir = f'../../data/kernel_transport/{exp_name}'
     try:
         os.mkdir(save_dir)
     except OSError:
         pass

     nr = len(ref_gen(2)[0])

     mmd_params = {'name': 'r_quadratic', 'l': torch.exp(torch.tensor(-1.25)), 'alpha': 1}
     fit_params = {'name': 'r_quadratic', 'l': torch.exp(torch.tensor(-1.25)), 'alpha': 1}
     exp_params = {'fit': mmd_params, 'mmd': fit_params}

     if not len(idx_dict):
         idx_dict = {'ref': [], 'cond': [[]], 'target': []}
         for k in range(nr):
             idx_dict['ref'].append([k])
             idx_dict['cond'].append(list(range(k+1)))
             idx_dict['target'].append([k])


     idx_dict = {key: get_idx_tensors(val) for key,val in idx_dict.items()}
     idx_dict = {key: val[skip_idx:] for key, val in idx_dict.items()}

     trained_models = train_cond_transport(ref_gen, target_gen, exp_params, N, n_iter,
                                           process_funcs, cond_model_trainer,
                                           idx_dict = idx_dict, n_transports = n_transports)
     N_plot = min(10 * N, 10000)
     target_sample = target_gen(N_plot)
     ref_sample = ref_gen(N_plot)

     gen_sample = compositional_gen(trained_models, ref_sample, target_sample, idx_dict)

     test_mmd = trained_models[-1].mmd(gen_sample, target_sample)
     print(f'Test mmd was {test_mmd}')

     if plots_hists:
        sode_hist(gen_sample,save_dir,save_name='marginal_hists')
        sode_hist(target_sample, save_dir, save_name='target_marginal_hists')

     if len(process_funcs):
         backward = process_funcs[1]
         gen_sample = backward(gen_sample.cpu())

     if not len(plot_idx):
        return trained_models, idx_dict
     gen_sample = gen_sample[:, plot_idx]
     target_sample = target_sample[:, plot_idx]
     try:
        d = len(gen_sample[0])
     except TypeError:
         d = 1

     sample_hmap(gen_sample, f'{save_dir}/gen_map_final.png', bins=bins, d=d , range=plt_range, vmax=vmax)
     sample_hmap(target_sample, f'{save_dir}/target_map.png', bins=bins, d=d , range=plt_range, vmax=vmax)

     return trained_models, idx_dict


def two_d_exp(ref_gen, target_gen, N, n_iter=1001, plt_range=None, process_funcs=[], bins = 70,
              slice_vals=[], slice_range=None, exp_name='exp', skip_idx=0, vmax=None, n_transports = 70):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    slice_vals = np.asarray(slice_vals)
    plot_idx = torch.tensor([0, 1]).long()
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=vmax, bins = bins,
                                                         exp_name=exp_name, plt_range=plt_range, n_transports = n_transports,
                                                         plot_idx=plot_idx, process_funcs=process_funcs, skip_idx=skip_idx)
    N_plot = min(10 * N, 10000)
    for slice_val in slice_vals:
        ref_sample = ref_gen(N_plot)
        ref_slice_sample = target_gen(N_plot)
        ref_slice_sample[:, idx_dict['cond'][0]] = slice_val
        slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)
        plt.hist(slice_sample[:, 1], bins=bins, range=slice_range, label = f'x ={slice_val}')
    plt.savefig(f'{save_dir}/slice_posteriors.png')
    clear_plt()
    return True


def spheres_exp(N = 5000, n_iter = 101, exp_name = 'spheres_exp', n_transports = 150):
    n = 10
    ref_gen = lambda N: sample_base_mixtures(N = N, d = 2, n = 2)
    target_gen = lambda N: sample_spheres(N = N, n = n)


    idx_dict = {'ref': [[0,1]],
                'cond': [list(range(2, 2 + (2*n)))],
                'target': [[0,1]]}

    plt_range = [[.5,1.5],[-1.5,1.5]]
    plot_idx = torch.tensor([0, 1]).long()
    skip_idx = 0
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=None, skip_idx=skip_idx,
                               exp_name=exp_name, process_funcs=[],cond_model_trainer=comp_cond_kernel_transport,
                               idx_dict= idx_dict, plot_idx= plot_idx, plt_range = plt_range, n_transports = n_transports)

    N_test =  min(10 * N, 10000)
    slice_vals = np.asarray([[1,.0], [1,.2], [1,.4], [1,.5], [1,.6], [1,.7], [1,.75], [1,.79]])

    save_dir = f'../../data/kernel_transport/{exp_name}'

    for slice_val in slice_vals:
        ref_sample = ref_gen(N_test)
        RX = np.full((N_test,2), slice_val)
        ref_slice_sample = sample_spheres(N = N_test, n = n, RX = RX)

        slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)
        sample_hmap(slice_sample[:,np.asarray([0,1])], f'{save_dir}/x={slice_val[1]}_map.png', bins=60, d=2,
                    range=plt_range)
    return True


def elden_exp(N=10000, n_iter=101, exp_name='elden_exp', n_transports=55):
    ref_gen = sample_normal
    target_gen = sample_elden_ring
    idx_dict = {'ref': [[0, 1]], 'cond': [[]],'target': [[0,1]]}
    skip_idx = 0
    plt_range = [[-1,1],[-1.05,1.15]]
    plot_idx = torch.tensor([0,1]).long()
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=6,
                                                         exp_name=exp_name, process_funcs=[],bins = 75,
                                                         cond_model_trainer=comp_cond_kernel_transport,
                                                         idx_dict=idx_dict, skip_idx=skip_idx, plot_idx=plot_idx,
                                                         plt_range=plt_range, n_transports=n_transports)
    return trained_models


def vl_exp(N=10000, n_iter=101, Yd=18, normal=True, exp_name='vl_exp', n_transports = 100):
    ref_gen = lambda N: sample_normal(N, 4)
    target_gen = lambda N: get_VL_data(N, Yd=Yd, normal=normal, T = 20)

    X_mean = np.asarray([1, 0.0564, 1, 0.0564])
    X_std = np.asarray([0.2836, 0.0009, 0.2836, 0.0009]) ** .5

    idx_dict = {'ref': [[0, 1, 2, 3]],
                'cond': [list(range(4, 4 + Yd))],
                'target': [[0, 1, 2, 3]]}

    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    skip_idx = 0
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=None,
                                                         exp_name=exp_name, process_funcs=[],
                                                         cond_model_trainer=comp_cond_kernel_transport,
                                                         idx_dict=idx_dict, skip_idx=skip_idx, plot_idx=[],
                                                         plt_range=None, n_transports = n_transports)

    N_plot =  min(10 * N, 10000)
    slice_val = np.asarray([.8, .041, 1.07, .04])
    #slice_val = np.asarray([2, .1, 2, .1])

    X = np.full((N_plot, 4), slice_val)
    ref_slice_sample = get_VL_data(10 * N_plot, X=X, Yd=Yd, normal=normal,  T = 20)
    ref_sample = ref_gen(N_plot)

    slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)
    slice_sample[:, :4] *= X_std
    slice_sample[:,:4] += X_mean


    params_keys = ['alpha', 'beta', 'gamma', 'delta']

    ranges1 = {'alpha': [0,1.5], 'beta': [-.06,.33], 'gamma':[.5,1.8], 'delta':[-.06,.33]}
    ranges2 = {'alpha': [.5,1.4], 'beta': [0.02,0.07], 'gamma':[.5,1.5], 'delta':[0.025,0.065]}
    ranges3 = {'alpha': [0, 2.25], 'beta': [-.03, .13], 'gamma': [0, 2.25], 'delta': [-.03, .13]}
    ranges4 = {'alpha': [None, None], 'beta': [None, None], 'gamma': [None, None], 'delta': [None, None]}


    for range_idx,ranges in enumerate([ranges1, ranges2, ranges3, ranges4]):
        for i, key_i in enumerate(params_keys):
            for j, key_j in enumerate(params_keys):
                if i <= j:
                    plt.subplot(4, 4, 1 + (4 * j + i))
                    if not i:
                        plt.ylabel(params_keys[j])
                    if j == 3:
                        plt.xlabel(params_keys[i])

                    if i < j:
                        x, y = slice_sample[:, torch.tensor([i, j]).long()].T
                        plt_range = [ranges[key_i], ranges[key_j]]
                        kdeplot(x=x, y=y, fill=True, bw_adjust=0.25, cmap='Blues')
                        plt.scatter(x=slice_val[i], y=slice_val[j], s=13, color='red')
                        if plt_range[0][0]!= None:
                            plt.xlim(plt_range[0][0], plt_range[0][1])
                            plt.ylim(plt_range[1][0], plt_range[1][1])

                    else:
                        x = slice_sample[:, i]
                        plt_range = ranges[key_i]
                        if plt_range[0] == None:
                            plt_range = None
                        plt.hist(x, bins=50, range = plt_range)
                        plt.axvline(slice_val[i], color='red', linewidth=3)

        plt.tight_layout(pad=0.3)
        plt.savefig(f'../../data/kernel_transport/{exp_name}/posterior_samples{range_idx}.png')
        clear_plt()
    return True


def run():
    ref_gen = mgan2
    target_gen = sample_spirals
    N = 3000
    two_d_exp(ref_gen, target_gen, N, n_iter=101, plt_range=[[-3, 3], [-3, 3]], process_funcs=[], skip_idx=1,
              slice_vals=[0], slice_range=[-3, 3], exp_name='spiral_kflow2', n_transports=200, vmax=.15)




if __name__=='__main__':
    run()