import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap,seaborne_hmap
import os
from copy import deepcopy,copy
from get_data import sample_banana, sample_normal, mgan2, sample_spirals, sample_pinweel, mgan1, sample_rings, \
    rand_covar, sample_torus, sample_x_torus, sample_sphere, sample_base_mixtures, sample_spheres, sample_swiss_roll

import matplotlib.pyplot as plt
import numpy as np
import random
from lk_sim import get_VL_data, sample_VL_prior
from picture_to_dist import sample_elden_ring
from datetime import datetime as dt
from seaborn import kdeplot


def format(n, n_digits = 5):
    if n > 1e-3:
        return round(n,4)
    a = '%E' % n
    str =  a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
    scale = str[-4:]
    digits = str[:-4]
    return digits[:min(len(digits),n_digits)] + scale


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
        final_eps = self.submodel_params['final_eps']
        self.noise_shrink_c = np.exp(np.log(final_eps)/(n-1))

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

    def mmd(self, map_vec, target):
        return self.submodel_params['mmd_func'](map_vec, target)


    def map_mean(self, x_mu, y_mean, y_var, Lambda_mean, X_mean, fit_kernel):
        x_mean = torch.concat([x_mu, y_mean + y_var], dim=1)
        z_mean = fit_kernel(X_mean, x_mean).T @ Lambda_mean
        return z_mean


    def map_var(self, x_mu, y_eta, y_mean, Lambda_var, X_var, y_var, fit_kernel):
        x_var = torch.concat([x_mu, shuffle(y_eta), y_mean + y_var], dim=1)
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
        y_eta *= self.noise_shrink_c

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


        #self.train_idx = self.get_train_idx()
        self.Y_eta = geq_1d(torch.tensor(base_params['Y_eta'], device=self.device, dtype=self.dtype))

        self.Y_mean = deepcopy(self.Y_eta)
        self.Y_var =  0 * self.Y_mean
        self.approx = self.params['approx']
        if self.approx:
            self.Y_mean = geq_1d(torch.tensor(base_params['Y_mean'], device=self.device, dtype=self.dtype))
            self.Y_var = geq_1d(torch.tensor(base_params['Y_var'], device=self.device, dtype=self.dtype))

        self.X_mu = geq_1d(torch.tensor(base_params['X_mu'], device=self.device, dtype=self.dtype))
        self.Y_mu = geq_1d(torch.tensor(base_params['Y_mu'], device=self.device, dtype=self.dtype))
        self.Y_target = torch.concat([deepcopy(self.X_mu), self.Y_mu], dim=1)

        self.X_mu = self.X_mu[:self.params['batch_size']]

        self.X_var = torch.concat([self.X_mu, shuffle(self.Y_eta), self.Y_mean + self.Y_var], dim=1)
        self.X_mean = torch.concat([self.X_mu, self.Y_mean + self.Y_var], dim=1)

        self.Nx = len(self.X_mean)
        self.Ny = len(self.Y_target)

        self.params['fit_kernel_params']['l'] *= l_scale(self.X_mean).cpu()
        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.Nx, device=self.device, dtype=self.dtype)
        self.fit_kXXmean_inv = torch.linalg.inv(self.fit_kernel(self.X_mean, self.X_mean) + self.nugget_matrix)
        self.fit_kXXvar_inv = torch.linalg.inv(self.fit_kernel(self.X_var, self.X_var) + self.nugget_matrix)

        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)

        self.Z_mean = nn.Parameter(self.init_Z(), requires_grad=True)
        self.Z_var = nn.Parameter(self.init_Z(), requires_grad=True)

        self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))
        self.Y_mean_test = deepcopy(self.Y_eta_test)
        self.Y_var_test = 0 * self.Y_eta_test

        if self.approx:
            self.Y_mean_test = geq_1d(torch.tensor(base_params['Y_mean_test'], device=self.device, dtype=self.dtype))
            self.Y_var_test = geq_1d(torch.tensor(base_params['Y_var_test'], device=self.device, dtype=self.dtype))

        self.X_mu_test = geq_1d(torch.tensor(base_params['X_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_mu_test = geq_1d(torch.tensor(base_params['Y_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_test = torch.concat([self.X_mu_test, self.Y_mu_test], dim=1)

        self.params['mmd_kernel_params']['l'] *= l_scale(self.Y_mu_test).cpu()

        self.alpha_z = self.p_vec(self.Nx)
        self.alpha_y = self.p_vec(self.Ny)

        if self.params['E_mmd_YY'] == 0:
            self.E_mmd_YY = self.alpha_y.T @ self.mmd_kernel(self.Y_target, self.Y_target) @ self.alpha_y
            self.params['E_mmd_YY'] = float(self.E_mmd_YY.detach().cpu())
        else:
            self.E_mmd_YY = self.params['E_mmd_YY']

        self.mmd_lambda = 1
        if self.params['mmd_lambda'] != 0:
            self.mmd_lambda = self.params['mmd_lambda']
        else:
            self.mmd_lambda = (1 / self.loss_mmd().detach())
            self.params['mmd_lambda'] = self.mmd_lambda
        self.reg_lambda = self.params['reg_lambda'] * self.mmd_lambda
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
        Z = torch.zeros(self.Y_eta.shape, device=self.device, dtype=self.dtype)
        return Z


    def get_Lambda_mean(self):
        return self.fit_kXXmean_inv @ self.Z_mean


    def get_Lambda_var(self):
        return self.fit_kXXvar_inv @ self.Z_var


    def map_mean(self, x_mu, y_mean, y_var):
        x_mean = torch.concat([x_mu, y_mean + y_var], dim=1)
        Lambda_mean = self.get_Lambda_mean()
        z_mean = self.fit_kernel(self.X_mean, x_mean).T @ Lambda_mean
        return z_mean


    def map_var(self, x_mu, y_eta, y_mean, y_var):
        x_var = torch.concat([x_mu, shuffle(y_eta), y_mean + y_var], dim=1)
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

        z_mean = self.map_mean(x_mu, y_mean, y_var)
        z_var = self.map_var(x_mu, y_eta, y_mean, y_var)
        z = z_mean + z_var

        y_approx = y_mean + y_var
        return_dict = {'y_eta': y_eta, 'y_mean': y_mean + z_mean, 'y_var': y_var + z_var,
                       'y_approx': y_approx + z, 'y': torch.concat([x_mu, z + y_approx], dim = 1)}

        return return_dict


    def mmd(self, map_vec, target):
        map_vec = geq_1d(torch.tensor(map_vec, device=self.device, dtype=self.dtype))
        target = geq_1d(torch.tensor(target, device=self.device, dtype=self.dtype))

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
        Y_approx = self.Y_var + self.Y_mean + self.Z_mean + self.Z_var
        map_vec = torch.concat([self.X_mu, Y_approx], dim=1)
        target = self.Y_target

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, target)

        alpha_z = self.alpha_z
        alpha_y = self.alpha_y

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        mmd  = Ek_ZZ - (2 * Ek_ZY) + Ek_YY
        return mmd * self.mmd_lambda


    def loss_reg(self):
        Z_mean = self.Z_mean
        Z_var = self.Z_var

        reg_1 =  torch.trace(Z_mean.T @ self.fit_kXXmean_inv @ Z_mean)
        reg_2 =  torch.trace(Z_var.T @ self.fit_kXXvar_inv @ Z_var)
        return  self.reg_lambda * (reg_1 + reg_2)


    def loss_test(self):
        x_mu = self.X_mu_test
        y_eta = self.Y_eta_test
        y_mean = self.Y_mean_test
        y_var = self.Y_var_test
        target = self.Y_test
        map_vec = self.map(x_mu, y_eta, y_mean, y_var)['y']
        return self.mmd(map_vec, target) * self.mmd_lambda


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def cond_kernel_transport(X_mu, Y_mu, Y_eta, Y_mean, Y_var,  X_mu_test, Y_eta_test, Y_mu_test,
                          Y_mean_test, Y_var_test, params, n_iter = 101,  iters = -1,  approx = False,
                          batch_size = 4000, reg_lambda = 1e-5, mmd_lambda = 0, E_mmd_yy = 0):

    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta,'nugget': 1e-4,'Y_var': Y_var, 'Y_mean': Y_mean,
                        'fit_kernel_params': deepcopy(params['fit']),'mmd_kernel_params': deepcopy(params['mmd']),
                         'print_freq': 500,'learning_rate': .001, 'reg_lambda': reg_lambda,
                        'Y_eta_test': Y_eta_test, 'X_mu_test': X_mu_test, 'Y_mu_test': Y_mu_test,
                        'Y_mean_test': Y_mean_test, 'approx': approx,'mmd_lambda': mmd_lambda,'Y_var_test': Y_var_test,
                        'iters': iters, 'batch_size': batch_size, 'E_mmd_YY': E_mmd_yy}

    model = CondTransportKernel(transport_params)
    model, loss_dict = train_kernel(model, n_iter)
    return model, loss_dict


def comp_cond_kernel_transport(X_mu, Y_mu, Y_eta, Y_eta_test, X_mu_test, Y_mu_test, params,
                               final_eps = 1, n_iter = 101, n = 50, batch_size = 4000, reg_lambda = 1e-5):
    model_params = {'fit_kernel': [], 'Lambda_mean': [], 'X_mean': [], 'Lambda_var': [], 'X_var': []}
    iters = 0
    noise_shrink_c = np.exp(np.log(final_eps)/(n-1))
    model_params['final_eps'] = final_eps
    Y_mean = 0
    Y_mean_test = 0
    Y_var = 0
    Y_var_test = 0
    approx = False
    mmd_lambda = 0
    E_mmd_yy  = 0

    for i in range(n):
        model = cond_kernel_transport(X_mu, Y_mu, Y_eta, Y_mean, Y_var,  X_mu_test, Y_eta_test, Y_mu_test,
                                      Y_mean_test, Y_var_test, params = params, n_iter = n_iter, iters = iters,
                                      batch_size = batch_size,  approx =approx, mmd_lambda = mmd_lambda,
                                      reg_lambda=reg_lambda, E_mmd_yy = E_mmd_yy)[0]

        model_params['Lambda_mean'].append(model.get_Lambda_mean().detach().cpu().numpy())
        model_params['Lambda_var'].append(model.get_Lambda_var().detach().cpu().numpy())
        model_params['fit_kernel'].append(model.fit_kernel)
        model_params['X_mean'].append(model.X_mean.detach().cpu().numpy())
        model_params['X_var'].append(model.X_var.detach().cpu().numpy())
        mmd_lambda = model.mmd_lambda

        if i==0:
            model_params['mmd_func'] = model.mmd

        map_dict = model.map(X_mu[:batch_size], Y_eta, Y_mean, Y_var)
        Y_mean, Y_var =  map_dict['y_mean'], map_dict['y_var']

        test_map_dict = model.map(X_mu_test, Y_eta_test, Y_mean_test, Y_var_test)
        Y_mean_test, Y_var_test = test_map_dict['y_mean'], test_map_dict['y_var']

        Y_eta *= noise_shrink_c
        Y_eta_test *= noise_shrink_c

        iters = model.iters
        approx = True
        E_mmd_yy = model.E_mmd_YY

    return Comp_transport_model(model_params)


def get_idx_tensors(idx_lists):
    return [torch.tensor(idx_list).long() for idx_list in idx_lists]


def zero_pad(array):
    zero_array = np.zeros([len(array),1])
    return np.concatenate([zero_array, array], axis = 1)


def train_cond_transport(ref_gen, target_gen, params, N, n_iter = 101, process_funcs = [],
                         batch_size = 4000, cond_model_trainer = cond_kernel_transport,
                         idx_dict = {}, reg_lambda = 1e-5, n_transports = 100, final_eps = 1):

    ref_sample = ref_gen(batch_size)
    target_sample = target_gen(N)

    N_test = min(4000, 10 * batch_size)
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

        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, Y_eta_test, X_mu_test, Y_mu_test,
                                                 params = params, n_iter = n_iter, reg_lambda = reg_lambda,
                                               n = n_transports, batch_size = batch_size, final_eps = final_eps))
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


def conditional_transport_exp(ref_gen, target_gen, N = 10000, n_iter = 1001, vmax = None,
                           exp_name= 'exp', plt_range = None,  process_funcs = [], N_plot = 5000,
                           cond_model_trainer= comp_cond_kernel_transport,idx_dict = {}, bins = 70, final_eps = 1,
                           skip_idx = 0, plot_idx = [], batch_size = 4000, n_transports = 50, reg_lambda = 1e-5):
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

     trained_models = train_cond_transport(N = N, ref_gen = ref_gen, target_gen = target_gen, params = exp_params,
                                           n_iter = n_iter, process_funcs = process_funcs,idx_dict = idx_dict,
                                           cond_model_trainer = cond_model_trainer, n_transports = n_transports,
                                           batch_size = batch_size, reg_lambda= reg_lambda, final_eps = final_eps)
     target_sample = target_gen(N_plot)
     ref_sample = ref_gen(N_plot)

     gen_sample = compositional_gen(trained_models, ref_sample, target_sample, idx_dict)
     test_mmd = float(trained_models[0].mmd(gen_sample, target_sample).detach().cpu())

     try:
        cref_sample  = deepcopy(ref_sample)
        cref_sample[:, idx_dict['cond'][0]] += target_sample[:, idx_dict['cond'][0]]
        base_mmd = float(trained_models[0].mmd(cref_sample, target_sample).detach().cpu())
        ntest_mmd = test_mmd/base_mmd
        print(f'Test mmd :{format(test_mmd)}, Base mmd: {format(base_mmd)}, NTest mmd :{format(ntest_mmd)}')
     except BaseException:
        print(f'Test mmd :{format(test_mmd)}')

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


def two_d_exp(ref_gen, target_gen, N = 10000, n_iter=1001, plt_range=None, process_funcs=[],
              slice_range=None, N_plot = 5000, slice_vals=[], bins = 70, exp_name='exp', skip_idx=0,
              vmax=None, n_transports = 70, batch_size = 4000, reg_lambda = 1e-5, final_eps = 1):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    slice_vals = np.asarray(slice_vals)
    plot_idx = torch.tensor([0, 1]).long()
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=vmax,
                                                         bins = bins,exp_name=exp_name, plt_range=plt_range,
                                                         n_transports = n_transports, process_funcs=process_funcs,
                                                         plot_idx=plot_idx, skip_idx=skip_idx, final_eps = final_eps,
                                                         batch_size = batch_size, N_plot=N_plot,reg_lambda = reg_lambda)

    for slice_val in slice_vals:
        ref_sample = ref_gen(N_plot)
        ref_slice_sample = target_gen(N_plot)
        ref_slice_sample[:, idx_dict['cond'][0]] = slice_val
        slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)
        plt.hist(slice_sample[:, 1], bins=bins, range=slice_range, label = f'x ={slice_val}')
    if len(slice_vals):
        plt.savefig(f'{save_dir}/slice_posteriors.png')
        clear_plt()
    return True


def spheres_exp(N = 10000, n_iter = 101, exp_name = 'spheres_exp', n_transports = 150, batch_size = 4000):
    n = 10
    ref_gen = lambda N: sample_base_mixtures(N = N, d = 2, n = 2)
    target_gen = lambda N: sample_spheres(N = N, n = n)


    idx_dict = {'ref': [[0,1]],
                'cond': [list(range(2, 2 + (2*n)))],
                'target': [[0,1]]}

    plt_range = [[.5,1.5],[-1.5,1.5]]
    plot_idx = torch.tensor([0, 1]).long()
    skip_idx = 0
    N_plot = min(10 * batch_size, 5000)
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, N_plot=N_plot,
                                                         skip_idx=skip_idx,exp_name=exp_name, process_funcs=[],
                                                         cond_model_trainer=comp_cond_kernel_transport, vmax=None,
                                                         plot_idx= plot_idx, plt_range = plt_range,idx_dict= idx_dict,
                                                         n_transports = n_transports, batch_size = batch_size)

    slice_vals = np.asarray([[1,.0], [1,.2], [1,.4], [1,.5], [1,.6], [1,.7], [1,.75], [1,.79]])

    save_dir = f'../../data/kernel_transport/{exp_name}'

    for slice_val in slice_vals:
        ref_sample = ref_gen(N_plot)
        RX = np.full((N_plot,2), slice_val)
        ref_slice_sample = sample_spheres(N = N_plot, n = n, RX = RX)

        slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)
        sample_hmap(slice_sample[:,np.asarray([0,1])], f'{save_dir}/x={slice_val[1]}_map.png', bins=60, d=2,
                    range=plt_range)
    return True


def elden_exp(N=10000, n_iter=51, exp_name='elden_exp', n_transports=100, batch_size = 4000):
    ref_gen = sample_normal
    target_gen = sample_elden_ring
    idx_dict = {'ref': [[0, 1]], 'cond': [[]],'target': [[0,1]]}
    skip_idx = 0

    plt_range = [[-1,1],[-1.05,1.15]]
    plot_idx = torch.tensor([0,1]).long()
    N_plot = min(10 *N, 15000)

    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, bins = 75,
                                                         skip_idx=skip_idx,exp_name=exp_name, N_plot=N_plot,
                                                         cond_model_trainer=comp_cond_kernel_transport, vmax=6,
                                                         plot_idx= plot_idx, plt_range = plt_range,idx_dict= idx_dict,
                                                         n_transports = n_transports, batch_size = batch_size)
    return trained_models


def vl_exp(N=10000, n_iter=49, Yd=18, normal=True, exp_name='kvl_exp', n_transports = 100, batch_size = 4000):
    ref_gen = lambda N: sample_normal(N, 4)
    target_gen = lambda N: get_VL_data(N, normal=normal, Yd = Yd)

    X_mean = np.asarray([1, 0.0564, 1, 0.0564])
    X_std = np.asarray([0.53, 0.03, 0.53, 0.03])

    idx_dict = {'ref': [[0, 1, 2, 3]],
                'cond': [list(range(4, 4 + Yd))],
                'target': [[0, 1, 2, 3]]}

    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    skip_idx = 0
    N_plot = min(10 * batch_size, 4000)
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, N_plot = N_plot,
                                                         skip_idx=skip_idx,exp_name=exp_name, process_funcs=[],
                                                         cond_model_trainer=comp_cond_kernel_transport, vmax=None,
                                                         plot_idx= [], plt_range = None ,idx_dict= idx_dict,
                                                         n_transports = n_transports, batch_size = batch_size)

    target_sample = get_VL_data(N_plot, normal=False, Yd = Yd)
    mu = np.mean(target_sample, axis = 0)
    sigma = np.std(target_sample, axis = 0)

    slice_val = np.asarray([.8, .041, 1.07, .04])
    X = np.full((N_plot, 4), slice_val)
    ref_slice_sample = get_VL_data(N_plot, X=X, Yd=Yd, normal = False,  T = 20)

    ref_slice_sample -= mu
    ref_slice_sample /= sigma

    ref_sample = ref_gen(N_plot)

    slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)[:, :4]
    slice_sample *= X_std
    slice_sample += X_mean



    params_keys = ['alpha', 'beta', 'gamma', 'delta']

    ranges1 = {'alpha': [.5, 1.4], 'beta': [0.02, 0.07], 'gamma': [.7, 1.5], 'delta': [0.025, 0.065]}
    ranges2 = {'alpha': [None, None], 'beta': [None, None], 'gamma': [None, None], 'delta': [None, None]}


    for range_idx,ranges in enumerate([ranges1, ranges2]):
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
                        kdeplot(x=x, y=y, fill=True, bw_adjust=0.4, cmap='Blues')
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
    vl_exp(n_iter=49, N=9000, batch_size=8000, n_transports=200, exp_name='kvl_exp_real2')


if __name__=='__main__':
    run()