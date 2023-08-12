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
from lokta_voltera import get_VL_data
from picture_to_dist import sample_elden_ring
from datetime import datetime as dt
from seaborn import kdeplot


def shuffle(tensor):
    if geq_1d(tensor).shape[1] <=1:
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
        self.param_keys = ['X_mu', 'Y_eta', 'Y_approx']
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'


    def param_map(self, y_eta, step_idx,y_approx = [], x_mu = []):
        temp_param_dict = self.temp_param_dict
        X_mu = geq_1d(torch.tensor(temp_param_dict['X_mu'], device=self.device, dtype=self.dtype))
        Y_eta =  geq_1d(torch.tensor(temp_param_dict['Y_eta'], device=self.device, dtype=self.dtype))
        Y_approx =  geq_1d(torch.tensor(temp_param_dict['Y_approx'], device=self.device, dtype=self.dtype))

        Lambda = self.submodel_params['Lambda'][step_idx]
        fit_kernel = self.submodel_params['fit_kernel'][step_idx]

        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        if len(x_mu):
            x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
            w = torch.concat([x_mu, y_eta], dim=1)
            W = torch.concat([X_mu, Y_eta], dim=1)

        if len(y_approx):
            y_approx = geq_1d(torch.tensor(y_approx, device=self.device, dtype=self.dtype))
            w = torch.concat([w, y_approx], dim = 1)
            W = torch.concat([W, Y_approx], dim = 1)
        else:
            y_approx = deepcopy(y_eta)
            Y_approx = deepcopy(Y_eta)

        z = fit_kernel(W, w).T @ Lambda
        Z = fit_kernel(W, W).T @ Lambda

        y_eta = shuffle(y_eta)
        temp_param_dict['Y_eta'] = shuffle(Y_eta)
        temp_param_dict['Y_approx'] = Z + Y_approx
        self.temp_param_dict = temp_param_dict
        return (z + y_approx, y_eta)


    def c_map(self, x, y, no_x = False):
        x = geq_1d(torch.tensor(x, device = self.device))
        y = geq_1d(torch.tensor(y, device = self.device))
        y_approx = []
        for step_idx in range(len(self.submodel_params['Lambda'])):
            y_approx,y = self.param_map(y_eta = y, step_idx = step_idx,
                                        y_approx = y_approx, x_mu = x)
        if no_x:
            return y_approx
        return torch.concat([x, y_approx], dim = 1)


    def map(self, x = [], y = [], no_x = False):
        self.temp_param_dict = {key: deepcopy(self.submodel_params[key]) for key in self.param_keys}
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

        self.params['no_mu'] = False
        if self.X_mu.shape[1]==0:
            self.params['no_mu'] = True


        self.Y_approx = self.Y_eta
        self.params['approx'] = False
        if len(base_params['Y_approx']):
            self.Y_approx = geq_1d(torch.tensor(base_params['Y_approx'], device=self.device, dtype=self.dtype))
            self.params['approx'] = True
        self.Y_mu = geq_1d(torch.tensor(base_params['Y_mu'], device=self.device, dtype=self.dtype))

        self.X = torch.concat([self.X_mu, self.Y_eta], dim=1)
        self.Y = torch.concat([self.X_mu, self.Y_mu], dim=1)

        if self.params['no_mu']:
            self.X = self.Y_eta
            self.Y = self.Y_mu

        if self.params['approx']:
            self.X = torch.concat([self.X, self.Y_approx], dim=1)

        self.Nx = len(self.X)
        self.Ny = len(self.Y)

        self.params['fit_kernel_params']['l'] *= l_scale(self.X).cpu()
        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.Nx, device=self.device, dtype=self.dtype)
        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)

        self.params['mmd_kernel_params']['l'] *= l_scale(self.Y_mu).cpu()
        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.mmd_YY = self.mmd_kernel(self.Y, self.Y)

        self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))
        self.Y_approx_test = self.Y_eta_test
        if len(base_params['Y_approx_test']):
            self.Y_approx_test = geq_1d(torch.tensor(base_params['Y_approx_test'], device=self.device, dtype=self.dtype))

        self.X_mu_test = geq_1d(torch.tensor(base_params['X_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_mu_test = geq_1d(torch.tensor(base_params['Y_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_test = torch.concat([self.X_mu_test, self.Y_mu_test], dim=1)

        self.alpha_z = self.p_vec(self.Nx)
        self.alpha_y = self.p_vec(self.Ny)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y
        self.iters = deepcopy(self.params['iters'])


    def p_vec(self, n):
        return torch.full([n], 1/n, device=self.device, dtype=self.dtype)


    def init_Z(self):
        Z = torch.zeros(self.Y_mu.shape, device=self.device, dtype=self.dtype)
        return Z


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def map(self, x_mu, y_eta, y_approx = [], no_x = False):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        if self.params['no_mu']:
            w = y_eta
        else:
            w = torch.concat([x_mu, y_eta], dim=1)

        if self.params['approx']:
            y_approx = geq_1d(torch.tensor(y_approx, device=self.device, dtype=self.dtype))
            w = torch.concat([w, y_approx], dim = 1)
        else:
            y_approx = deepcopy(y_eta)
        Lambda = self.get_Lambda()
        z = self.fit_kernel(self.X, w).T @ Lambda

        y_eta = shuffle(y_eta)
        if no_x or self.params['no_mu']:
            return  (z + y_approx, y_eta)
        return (torch.concat([x_mu, z + y_approx], dim = 1), y_eta)


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


    def loss_mmd_no_mu(self):
        map_vec = self.Y_approx + self.Z
        target = self.Y_mu

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, target)

        alpha_z = self.alpha_z
        alpha_y = self.alpha_y

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        return Ek_ZZ - (2 * Ek_ZY) + Ek_YY


    def loss_mmd(self):
        map_vec = torch.concat([self.X_mu, self.Y_approx + self.Z], dim=1)
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
        y_approx = self.Y_approx_test
        target = self.Y_test
        map_vec = self.map(x_mu, y_eta, y_approx)[0]
        return self.mmd(map_vec, target)


    def loss(self):
        if self.params['no_mu']:
            loss_mmd = self.loss_mmd_no_mu()
        else:
            loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 10001, Y_approx = [],
                          Y_eta_test = [], X_mu_test = [],Y_mu_test = [], Y_approx_test = [], iters = 0):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 1e-5, 'Y_approx': Y_approx,
                        'fit_kernel_params': deepcopy(params['mmd']), 'mmd_kernel_params': deepcopy(params['fit']),
                        'print_freq': 100, 'learning_rate': .006, 'nugget': 1e-4, 'Y_eta_test': Y_eta_test,
                        'X_mu_test': X_mu_test, 'Y_mu_test': Y_mu_test, 'Y_approx_test': Y_approx_test, 'iters': iters}
    ctransport_kernel = CondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel


def comp_cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 1001, n = 200, Y_approx = [],
                               Y_eta_test = [], X_mu_test = [],Y_mu_test = [], Y_approx_test = []):
    model_params = {'fit_kernel': [], 'Lambda': [], 'X': []}
    iters = 0
    for i in range(n):
        model = cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test,
                                      Y_approx = Y_approx , X_mu_test = X_mu_test, Y_mu_test = Y_mu_test,
                                      Y_approx_test = Y_approx_test, iters = iters)
        model_params['Lambda'].append(model.get_Lambda())
        model_params['fit_kernel'].append(model.fit_kernel)

        if i==0:
            model_params['X_mu']= X_mu
            model_params['Y_eta']= Y_eta
            model_params['Y_approx'] = 0 * Y_mu


        Y_approx, Y_eta = model.map(model.X_mu, model.Y_eta, model.Y_approx, no_x = True)
        Y_approx_test, Y_eta_test = model.map(model.X_mu_test, model.Y_eta_test, model.Y_approx_test, no_x = True)
        iters = model.iters
    return Comp_transport_model(model_params)


def get_idx_tensors(idx_lists):
    return [torch.tensor(idx_list).long() for idx_list in idx_lists]


def zero_pad(array):
    zero_array = np.zeros([len(array),1])
    return np.concatenate([zero_array, array], axis = 1)


def train_cond_transport(ref_gen, target_gen, params, N = 1000, n_iter = 1001, process_funcs = [],
                         cond_model_trainer = cond_kernel_transport, idx_dict = {},  n_transports = 200):
    ref_sample = ref_gen(N)
    target_sample = target_gen(N)

    N_test = N
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
                           cond_model_trainer= comp_cond_kernel_transport,idx_dict = {},
                           skip_idx = 0, plot_idx = [], plots_hists = False, n_transports = 200):
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
     N_test = min(10 * N, 15000)
     target_sample = target_gen(N_test)
     ref_sample = ref_gen(N_test)

     gen_sample = compositional_gen(trained_models, ref_sample, target_sample, idx_dict)

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

     sample_hmap(gen_sample, f'{save_dir}/gen_map.png', bins=70, d=d , range=plt_range, vmax=vmax)
     sample_hmap(target_sample, f'{save_dir}/target_map.png', bins=70, d=d , range=plt_range, vmax=vmax)

     return trained_models, idx_dict


def two_d_exp(ref_gen, target_gen, N, n_iter=1001, plt_range=None, process_funcs=[],
              slice_vals=[], slice_range=None, exp_name='exp', skip_idx=0, vmax=None, n_transports = 200):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    slice_vals = np.asarray(slice_vals)
    plot_idx = torch.tensor([0, 1]).long()
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=vmax,
                                                         exp_name=exp_name, plt_range=plt_range, n_transports = n_transports,
                                                         plot_idx=plot_idx, process_funcs=process_funcs, skip_idx=skip_idx)
    N_test = min(10 * N, 15000)
    for slice_val in slice_vals:
        ref_sample = ref_gen(N_test)
        ref_slice_sample = target_gen(N_test)
        ref_slice_sample[:, idx_dict['cond'][0]] = slice_val
        slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)
        plt.hist(slice_sample[:, 1], bins=50, range=slice_range, label = f'x ={slice_val}')
    plt.savefig(f'{save_dir}/slice_posteriors.png')
    clear_plt()
    return True


def spheres_exp(N = 5000, n_iter = 101, exp_name = 'spheres_exp', n_transports = 250):
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

    N_test =  min(10 * N, 15000)
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


def vl_exp(N=10000, n_iter=10000, Yd=18, normal=True, exp_name='vl_exp'):
    ref_gen = lambda N: sample_normal(N, 4)
    target_gen = lambda N: get_VL_data(N, Yd=Yd, normal=normal)

    idx_dict = {'ref': [[0, 1, 2, 3]],
                'cond': [list(range(4, 4 + Yd))],
                'target': [[0, 1, 2, 3]]}

    skip_idx = 0
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, vmax=None,
                                                         exp_name=exp_name, process_funcs=[],
                                                         cond_model_trainer=comp_cond_kernel_transport,
                                                         idx_dict=idx_dict, skip_idx=skip_idx, plot_idx=[],
                                                         plt_range=None)

    N_test = min(10 * N, 15000)
    slice_val = np.asarray([.8, .041, 1.07, .04])

    X = np.full((N_test, 4), slice_val)
    ref_slice_sample = normalize(get_VL_data(N_test, X=X, Yd=Yd, normal=normal))
    ref_sample = ref_gen(N_test)

    slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict)

    params_keys = ['alpha', 'beta', 'gamma', 'delta']
    ranges = {'alpha': [.5, 1.4], 'beta': [.02, .07], 'gamma': [.7, 1.5], 'delta': [0, .07]}

    for i, key_i in enumerate(params_keys):
        for j, key_j in enumerate(params_keys):
            if i <= j:
                plt.subplot(4, 4, 1 + (4 * j + i))

                if i < j:
                    x, y = slice_sample[:, torch.tensor([i, j]).long()].T
                    plt_range = [ranges[key_i], ranges[key_j]]
                    kdeplot(x=x, y=y, fill=True, bw_adjust=0.25, cmap='Blues')
                    plt.scatter(x=slice_val[i], y=slice_val[j], s=20, color='red')
                    plt.xlim(plt_range[0][0], plt_range[0][1])
                    plt.ylim(plt_range[1][0], plt_range[1][1])

                else:
                    x = slice_sample[:, i]
                    plt_range = ranges[key_i]
                    plt.hist(x, bins=50, range=plt_range)
                    plt.axvline(slice_val[i], color='red', linewidth=5)
                    plt.xlim(plt_range[0], plt_range[1])

                if not i:
                    plt.xlabel(params_keys[i])
                if not j:
                    plt.ylabel(params_keys[j])

    plt.savefig(f'../../data/kernel_transport/{exp_name}/posterior_samples.png')
    return True


def run():
    spheres_exp(8000, 101, exp_name='spheres_exp3')

    '''
              
    two_d_exp(sample_normal, mgan2, N=8000, n_iter=101, plt_range=[[-2.5, 2.5], [-1.05, 1.05]],
              slice_vals=[-1, 0, 1], slice_range=[-1.5, 1.5], exp_name='mgan2_composed3', skip_idx=1, vmax=2)


    two_d_exp(sample_normal, mgan1, N=8000, n_iter=101, plt_range=[[-2.5, 2.5], [-1, 3]],
              slice_vals=[-1, 0, 1], slice_range=[-1.5, 1.5], exp_name='mgan1_composed3', skip_idx=1, vmax=.5)
    

    two_d_exp(sample_normal, sample_spirals, N=8000, n_iter=101, plt_range=[[-3, 3], [-3, 3]],
              slice_vals=[0], slice_range=[-3, 3], exp_name='spiral_composed3', skip_idx=1, vmax=.15)


    two_d_exp(sample_normal, sample_swiss_roll, N=5000, n_iter=101, plt_range=[[-3, 3], [-3, 3]],
              slice_vals=[0], slice_range=[-3, 3], exp_name='swiss_roll_composed3', skip_idx=1, vmax=.25)
    

    vl_exp(8000, 601, exp_name='vl_exp2')


    spheres_exp(8000, 601,  exp_name='spheres_exp2')
    '''


if __name__=='__main__':
    run()