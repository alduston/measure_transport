import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap
import os
from copy import deepcopy
from get_data import sample_banana, sample_normal, mgan2, sample_spirals, sample_pinweel, mgan1, sample_rings, \
    sample_swiss_roll,rand_covar, rand_diag_covar,sample_mixtures
import matplotlib.pyplot as plt
import numpy as np
import random
from lokta_voltera import get_VL_data,get_cond_VL_data
from picture_to_dist import sample_elden_ring
from datetime import datetime as dt

def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1,1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor


def replace_zeros(array, eps = 1e-5):
    for i,val in enumerate(array):
        if np.abs(val) < eps:
            array[i] = 1.0
    return array


def normalize(array, keep_axes=[]):
    normal_array = deepcopy(array)
    if len(keep_axes):
        norm_axes = np.asarray([axis for axis in range(len(array.shape)) if (axis not in keep_axes)])
        keep_array = deepcopy(normal_array)[:, keep_axes]
        normal_array = normal_array[:, norm_axes]
    normal_array = normal_array - np.mean(normal_array, axis = 0)
    std_vec = replace_zeros(np.std(normal_array, axis = 0))
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


    def base_map(self, y):
        for submodel in self.submodels:
            y = submodel.map(y).T
        return y


    def c_map(self, x, y):
        x = geq_1d(torch.tensor(x, device = self.device))
        y = geq_1d(torch.tensor(y, device = self.device))
        for submodel in self.submodels:
            y = submodel.map(x,y, no_x = True)
        return torch.concat([x, y], dim = 1)


    def map(self, x = [], y = []):
        if self.cond:
            return self.c_map(x,y)
        return self.base_map(x)


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
        if torch.max(self.X_mu)==0:
            self.params['no_mu'] = True

        self.Y_mu = geq_1d(torch.tensor(base_params['Y_mu'], device=self.device, dtype=self.dtype))

        self.X = torch.concat([self.X_mu, self.Y_eta], dim=1)
        self.Y = torch.concat([self.X_mu, self.Y_mu], dim=1)

        if self.params['no_mu']:
            self.X = self.Y_eta
            self.Y = self.Y_mu

        self.Nx = len(self.X)
        self.Ny = len(self.Y)

        self.params['fit_kernel_params']['l'] *= l_scale(self.X)
        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.Nx, device=self.device, dtype=self.dtype)
        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)

        self.params['mmd_kernel_params']['l'] *= l_scale(self.Y_mu)
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


    def map(self, x_mu, y_eta, no_x = False):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        w = torch.concat([x_mu, y_eta], dim=1)
        if self.params['no_mu']:
            w = y_eta
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

    def loss_mmd_no_mu(self):
        map_vec = self.Y_eta + self.Z
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

        try:
            plot_test(self, map_vec, target, x_mu, y_eta,
                      exp_name='spiral_composed2', plt_range=[[-3, 3], [-3, 3]])
        except ValueError:
            pass

        return self.mmd(map_vec, target)


    def loss(self):
        loss_mmd = self.loss_mmd()
        if self.params['no_mu']:
            loss_mmd = self.loss_mmd_no_mu()
        loss_reg = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict



def plot_test(model, map_vec, target, x_mu, y_eta, plt_range = None, vmax = None,
              slice_vals= [0], slice_range = None, exp_name = 'exp', flip = False):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    for slice_val in slice_vals:
        x_slice = torch.full(x_mu.shape, slice_val, device=model.device)
        y = model.map(x_slice, y_eta, no_x = True).detach().cpu().numpy()
        plt.hist(y, bins=60, range=slice_range, label=f'z = {slice_val}')
    plt.legend()
    plt.savefig(f'{save_dir}/slice_hist.png')
    clear_plt()

    range = plt_range
    x_left, x_right = range[0]
    y_bottom, y_top = range[1]

    if flip:
        map_vec = flip_2tensor(map_vec)
        target = flip_2tensor(target)

    plot_vec = map_vec.detach().cpu().numpy()
    if len(plot_vec.T) > 2:
        plot_vec = plot_vec[:, 1:]

    x, y = plot_vec.T
    plt.hist2d(x, y, density=True, bins=50, range=range, cmin=0, vmin=0, vmax=vmax)
    plt.colorbar()
    plt.savefig(f'{save_dir}/output_map.png')
    clear_plt()

    plt.scatter(x, y, s=5)
    plt.xlim(x_left, x_right)
    plt.ylim(y_bottom, y_top)
    plt.savefig(f'{save_dir}/output_scatter.png')
    clear_plt()

    if model.iters < 50:
        plot_vec = target.detach().cpu().numpy()
        if len(plot_vec.T) > 2:
            plot_vec = plot_vec[:, 1:]
        x, y = plot_vec.T
        plt.hist2d(x, y, density=True, bins=50, range=range, cmin=0, vmin=0, vmax=vmax)
        plt.colorbar()
        plt.savefig(f'{save_dir}/target_map.png')
        clear_plt()

        plt.scatter(x, y, s=5)
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
        plt.savefig(f'{save_dir}/target_scatter.png')
        clear_plt()
    return True


def base_kernel_transport(Y_eta, Y_mu, params, n_iter = 1001, Y_eta_test = []):
    transport_params = {'X': Y_eta, 'Y': Y_mu, 'reg_lambda': 1e-5,'normalize': False,
                   'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                   'print_freq':  100, 'learning_rate': .01, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=n_iter)
    return transport_kernel


def comp_base_kernel_transport(Y_eta, Y_mu, params, n_iter = 1001, Y_eta_test = [], n = 3, f = .5):
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
                        'fit_kernel_params': deepcopy(params['mmd']), 'mmd_kernel_params': deepcopy(params['fit']),
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


def get_idx_tensors(idx_lists):
    return [torch.tensor(idx_list).long() for idx_list in idx_lists]


def zero_pad(array):
    zero_array = np.zeros([len(array),1])
    return np.concatenate([zero_array, array], axis = 1)


def train_cond_transport(ref_gen, target_gen, params, N = 1000, n_iter = 1001, process_funcs = [],
                         cond_model_trainer = cond_kernel_transport, ref_idx_lists = [], target_idx_lists = []):

    ref_sample = ref_gen(N)
    target_sample = zero_pad(target_gen(N))

    test_sample = ref_gen(5 * N)
    test_target_sample = zero_pad(target_gen(5 * N))

    if len(process_funcs):
        forward = process_funcs[0]
        target_sample = forward(target_sample)

    ref_idx_tensors = get_idx_tensors(ref_idx_lists)
    target_idx_tensors = get_idx_tensors(target_idx_lists)

    trained_models = []

    for i in range(len(ref_idx_tensors)):
        X_mu = target_sample[:,  ref_idx_tensors[i]]
        X_mu_test = test_target_sample[:, ref_idx_tensors[i]]

        Y_mu = target_sample[:, target_idx_tensors[i]]
        Y_mu_test = test_target_sample[:, target_idx_tensors[i]]


        Y_eta = ref_sample[:,target_idx_tensors[i]-1]
        Y_eta_test = test_sample[:, target_idx_tensors[i]-1]
        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test,
                                                 Y_mu_test = Y_mu_test, X_mu_test = X_mu_test))

    return trained_models


def compositional_gen(trained_models, ref_sample, cond_indexes):
    ref_sample = geq_1d(ref_sample)
    X = geq_1d(ref_sample)
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, i]
        try:
            X = model.map(X[:, cond_indexes[i]], Y_eta)
        except IndexError:
            X = model.map(X[:, cond_indexes[i]-1], Y_eta)

    return X


def conditional_gen(trained_models, ref_sample, cond_sample, ref_idx_tensors):
    X = geq_1d(cond_sample)
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, ref_idx_tensors[i]]
        X = model.map(X, Y_eta)
    return X



def comp_gen_exp(ref_gen, target_gen, N = 1000, n_iter = 1001, exp_name= 'exp', plt_range = None, vmax = None):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    l = l_scale(torch.tensor(ref_gen(N)[:, 1]))
    mmd_params = {'name': 'r_quadratic', 'l': torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    fit_params = {'name': 'r_quadratic', 'l':  torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    exp_params = {'fit': mmd_params, 'mmd': fit_params}

    Y_eta = ref_gen(N)
    Y_eta_test = ref_gen(N)
    Y_mu = target_gen(N)

    comp_model = comp_base_kernel_transport(Y_eta, Y_mu, exp_params, n_iter, Y_eta_test=Y_eta_test, n=25, f=1)

    Y_eta_plot = ref_gen(N)
    Y_mu_plot = target_gen(N)
    gen_sample = comp_model.map(torch.tensor(Y_eta_plot, device=comp_model.device))

    sample_scatter(gen_sample, f'{save_dir}/gen_scatter.png', bins=25, d=2, range=plt_range)
    sample_hmap(gen_sample, f'{save_dir}/gen_map.png', bins=65, d=2, range=plt_range, vmax= vmax)

    sample_scatter(Y_mu_plot, f'{save_dir}/target.png', bins=25, d=2, range=plt_range)
    sample_hmap(Y_mu_plot, f'{save_dir}/target_map.png', bins=65, d=2, range=plt_range, vmax= vmax)

    return True

def sode_hist(trajectories, savedir, save_name = 'traj_hist'):
    trajectories = torch.tensor(trajectories)
    N,n = trajectories.shape
    fig, axs = plt.subplots(n)
    for i in range(n):
        hist_data = trajectories[:, i]
        axs[i].hist(hist_data.detach().cpu().numpy(), label=f't = {i}', bins = 50, range = [-6,6])
    for ax in fig.get_axes():
        ax.label_outer()
    plt.savefig(f'{savedir}/{save_name}.png')
    clear_plt()


def conditional_transport_exp(ref_gen, target_gen, N = 1000, n_iter = 1001, slice_vals = [], vmax = None,
                           exp_name= 'exp', plt_range = None, slice_range = None, process_funcs = [],
                           cond_model_trainer= comp_cond_kernel_transport,ref_idx_lists = [], target_idx_lists = [],
                           skip_base  = 0, skip_idx = 0, traj_hist = False, plot_idx = []):
     save_dir = f'../../data/kernel_transport/{exp_name}'
     try:
         os.mkdir(save_dir)
     except OSError:
         pass

     #l = l_scale(torch.tensor(ref_gen(N)))
     nr = len(ref_gen(1)[0])

     mmd_params = {'name': 'r_quadratic', 'l': torch.exp(torch.tensor(-1.25)), 'alpha': 1}
     fit_params = {'name': 'r_quadratic', 'l': torch.exp(torch.tensor(-1.25)), 'alpha': 1}
     exp_params = {'fit': mmd_params, 'mmd': fit_params}


     if not len(ref_idx_lists):
         ref_idx_lists = [[0]] + [list(range(k + 1))[1:] for k in range(nr)][1:]
         target_idx_lists = [[k+1] for k in range(nr)]

     trained_models = train_cond_transport(ref_gen, target_gen, exp_params, N, n_iter,
                                           process_funcs, cond_model_trainer, ref_idx_lists, target_idx_lists)
     ref_idx_tensors = get_idx_tensors(ref_idx_lists)

     if not skip_base:
        gen_sample = compositional_gen(trained_models, ref_gen(10 * N),ref_idx_tensors)
     else:
         cond_idx_tensor =  get_idx_tensors(ref_idx_lists)[skip_idx]
         cref_idx_tensors = get_idx_tensors(target_idx_lists)[skip_idx:]

         cond_ref_sample = target_gen(10 * N)[:, cond_idx_tensor]
         eta_ref_sample = ref_gen(10 * N)
         gen_sample = conditional_gen(trained_models[skip_idx+1:], eta_ref_sample, cond_ref_sample, cref_idx_tensors)

     if traj_hist:
        sode_hist(gen_sample, save_dir, 'gen_traj_hist')
        sode_hist(target_gen(10*N), save_dir, 'traj_hist')

     hist_idx = 1
     if len(slice_vals):
         for slice_val in slice_vals:
             ref_slice_sample = torch.tensor([slice_val for i in range(len(cond_ref_sample))],
                                             device = trained_models[0].device).reshape(cond_ref_sample.shape)
             slice_sample = conditional_gen(trained_models[skip_idx+1:], eta_ref_sample, ref_slice_sample, cref_idx_tensors)
             plt.hist(slice_sample[:, hist_idx].detach().cpu().numpy(), label = f'z  = {slice_val}', bins = 60, range=slice_range)
         plt.legend()
         plt.savefig(f'{save_dir}/conditional_hists.png')
         clear_plt()

     if len(process_funcs):
         backward = process_funcs[1]
         gen_sample = backward(gen_sample.cpu())

     if not len(plot_idx):
        plot_idx = torch.tensor([0,1]).long()
     gen_sample = gen_sample[:, plot_idx]
     target_sample = target_gen(10 * N)[:, plot_idx]

     sample_scatter(gen_sample, f'{save_dir}/gen_scatter.png', bins=25, d = 2, range = plt_range)
     sample_hmap(gen_sample, f'{save_dir}/gen_map.png', bins=70, d=2, range=plt_range, vmax=vmax)

     sample_scatter(target_sample, f'{save_dir}/target.png', bins=25, d=2, range=plt_range)
     sample_hmap(target_sample, f'{save_dir}/target_map.png', bins=70, d=2, range=plt_range, vmax=vmax)
     return trained_models



def lokta_vol_exp(N = 10000, n_iter = 10000):
    d = 4
    covar = rand_diag_covar(d)
    mu = 3 * np.random.rand(d)
    ref_gen = lambda n: sample_normal(n, d)
    target_gen = lambda n: sample_normal(n, d, mu = mu, sigma = covar)

    ref_idx_lists = [[0],[0,1],[0,1,2]]
    target_idx_list = [[1],[2],[3]]
    skip_base = False

    trained_models, eta_ref_sample, cref_idx_tensors, cond_idx_tensors = \
        conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, slice_vals=[], vmax=None,
                              exp_name='covar_exp', plt_range=None, slice_range=None, process_funcs=[],
                              cond_model_trainer=comp_cond_kernel_transport,
                              ref_idx_lists=ref_idx_lists , target_idx_lists=target_idx_list,
                              skip_base=skip_base, skip_idx=0, traj_hist=True)

    #cond_sample = get_cond_VL_data(10 * N)[:, cond_idx_tensors]
    #gen_sample = conditional_gen(trained_models, eta_ref_sample, cond_sample, cref_idx_tensors)
    #plt.hist(gen_sample[-1].detach().numpy())
    #plt.savefig(f'../../data/kernel_transport/lk_exp/param_hist.png')
    #return True


#At step 300: fit_loss = 0.000441, reg_loss = 3.1e-05, test loss = 0.000952

def run():
    ref_gen = sample_normal
    target_gen = sample_spirals
    range = [[-3, 3], [-3, 3]]

    conditional_transport_exp(ref_gen, target_gen, N=1000, n_iter=1001, slice_vals=[0], vmax=.15,
                              exp_name='spiral_composed3', plt_range=range, slice_range=[-3, 3],
                              process_funcs=[], skip_base=True, traj_hist=True)



if __name__=='__main__':
    run()