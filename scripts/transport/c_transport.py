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
        y_approx = deepcopy(y)
        for submodel in self.submodels:
            y_approx = submodel.map(y, y_approx).T
        return y_approx


    def c_map(self, x, y):
        x = geq_1d(torch.tensor(x, device = self.device))
        y = geq_1d(torch.tensor(y, device = self.device))
        y_approx = deepcopy(y)
        for submodel in self.submodels:
            y_approx = submodel.map(x, y, y_approx, no_x = True)
        return torch.concat([x, y_approx], dim = 1)


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
        self.iters = 0


    def p_vec(self, n):
        return torch.full([n], 1/n, device=self.device, dtype=self.dtype)


    def init_Z(self):
        return torch.zeros(self.Y_mu.shape, device=self.device, dtype=self.dtype)


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

        if no_x or self.params['no_mu']:
            return z + y_approx
        return torch.concat([x_mu, z + y_approx], dim = 1)


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
        map_vec = self.map(x_mu, y_eta, y_approx)

        try:
            plot_test(self, map_vec, target, x_mu, y_eta, y_approx,
                      plt_range=[[-2.5, 2.5], [-1.05, 1.05]], vmax=2,
                      slice_vals=[0], slice_range=[-1.5, 1.5],
                      exp_name='mgan2_composed2')
        except ValueError:
            pass
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



def plot_test(model, map_vec, target, x_mu, y_eta, y_approx, plt_range = None, vmax = None,
              slice_vals= [0], slice_range = None, exp_name = 'exp', flip = False):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    for slice_val in slice_vals:
        x_slice = torch.full(x_mu.shape, slice_val, device=model.device)
        y_approx = model.map(x_slice, y_eta, y_approx, no_x = True).detach().cpu().numpy()
        plt.hist(y_approx, bins=60, range=slice_range, label=f'z = {slice_val}')
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
                   'print_freq':  100, 'learning_rate': .002, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=n_iter)
    return transport_kernel


def cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 10001, Y_approx = [],
                          Y_eta_test = [], X_mu_test = [],Y_mu_test = [], Y_approx_test = []):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 1e-5, 'Y_approx': Y_approx,
                        'fit_kernel_params': deepcopy(params['mmd']), 'mmd_kernel_params': deepcopy(params['fit']),
                        'print_freq': 100, 'learning_rate': .002, 'nugget': 1e-4, 'Y_eta_test': Y_eta_test,
                        'X_mu_test': X_mu_test, 'Y_mu_test': Y_mu_test, 'Y_approx_test': Y_approx_test}
    ctransport_kernel = CondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel


def comp_cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 1001, Y_approx = [],
                               Y_eta_test = [], X_mu_test = [],Y_mu_test = [], Y_approx_test = [], n = 6, f = .8):
    models = []
    for i in range(n):
        model = cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test = Y_eta_test,
                                      Y_approx = Y_approx , X_mu_test = X_mu_test, Y_mu_test = Y_mu_test,
                                      Y_approx_test = Y_approx_test)
        n_iter = int(n_iter * f)

        #Y_eta = model.map(model.X_mu, model.Y_eta, no_x = True)
        #Y_eta_test = model.map(model.X_mu_test, model.Y_eta_test, no_x = True)

        Y_approx = model.map(model.X_mu, model.Y_eta, model.Y_approx, no_x = True)
        Y_approx_test = model.map(model.X_mu_test, model.Y_eta_test, model.Y_approx_test, no_x = True)
        models.append(model)
    return Comp_transport_model(models, cond=True)


def get_idx_tensors(idx_lists):
    return [torch.tensor(idx_list).long() for idx_list in idx_lists]


def zero_pad(array):
    zero_array = np.zeros([len(array),1])
    return np.concatenate([zero_array, array], axis = 1)


def train_cond_transport(ref_gen, target_gen, params, N = 1000, n_iter = 1001, process_funcs = [],
                         cond_model_trainer = cond_kernel_transport, idx_dict = {}):

    ref_sample = ref_gen(N)
    target_sample = target_gen(N)

    N_test = min(10 * N, 20000)
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
                                                 Y_mu_test = Y_mu_test, X_mu_test = X_mu_test))
    return trained_models


def compositional_gen(trained_models, ref_sample, idx_dict):
    cond_indexes = idx_dict['cond']
    ref_indexes = idx_dict['ref']
    ref_sample = geq_1d(ref_sample)
    X = geq_1d(ref_sample)
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        X = model.map(X[:, cond_indexes[i]], ref_sample[:, ref_indexes[i]])
    return X


#[[ 1.03982393]
 #[-1.04414654]]
#[0.44714918 0.43064544]
#torch.Size([1000, 2])
#tensor([[0],[0]])
#[0.44714918 0.43064544]
#torch.Size([1000, 2])


def conditional_gen(trained_models, ref_sample, target_sample, idx_dict, skip_idx):
    idx_dict = {key: val[skip_idx:] for key,val in idx_dict.items()}
    trained_models = trained_models[skip_idx:]
    X = target_sample
    ref_indexes = idx_dict['ref']
    cond_indexes = idx_dict['cond']

    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, ref_indexes[i]]
        X = model.map(X[:, cond_indexes[i]], Y_eta)
    return X


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
                           cond_model_trainer= comp_cond_kernel_transport,idx_dict = {},
                           skip_base  = 0, skip_idx = 1, traj_hist = False, plot_idx = []):
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
     trained_models = train_cond_transport(ref_gen, target_gen, exp_params, N, n_iter,
                                           process_funcs, cond_model_trainer, idx_dict = idx_dict)
     N_test = min(10 * N, 20000)
     target_sample = target_gen(N_test)
     ref_sample = ref_gen(N_test)

     if not skip_base:
        gen_sample = compositional_gen(trained_models, ref_sample, idx_dict)
     else:
         gen_sample = conditional_gen(trained_models, ref_sample, target_sample, idx_dict, skip_idx)
     if traj_hist:
        sode_hist(gen_sample, save_dir, 'gen_traj_hist')
        sode_hist(target_gen(N_test), save_dir, 'traj_hist')

     hist_idx = 1
     if len(slice_vals):
         for slice_val in slice_vals:
             ref_slice_sample = torch.zeros(target_sample.shape, device = trained_models[0].device)
             ref_slice_sample += geq_1d(torch.tensor([slice_val for i in range(len(ref_sample))],
                                             device = trained_models[0].device))
             slice_sample = conditional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict, skip_idx)
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
     target_sample = target_sample[:, plot_idx]

     sample_scatter(gen_sample, f'{save_dir}/gen_scatter.png', bins=25, d = 2, range = plt_range)
     sample_hmap(gen_sample, f'{save_dir}/gen_map.png', bins=70, d=2, range=plt_range, vmax=vmax)

     sample_scatter(target_sample, f'{save_dir}/target.png', bins=25, d=2, range=plt_range)
     sample_hmap(target_sample, f'{save_dir}/target_map.png', bins=70, d=2, range=plt_range, vmax=vmax)
     return trained_models


def lokta_vol_exp(N = 10000, n_iter = 10000, Yd = 4):
    ref_gen = lambda n: sample_normal(n, Yd)
    target_gen = lambda N: get_VL_data(N, Yd = Yd)

    idx_dict = {'ref': list(range(Yd)),
                'cond': [list(range(i + 4)) for i in range(Yd)],
                'target': [[i + 4] for i in range(Yd)]}

    conditional_transport_exp(ref_gen, target_gen, N=N, n_iter=n_iter, slice_vals=[0], vmax=None,
                              exp_name='lk_exp', plt_range=None, slice_range=None, process_funcs=[],
                              cond_model_trainer=comp_cond_kernel_transport,idx_dict= idx_dict,
                              skip_base=True, skip_idx=1, traj_hist=True)
    return True


def param_infer_exp(N = 10000, n_iter = 10000, Yd = 6):
    params = [-0.125, -3, -0.125, -3]

    ref_gen = lambda n: sample_normal(n, 4)
    target_gen = lambda N: normalize(get_cond_VL_data(N, Yd=Yd))
    ref_idx_lists = [list(range(Yd,))]
    return True



def run():
    ref_gen = sample_normal
    process_funcs = []

    target_gen = sample_spirals
    range = [[-3, 3], [-3, 3]]
    conditional_transport_exp(ref_gen, target_gen, N=5000, n_iter=8001, slice_vals=[0], vmax=.15,
                              exp_name='spiral_composed', plt_range=range, slice_range=[-3, 3],
                              process_funcs=process_funcs, skip_base=False)

    target_gen = sample_rings
    range = [[-3, 3], [-3, 3]]
    conditional_transport_exp(ref_gen, target_gen, N=5000, n_iter=8001, slice_vals=[0], vmax=.16,
                              exp_name='rings_composed', plt_range=range, slice_range=[-3, 3],
                              process_funcs=process_funcs, skip_base=False)

    target_gen = mgan1
    range = [[-2.5, 2.5], [-1, 3]]
    conditional_transport_exp(ref_gen, target_gen, N=5000, n_iter=8001, slice_vals=[0], vmax=.5,
                              exp_name='mgan1_composed', plt_range=range, slice_range=[-1.5, 1.5],
                              process_funcs=process_funcs, skip_base=False)

    target_gen = mgan2
    range = [[-2.5, 2.5], [-1, 1]]
    conditional_transport_exp(ref_gen, target_gen, N=5000, n_iter=8001, slice_vals=[0], vmax=2,
                              exp_name='mgan2_composed', plt_range=range, slice_range=[-1.5, 1.5],
                              process_funcs=process_funcs, skip_base=False)

    target_gen = sample_elden_ring
    range = [[-2.5, 2.5], [-1, 1]]
    conditional_transport_exp(ref_gen, target_gen, N= 10000, n_iter=8001, slice_vals=[0], vmax=4.5,
                              exp_name='elden_composed', plt_range=range, slice_range=[-1, 1],
                              process_funcs=process_funcs, skip_base=False)


'''
    ref_gen = sample_normal
    target_gen = sample_spirals()
    range = [[-3, 3], [-3, 3]]

    conditional_transport_exp(ref_gen, target_gen, N=5000, n_iter=4001, slice_vals=[0], vmax=.15,
                              exp_name='spiral_approx_test', plt_range=range, slice_range=[-1.5, 1.5],
                              process_funcs=[], skip_base=False, traj_hist=True)

    d = 8
    n_mixtures = 8
    ref_gen = lambda N: sample_normal(N, d)

    sigma_vecs = [.5 * rand_covar(d) for i in range(n_mixtures)]
    mu_vecs  = [15 * np.random.random(d) for i in range(n_mixtures)]

    #target_gen = lambda N: normalize(get_cond_VL_data(N, Yd=4))
    target_gen = lambda N: normalize(sample_mixtures(N, mu_vecs, sigma_vecs))
    conditional_transport_exp(ref_gen, target_gen, N=5000, n_iter=3001, slice_vals=[],
                              exp_name='nd_mixtures', plt_range=[[-4,4], [-4,4]], slice_range=[],
                              process_funcs=[], skip_base=False, traj_hist=True, plot_idx= torch.tensor([6,7]).long())
    '''



if __name__=='__main__':
    run()






