import torch
import torch.nn as nn
from transport_kernel import TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap,seaborne_hmap, process_frames
import os
from copy import deepcopy,copy
from get_data import sample_banana, sample_normal, mgan2, sample_spirals, sample_checkerboard, mgan1, sample_rings, \
    rand_covar, sample_torus, sample_x_torus, sample_sphere, sample_base_mixtures, sample_spheres, sample_swiss_roll

import matplotlib.pyplot as plt
import numpy as np
import random
from lk_sim import get_VL_data, sample_VL_prior
from picture_to_dist import sample_elden_ring,sample_dobby, sample_t_fractal
from datetime import datetime as dt
from seaborn import kdeplot
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.stats as st
from sklearn.cluster import KMeans


def wasserstain_distance(Y1, Y2, full = False):
    if not full:
        Y1 = Y1[:2000]
        Y2 = Y2[:2000]
    n = len(Y1)
    d = len(Y1[0])
    if d > 2:
        return 1
    try:
        Y1 = Y1.detach().cpu().numpy()
    except AttributeError:
        pass
    try:
        Y2 = Y2.detach().cpu().numpy()
    except AttributeError:
        pass
    d = cdist(Y1, Y2)
    assignment = linear_sum_assignment(d)
    mover_distance = (d[assignment].sum() / n)
    return mover_distance


def get_base_stats(gen, N = 10000):
    gen_sample = geq_1d(torch.tensor(gen(N)))
    mu = torch.mean(gen_sample, dim = 0).detach().numpy()
    sigma = torch.std(gen_sample, dim = 0).detach().numpy()
    return mu, sigma


def format(n, n_digits = 6):
    try:
        if n > 1e-3:
            return round(n,n_digits)
        a = '%E' % n
        str =  a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
        scale = str[-4:]
        digits = str[:-4]
        return digits[:min(len(digits),n_digits)] + scale
    except IndexError:
        return n


def shuffle(tensor):
    if geq_1d(tensor).shape[0] <=1:
        return tensor
    else:
        return tensor[torch.randperm(len(tensor))]


def flip(tensor):
    if geq_1d(tensor).shape[0] <=1:
        return tensor
    else:
        return torch.flip(tensor, dims = [0])


def geq_1d(tensor, np = False):
    if np:
        tensor = torch.tensor(tensor)
    if not len(tensor.shape):
        tensor = tensor.reshape(1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    if np:
        tensor = tensor.detach().cpu().numpy()
    return tensor


def replace_zeros(array, eps = 1e-5):
    for i,val in enumerate(array):
        try:
            if np.abs(val) < eps:
                array[i] = 1.0
        except BaseException:
            if torch.abs(val) < eps:
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


def torch_normalize(tensor, keep_axes=[], just_var = False, just_mean = False):
    device = tensor.device
    dtype = tensor.dtype
    tensor = tensor.detach().cpu().numpy()
    normal_tensor = normalize(tensor, keep_axes, just_var, just_mean)
    normal_tensor = torch.tensor(normal_tensor, device = device, dtype = dtype)
    return normal_tensor


def is_normal(tensor, eps = 1e-2):
    mu = torch.mean(tensor, dim  = 0)
    sigma = torch.std(tensor, dim = 0) - 1
    if torch.linalg.norm(mu) > eps:
        return False
    if torch.linalg.norm(sigma) > eps:
        return False
    return True


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
        self.save_dir = '../../../data/transport/exp/'
        self.plt_range = [[-2.5,2.5],[-2.5,2.5]]
        self.vmax = None
        self.bins = 75
        self.mu = 0
        self.sigma = 1

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'


    def mmd(self, map_vec, target):
        return self.submodel_params['mmd_func'](map_vec, target)


    def map_mean(self, x_mu, y_mean, y_var, V_mean, X_mean, fit_kernel):
        x_mean = torch.concat([x_mu, y_mean + y_var], dim=1)
        z_mean = fit_kernel(X_mean, x_mean).T @ V_mean
        return z_mean


    def map_var(self, x_mu, y_eta, y_mean, V_var, X_var, y_var, fit_kernel, var_eps):
        x_var = torch.concat([x_mu, var_eps * flip(y_eta), y_mean + y_var], dim=1)
        V_var = V_var
        z_var = fit_kernel(X_var, x_var).T @ V_var
        return z_var


    def param_map(self, step_idx, param_dict):
        V_mean = torch.tensor(self.submodel_params['V_mean'][step_idx],
                                   device=self.device, dtype=self.dtype)
        V_var = torch.tensor(self.submodel_params['V_var'][step_idx],
                                  device=self.device, dtype=self.dtype)
        fit_kernel = self.submodel_params['fit_kernel'][step_idx]
        X_mean = torch.tensor(self.submodel_params['X_mean'][step_idx],device=self.device, dtype=self.dtype)
        X_var = torch.tensor(self.submodel_params['X_var'][step_idx], device=self.device, dtype=self.dtype)
        var_eps = self.submodel_params['var_eps'][step_idx]

        y_eta = geq_1d(torch.tensor(param_dict['y_eta'], device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(param_dict['x_mu'], device=self.device, dtype=self.dtype))
        y_mean = geq_1d(torch.tensor(param_dict['y_mean'], device=self.device, dtype=self.dtype))
        y_var = geq_1d(torch.tensor(param_dict['y_var'], device=self.device, dtype=self.dtype))

        z_mean = self.map_mean(x_mu, y_mean, y_var, V_mean, X_mean, fit_kernel)
        z_var = self.map_var(x_mu, y_eta, y_mean, V_var, X_var,  y_var, fit_kernel, var_eps)
        z = z_mean + z_var

        y_approx = y_mean + y_var
        param_dict = {'y_eta': y_eta, 'y_mean': y_mean + z_mean, 'y_var': y_var + z_var, 'x_mu': x_mu,
                       'y_approx': y_approx + z, 'y': torch.concat([x_mu, y_approx + z], dim=1)}

        if self.plot_steps:
            plt.figure(figsize=(10,10))
            save_loc = f'{self.save_dir}/frame{step_idx}.png'
            y_map = param_dict['y'].detach().cpu().numpy()*self.sigma + self.mu
            x_plot,y_plot = y_map.T
            plt.hist2d(x_plot, y_plot, density=True, bins=self.bins, range=self.plt_range, vmin=0, vmax=self.vmax)
            plt.savefig(save_loc)
            clear_plt()
        return param_dict


    def c_map(self, x, y, no_x = False):
        param_dict = {'y_eta': y, 'y_mean': deepcopy(y) , 'y_var': 0 * deepcopy(y),
                       'x_mu': x, 'y_approx': deepcopy(y),
                      'y': np.concatenate([geq_1d(x, True), geq_1d(y, True)], axis = 1)}
        self.approx = False
        for step_idx in range(len(self.submodel_params['V_mean'])):
            param_dict = self.param_map(step_idx, param_dict)
            self.approx = True
        if no_x:
            return param_dict['y_approx']
        return param_dict['y']


    def map(self, x = [], y = [], no_x = False):
        return self.c_map(x,y, no_x = no_x)


def get_coeffs(noise_eps, step_num):
    mu_coeff = noise_eps
    approx_coeff = (1 - noise_eps) ** (step_num)
    norm_factor = 1 / (mu_coeff + approx_coeff)
    mu_coeff *= norm_factor
    approx_coeff *= norm_factor
    return mu_coeff, approx_coeff


def get_periodic_coeffs(noise_eps, step_num, k = 9):
    return get_coeffs(noise_eps, k * (step_num % k))


def concat_dicts(base_dict, udpate_dict):
    for key,val in udpate_dict.items():
        if key in base_dict.keys():
            base_dict[key] = torch.concat([base_dict[key],val], dim = 0)
        else:
            base_dict[key] = val
    return base_dict


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
        self.iters = deepcopy(self.params['iters'])

        self.noise_eps = self.params['target_eps']
        self.var_eps =  self.params['var_eps']
        self.step_num = self.params['step_num']

        self.Y_eta = geq_1d(torch.tensor(base_params['Y_eta'], device=self.device, dtype=self.dtype))
        self.Y_eta_flip = flip(self.Y_eta)

        self.Y_mean = geq_1d(torch.tensor(base_params['Y_mean'], device=self.device, dtype=self.dtype))
        self.Y_var = geq_1d(torch.tensor(base_params['Y_var'], device=self.device, dtype=self.dtype))

        self.X_mu = geq_1d(torch.tensor(base_params['X_mu'], device=self.device, dtype=self.dtype))
        self.Y_mu = geq_1d(torch.tensor(base_params['Y_mu'], device=self.device, dtype=self.dtype))

        self.mu_coeff, self.approx_coeff = get_coeffs(self.noise_eps, self.step_num)
        self.Y_mu_approx = geq_1d(torch.tensor(base_params['Y_mu_approx'], device=self.device, dtype=self.dtype))

        if is_normal(self.Y_mu):
            self.Y_mu_noisy = (self.mu_coeff * self.Y_mu) + (self.approx_coeff * torch_normalize(self.Y_mu_approx))
            self.Y_mu_noisy = torch_normalize(self.Y_mu_noisy)
        else:
            self.Y_mu_noisy = (self.mu_coeff * self.Y_mu) + (self.approx_coeff * self.Y_mu_approx)

        self.Y_target = torch.concat([deepcopy(self.X_mu), self.Y_mu_noisy], dim=1)
        self.X_mu = self.X_mu

        self.X_var = torch.concat([self.X_mu, self.var_eps * flip(self.Y_eta), self.Y_mean + self.Y_var], dim=1)
        self.X_mean = torch.concat([self.X_mu, self.Y_mean + self.Y_var], dim=1)

        self.Nx = len(self.X_mean)
        self.Ny = len(self.Y_target)

        self.params['fit_kernel_params']['l'] *= l_scale(self.X_mean).cpu()
        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)

        self.nc = self.params['n_clusters']

        #self.E_mean, self.W_mean_inv, self.mean_centers  = self.nystrom_components(self.X_mean, self.fit_kernel,
                                                                             #init_centers = self.params['mean_centers'])
        #self.E_var, self.W_var_inv, self.var_centers = self.nystrom_components(self.X_var, self.fit_kernel,
                                                                             #init_centers =self.params['var_centers'])

        self.k_mean = self.fit_kernel(self.X_mean, self.X_mean)
        self.k_var = self.fit_kernel(self.X_var, self.X_var)

        self.V_mean = nn.Parameter(self.init_V(), requires_grad=True)
        self.V_var = nn.Parameter(self.init_V(), requires_grad=True)

        self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))
        self.Y_mean_test = geq_1d(torch.tensor(base_params['Y_mean_test'], device=self.device, dtype=self.dtype))
        self.Y_var_test = geq_1d(torch.tensor(base_params['Y_var_test'], device=self.device, dtype=self.dtype))

        self.X_mu_val = geq_1d(torch.tensor(base_params['X_mu_val'], device=self.device, dtype=self.dtype))

        self.X_mu_test = geq_1d(torch.tensor(base_params['X_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_mu_test = geq_1d(torch.tensor(base_params['Y_mu_test'], device=self.device, dtype=self.dtype))
        self.Y_test = torch.concat([self.X_mu_test, self.Y_mu_test], dim=1)

        test_mmd_params = deepcopy(self.params['mmd_kernel_params'])
        test_mmd_params['l'] *= l_scale(self.Y_mu_test).cpu()

        self.test_mmd_kernel = get_kernel(test_mmd_params, self.device)
        self.params['mmd_kernel_params']['l'] *= l_scale(self.Y_mu).cpu()

        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)

        self.alpha_z = self.p_vec(self.Nx)
        self.alpha_y = self.p_vec(self.Ny)

        self.E_mmd_YY = self.alpha_y @  self.mmd_kernel(self.Y_target, self.Y_target) @ self.alpha_y

        self.mmd_lambda = 1
        self.mmd_lambda = (1 / self.loss_mmd().detach())

        self.reg_lambda = self.params['reg_lambda'] * self.mmd_lambda

        goal_mmd = self.mmd(self.Y_target, self.Y_test)
        goal_emd = wasserstain_distance(self.Y_target, self.Y_test)

        print(f"Transport {self.step_num}: Goal mmd is {format(float(goal_mmd.detach().cpu()))},"
              f"Goal emd is {goal_emd}")


    def nystrom_components(self, X, k, include_approx = False, return_dict = False, init_centers = []):
        n_clutsters = self.nc
        if len(init_centers):
            kmeans = KMeans(n_clusters=n_clutsters, n_init=10
                            ,init = init_centers).fit(X.detach().cpu().numpy())
        else:
            kmeans = KMeans(n_clusters=n_clutsters, random_state=0,n_init=10).fit(X.detach().cpu().numpy())
        centers = kmeans.cluster_centers_
        centers_tensor = geq_1d(torch.tensor(centers, device=self.device, dtype=self.dtype))

        E = k(X, centers_tensor)
        W = k(centers_tensor, centers_tensor)
        nugget_matrix = torch.eye(self.nc, device  = self.device, dtype = self.dtype) * self.params['nugget']
        W_inv = torch.linalg.inv(W + nugget_matrix)

        if not return_dict:
            return E, W_inv, centers

        u, s, v = torch.svd(E)
        component_dict = {'E': E, 'W': W, 'W_inv': W_inv,
                        'U_1': v.mT, 'U_2': u, 'S': torch.diag_embed(s)}
        component_dict['EU_1'] = E @ component_dict['U_1'].T
        if include_approx:
            component_dict['k_approx'] = E @ W_inv @ E.T
        return component_dict


    def total_grad(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


    def p_vec(self, n, density = 0):
        if not density:
            density = 1/n
        return torch.full([n], density, device=self.device, dtype=self.dtype)


    def norm_vec(self, vec):
        return vec/torch.linalg.norm(vec)


    def prob_add(self, t_1, t_2, p = .001):
        T = []
        for i in range(len(t_1)):
            if random.random() < p:
                T.append(t_2[i])
            else:
                T.append(t_1[i])
        return torch.tensor(T, device= self.device).reshape(t_1.shape)


    def init_V(self):
        V = torch.zeros(self.Y_eta.shape, device=self.device, dtype=self.dtype)
        return V


    def get_Z_mean(self):
        k_XX = self.k_mean
        V = self.V_mean
        #E = self.E_mean
        #W_inv = self.W_mean_inv
        #return E @ W_inv @ E.T @ V
        return k_XX @ V


    def get_Z_var(self):
        k_XX = self.k_var
        #E = self.E_var
        #W_inv = self.W_var_inv
        V = self.V_var
        # E = self.E_var
        # W_inv = self.W_var_inv
       # return E @ W_inv @ E.T @ V
        return k_XX @ V


    def map_mean(self, x_mu, y_mean, y_var):
        x_mean = torch.concat([x_mu, y_mean + y_var], dim=1)
        z_mean = self.fit_kernel(self.X_mean, x_mean).T @ self.V_mean
        return z_mean


    def map_var(self, x_mu, y_eta, y_mean, y_var):
        x_var = torch.concat([x_mu, self.var_eps * flip(y_eta), y_mean + y_var], dim=1)
        z_var = self.fit_kernel(self.X_var, x_var).T @ self.V_var
        return z_var


    def map(self, X_mu, Y_eta, Y_mean = [], Y_var = []):
        if not len(Y_mean):
            Y_mean = deepcopy(Y_eta)
            Y_var = 0 * deepcopy(Y_mean)
        batch_size = self.params['batch_size']
        N = len(X_mu)
        map_dict = {}
        batch_idxs = [torch.tensor(list(range((j * batch_size), min((j + 1) * batch_size, N)))).long()
                      for j in range(1 + N // batch_size)]
        for batch_idx in batch_idxs:
            x_mu ,y_eta ,y_mean ,y_var = X_mu[batch_idx],Y_eta[batch_idx],\
                                      Y_mean[batch_idx], Y_var[batch_idx]
            batch_dict = self.map_batch(x_mu, y_eta, y_mean, y_var)
            map_dict = concat_dicts(map_dict, batch_dict)
        return map_dict


    def map_batch(self, x_mu, y_eta, y_mean = 0, y_var = 0):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        y_mean = geq_1d(torch.tensor(y_mean, device=self.device, dtype=self.dtype))
        y_var = geq_1d(torch.tensor(y_var, device=self.device, dtype=self.dtype))

        z_mean = self.map_mean(x_mu, y_mean, y_var)
        z_var = self.map_var(x_mu, y_eta, y_mean, y_var)
        z = z_mean + z_var

        y_approx = y_mean + y_var
        return_dict = {'y_eta': y_eta, 'y_mean': y_mean + z_mean, 'y_var': y_var + z_var,
                       'y_approx': y_approx + z, 'y': torch.concat([x_mu, z + y_approx], dim = 1)}
        return return_dict



    def mmd(self, map_vec, target, test = True, pre_process = True):
        if pre_process:
            map_vec = geq_1d(torch.tensor(map_vec, device=self.device, dtype=self.dtype))
            target = geq_1d(torch.tensor(target, device=self.device, dtype=self.dtype))
        N  = len(map_vec)
        batch_size = self.params['batch_size']
        batch_idxs = [torch.tensor(list(range((j * batch_size), min((j + 1) * batch_size, N)))).long()
                      for j in  range( N//batch_size)]
        mmd = 0
        n = 0
        for x_idx in batch_idxs:
            for y_idx in batch_idxs:
                n += len(x_idx) * len(y_idx)
                mmd += self.batch_mmd(map_vec, target, x_idx, y_idx, test = test)
        return mmd/n


    def batch_mmd(self, map_vec, target, x_idx, y_idx, test = True):
        if test:
            K_mmd = self.test_mmd_kernel
        else:
            K_mmd = self.mmd_kernel

        mmd_ZZ = K_mmd(map_vec[x_idx], map_vec[y_idx])
        mmd_ZY = K_mmd(map_vec[x_idx], target[y_idx])
        mmd_YY = K_mmd(target[x_idx], target[y_idx])

        Ek_ZZ =  torch.sum(mmd_ZZ)
        Ek_ZY =  torch.sum(mmd_ZY)
        Ek_YY =  torch.sum(mmd_YY)

        return (Ek_ZZ - (2 * Ek_ZY) + Ek_YY)


    def loss_mmd(self):
        Z_mean = self.get_Z_mean()
        Z_var = self.get_Z_var()

        Y_approx = self.Y_var + self.Y_mean + Z_mean + Z_var
        map_vec = torch.concat([self.X_mu, Y_approx], dim=1)
        target = self.Y_target

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, target)

        alpha_z = self.alpha_z
        alpha_y = self.alpha_y

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        mmd = Ek_ZZ - (2 * Ek_ZY) + Ek_YY
        return mmd * self.mmd_lambda

        #mmd = self.mmd(map_vec, target, test = False, pre_process = False)
        #return mmd * self.mmd_lambda


    def loss_reg_mean(self):
        V = self.V_mean
        k_XX = self.k_mean
        #E = self.E_mean
        #W_inv = self.W_mean_inv
        #reg_loss = torch.trace(V.T @ E @ W_inv @ E.T @ V)
        reg_loss = torch.trace(V.T @ k_XX @ V)
        return  self.reg_lambda * reg_loss


    def loss_reg_var(self):
        V = self.V_var
        k_XX = self.k_var
        #E = self.E_var
        #W_inv = self.W_var_inv
        #reg_loss = torch.trace(V.T @ E @ W_inv @ E.T @ V)
        reg_loss = torch.trace(V.T @ k_XX @ V)
        return self.reg_lambda * reg_loss


    def loss_test(self):
        x_mu = self.X_mu_val
        y_eta = self.Y_eta_test
        y_mean = self.Y_mean_test
        y_var = self.Y_var_test
        target = self.Y_test

        map_vec = self.map(x_mu, y_eta, y_mean, y_var)['y']
        test_mmd = self.mmd(map_vec, target, test = True)
        test_emd = wasserstain_distance(map_vec, target)
        return  test_mmd, test_emd


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg_var() + self.loss_reg_mean()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def cond_kernel_transport(X_mu, Y_mu, Y_eta, Y_mean, Y_var, X_mu_test, Y_eta_test, Y_mu_test, X_mu_val,
                        Y_mean_test, Y_var_test, Y_mu_approx, params, iters=-1, mmd_lambda=0, step_num = 1,
                        reg_lambda=1e-7, grad_cutoff = .0001, n_iter = 200, target_eps = 1, var_eps = 1/3, nc = 500,
                        mean_centers = [], var_centers = []):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'Y_var': Y_var, 'Y_mean': Y_mean, 'nugget': 1e-4,
                        'fit_kernel_params': deepcopy(params['fit']), 'mmd_kernel_params': deepcopy(params['mmd']),
                        'print_freq': 10, 'learning_rate': .001, 'reg_lambda': reg_lambda, 'var_eps': var_eps,
                        'Y_eta_test': Y_eta_test, 'X_mu_test': X_mu_test, 'Y_mu_test': Y_mu_test, 'X_mu_val': X_mu_val,
                        'Y_mean_test': Y_mean_test, 'mmd_lambda': mmd_lambda, 'n_clusters': min(nc, len(X_mu) - 1),
                        'Y_var_test': Y_var_test, 'iters': iters, 'grad_cutoff': grad_cutoff, 'step_num': step_num,
                        'Y_mu_approx': Y_mu_approx, 'target_eps': target_eps, 'batch_size': min(len(X_mu), 5000),
                        'mean_centers': mean_centers, 'var_centers': var_centers}

    model = CondTransportKernel(transport_params)
    model, loss_dict = train_kernel(model, n_iter= n_iter)
    return model, loss_dict


def dict_not_valid(loss_dict):
    for key,val_list in loss_dict.items():
        for value in val_list:
            if np.isnan(value) or value < -1:
                return True
    return False


def comp_cond_kernel_transport(X_mu, Y_mu, Y_eta, Y_eta_test, X_mu_test, Y_mu_test, X_mu_val, params,
                               target_eps = .1, n_transports=75, reg_lambda=1e-7, n_iter = 200,var_eps = 1/3,
                               grad_cutoff = .0001, approx_path = False, nc = 500):
    param_keys = ['fit_kernel','V_mean', 'X_mean',  'V_var', 'X_var', 'var_eps']
    models_param_dict = {key: [] for key in param_keys}
    iters = 0

    Y_mean = deepcopy(Y_eta)
    Y_var = 0 * deepcopy(Y_mean)
    Y_mean_test = deepcopy(Y_eta_test)
    Y_var_test = 0 * deepcopy(Y_mean_test)

    approx = False
    mmd_lambda = 0
    Y_mu_approx = Y_eta
    step_num = 1
    mean_centers = []
    var_centers = [ ]
    for i in range(n_transports):
        model, loss_dict = cond_kernel_transport(X_mu, Y_mu, Y_eta, Y_mean, Y_var, X_mu_test, Y_eta_test, Y_mu_test,
                                    X_mu_val, Y_mean_test, Y_var_test, Y_mu_approx, n_iter = n_iter, params=params,
                                    mmd_lambda=mmd_lambda, reg_lambda=reg_lambda,var_eps = var_eps, step_num = step_num,
                                    grad_cutoff = grad_cutoff, target_eps = target_eps, iters=iters, nc = nc,
                                    mean_centers = mean_centers, var_centers = var_centers)

        if dict_not_valid(loss_dict):
            break

        models_param_dict['V_mean'].append(model.V_mean.detach().cpu().numpy())
        models_param_dict['V_var'].append(model.V_var.detach().cpu().numpy())
        models_param_dict['fit_kernel'].append(model.fit_kernel)
        models_param_dict['X_mean'].append(model.X_mean.detach().cpu().numpy())
        models_param_dict['X_var'].append(model.X_var.detach().cpu().numpy())
        models_param_dict['var_eps'].append(model.var_eps)
        mmd_lambda = model.mmd_lambda

        if i == 0:
            models_param_dict['mmd_func'] = model.mmd

        Y_mean = model.Y_mean + model.get_Z_mean()
        Y_var = model.Y_var + model.get_Z_var()


        test_map_dict = model.map(X_mu_val, Y_eta_test, Y_mean_test, Y_var_test)
        Y_mean_test, Y_var_test = test_map_dict['y_mean'], test_map_dict['y_var']

        step_num += 1
        mean_centers = model.mean_centers
        var_centers = model.var_centers
        iters = model.iters
        if approx_path:
            Y_mu_approx = Y_mean + Y_var

    for key in param_keys:
        models_param_dict[key] = models_param_dict[key]
    return Comp_transport_model(models_param_dict)


def get_idx_tensors(idx_lists):
    return [torch.tensor(idx_list).long() for idx_list in idx_lists]


def zero_pad(array):
    zero_array = np.zeros([len(array), 1])
    return np.concatenate([zero_array, array], axis=1)


def train_cond_transport(ref_gen, target_gen, params, N = 4000,  process_funcs=[],var_eps = 1/3, approx_path = True,
                         cond_model_trainer=cond_kernel_transport, idx_dict={}, reg_lambda=1e-7, n_transports=100,
                         nc = 500):
    ref_sample = ref_gen(N)
    target_sample = target_gen(N)

    N_test = N
    test_sample = ref_gen(N_test)
    test_target_sample = target_gen(N_test)
    test_val_sample = target_gen(N_test)

    if len(process_funcs):
        forward = process_funcs[0]
        target_sample = forward(target_sample)

    ref_idx_tensors = idx_dict['ref']
    target_idx_tensors = idx_dict['target']
    cond_idx_tensors = idx_dict['cond']
    trained_models = []

    for i in range(len(ref_idx_tensors)):
        X_mu = target_sample[:, cond_idx_tensors[i]]
        X_mu_test = test_target_sample[:, cond_idx_tensors[i]]
        X_mu_val = test_val_sample[:, cond_idx_tensors[i]]

        Y_mu = target_sample[:, target_idx_tensors[i]]
        Y_mu_test = test_target_sample[:, target_idx_tensors[i]]

        Y_eta = ref_sample[:, ref_idx_tensors[i]]
        Y_eta_test = test_sample[:, ref_idx_tensors[i]]

        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, Y_eta_test, X_mu_test, Y_mu_test, X_mu_val,
                                                 params=params, reg_lambda=reg_lambda,  n_transports=n_transports,
                                                 var_eps = var_eps, approx_path =  approx_path, nc = nc))

    return trained_models


def MC_cond_sample(target_gen, slice_val, cond_idx, N = 5000, eps = 5e-3):
    cond_sample = np.empty([0] + list(target_gen(5).shape[1:]))
    while len(cond_sample) < N:
        target_sample = target_gen(10 * N)
        cond_bool = np.linalg.norm(geq_1d(target_sample[:, cond_idx] - slice_val, np = True), axis = 1) < eps
        cond_sample = np.concatenate([cond_sample, target_sample[cond_bool]], axis = 0)
    return cond_sample[:N]




def compositional_gen(trained_models, ref_sample, target_sample, idx_dict, plot_steps = False, sigma = 1, mu = 0):
    ref_indexes = idx_dict['ref']
    cond_indexes = idx_dict['cond']
    target_indexes = idx_dict['target']

    X = geq_1d(0 * deepcopy(target_sample))
    X[:, cond_indexes[0]] += deepcopy(target_sample)[:, cond_indexes[0]]
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, ref_indexes[i]]
        target_shape = X[:, target_indexes[i]].shape
        model.plot_steps = plot_steps
        X[:, target_indexes[i]] = model.map(X[:, cond_indexes[i]], Y_eta, no_x=True) \
            .detach().cpu().numpy().reshape(target_shape)
        model.plot_steps = False
    return X * sigma + mu


def conditional_transport_exp(ref_gen, target_gen, N=4000, vmax=None, exp_name='exp', plt_range=None, bins=70,
                              process_funcs=[], N_plot=0, cond_model_trainer=comp_cond_kernel_transport,
                              skip_idx=0, plot_idx=[], n_transports=75, idx_dict={},plot_steps = False,
                              reg_lambda = 1e-7, mu = 0, sigma = 1,var_eps = 1/3, approx_path = True, nc = 500):
    save_dir = f'../../data/transport/{exp_name}'
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
            idx_dict['cond'].append(list(range(k + 1)))
            idx_dict['target'].append([k])

    idx_dict = {key: get_idx_tensors(val) for key, val in idx_dict.items()}
    idx_dict = {key: val[skip_idx:] for key, val in idx_dict.items()}

    trained_models = train_cond_transport(N=N, ref_gen=ref_gen, target_gen=target_gen, params=exp_params,
                                          cond_model_trainer=cond_model_trainer, n_transports=n_transports,
                                          process_funcs=process_funcs, idx_dict=idx_dict,reg_lambda = reg_lambda,
                                          var_eps = var_eps, approx_path=approx_path, nc = nc)

    for model in trained_models:
        model.save_dir = save_dir
        model.plt_range = plt_range
        model.vmax = vmax
        model.mu = mu
        model.sigma = sigma
        model.bins = bins
    test_target_sample = target_gen(N)
    test_ref_sample = ref_gen(N)
    test_gen_sample = compositional_gen(trained_models, test_ref_sample, test_target_sample, idx_dict,
                                        plot_steps=False, mu=mu, sigma=sigma)
    test_target_sample = test_target_sample * sigma + mu
    test_mmd = float(trained_models[0].mmd(test_gen_sample, test_target_sample).detach().cpu())
    test_emd = wasserstain_distance(test_gen_sample, test_target_sample, full = True)
    try:
        cref_sample = deepcopy(test_ref_sample)
        cref_sample[:, idx_dict['cond'][0]] += test_target_sample[:, idx_dict['cond'][0]]

        base_mmd = float(trained_models[0].mmd(cref_sample, test_target_sample).detach().cpu())
        base_emd = wasserstain_distance(cref_sample, test_target_sample, full = True)

        ntest_mmd = test_mmd / base_mmd
        ntest_emd = test_emd / base_emd

        print_str1 = f'Test mmd :{format(test_mmd)}, Base mmd: {format(base_mmd)}, NTest mmd :{format(ntest_mmd)}, '
        print_str2 = f'Test emd :{format(test_emd)}, Base emd: {format(base_emd)}, NTest mmd :{format(ntest_emd)}'
    except BaseException:
        print_str1 = f'Test mmd :{format(test_mmd)}, '
        print_str2 = f'Test emd :{format(test_emd)}'

    print_str = print_str1 + print_str2
    print(print_str)
    os.system(f'echo {print_str} > {save_dir}/test_res.txt')

    if not N_plot:
        N_plot = min(10 * N, 4000)

    plot_target_sample = target_gen(N_plot)
    plot_ref_sample = ref_gen(N_plot)

    plot_gen_sample = compositional_gen(trained_models, plot_ref_sample, plot_target_sample, idx_dict,
                                        plot_steps=plot_steps, mu=mu, sigma=sigma)
    plot_target_sample = plot_target_sample * sigma + mu

    if len(process_funcs):
        backward = process_funcs[1]
        plot_gen_sample = backward(plot_gen_sample.cpu())

    if not len(plot_idx):
        return trained_models, idx_dict

    plot_gen_sample = plot_gen_sample[:, plot_idx]
    plot_target_sample = plot_target_sample[:, plot_idx]
    try:
        d = len(plot_gen_sample[0])
    except TypeError:
        d = 1

    sample_hmap(plot_gen_sample, f'{save_dir}/gen_map_final.png', bins=bins, d=d, range=plt_range, vmax=vmax)
    sample_hmap(plot_target_sample, f'{save_dir}/target_map.png', bins=bins, d=d, range=plt_range, vmax=vmax)

    return trained_models, idx_dict


def two_d_exp(ref_gen, target_gen, N=4000, plt_range=None, process_funcs=[], normal = True,
              slice_range=None, N_plot=4000, slice_vals=[], bins=70, exp_name='exp', skip_idx=1, nc = 500,
              vmax=None, n_transports=75, reg_lambda=1e-7, plot_steps = False, var_eps = 1/3, approx_path=True):
    save_dir = f'../../data/transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass


    mu, sigma = 0, 1
    if normal:
        mu,sigma = get_base_stats(target_gen, 10000)
        normal_target_gen = lambda N: normalize(target_gen(N))
    else:
        normal_target_gen = target_gen

    plot_idx = torch.tensor([0, 1]).long()
    trained_models, idx_dict = conditional_transport_exp(ref_gen, normal_target_gen, N=N, vmax=vmax,N_plot=N_plot,
                                                         skip_idx=skip_idx, exp_name=exp_name, plot_steps = plot_steps,
                                                         n_transports=n_transports, process_funcs=process_funcs,
                                                         plt_range=plt_range,  bins=bins, mu = mu, sigma = sigma,
                                                         plot_idx=plot_idx, reg_lambda=reg_lambda, var_eps = var_eps,
                                                         approx_path = approx_path, nc = nc)

    cmu, csigma = 0,1
    if normal:
        cmu = mu[idx_dict['cond'][0]]
        csigma = sigma[idx_dict['cond'][0]]
    normal_slice_vals = (np.asarray(slice_vals)-cmu)/csigma

    for i,slice_val in enumerate(normal_slice_vals):
        ref_sample = ref_gen(N_plot)
        ref_slice_sample = normal_target_gen(N_plot)
        ref_slice_sample[:, idx_dict['cond'][0]] = slice_val
        slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict,
                                         mu= mu, sigma = sigma)
        plt.hist(slice_sample[:, 1], bins= bins, range=slice_range, label=f'x = {slice_vals[i]}')

    if len(slice_vals):
        plt.legend()
        plt.savefig(f'{save_dir}/slice_posteriors.png')
        clear_plt()

    if plot_steps:
        process_frames(save_dir)
    return True


def spheres_exp(N=4000, exp_name='spheres_exp', n_transports=100, N_plot = 0,
                normalize_data = False, approx_path = False, nc = 500):
    n = 10
    ref_gen = sample_normal
    mu, sigma = 0,1
    target_gen = sample_spheres
    if normalize_data:
        mu, sigma = get_base_stats(sample_spheres, 10000)
        target_gen = lambda N: normalize(sample_spheres(N=N, n=n))

    idx_dict = {'ref': [[0, 1]],
                'cond': [list(range(2, 2 + (2 * n)))],
                'target': [[0, 1]]}

    plt_range = [[.6, 1.5], [-1.2, 1.2]]
    plot_idx = torch.tensor([0, 1]).long()
    skip_idx = 0
    if not N_plot:
        Np = min(10 * N, 4000)
    else:
        Np = N_plot
    trained_models, idx_dict = conditional_transport_exp(ref_gen, target_gen, N=N, N_plot=Np, approx_path = approx_path,
                                                         skip_idx=skip_idx, exp_name=exp_name, process_funcs=[],
                                                         cond_model_trainer=comp_cond_kernel_transport, vmax=None,
                                                         plot_idx=plot_idx, plt_range=plt_range, idx_dict=idx_dict,
                                                         n_transports=n_transports, mu = mu, sigma=sigma, nc = nc)

    slice_vals = np.asarray([[1, .0], [1, .4],  [1, .6], [1, .75]])
    save_dir = f'../../data/transport{exp_name}'
    fig, axs = plt.subplots(sharex="col", sharey="row", figsize = (18,4))
    plt.rcParams.update({'font.size': 12})
    ns = len(slice_vals)
    for j in range(10):
        for i, slice_val in enumerate(slice_vals):
            ref_sample = ref_gen(Np)
            RX = np.full((Np, 2), slice_val)
            ref_slice_sample = sample_spheres(N=Np, n=n, RX=RX)

            if j != 0:
                ref_slice_sample = np.full(ref_slice_sample.shape, ref_slice_sample[0])

            if normalize_data:
                ref_slice_sample = (ref_slice_sample - mu) / sigma
            slice_sample = compositional_gen(trained_models, ref_sample, ref_slice_sample, idx_dict,
                                             sigma = sigma, mu = mu)
            x,y = slice_sample[:, np.asarray([0, 1])].T


            plt.title(f'r = {slice_val[0]}, x = {slice_val[1]}')
            plt.subplot(1, ns, i + 1)
            kdeplot(x=x, y=y, fill=True, bw_adjust=0.4, cmap='Blues')
            plt.xlim(plt_range[0][0], plt_range[0][1])
            plt.ylim(plt_range[1][0], plt_range[1][1])

        plt.legend()
        plt.tight_layout(pad=0.3)
        plt.savefig(f'{save_dir}/slice_plots{j}.png')
        clear_plt()
    return True



def plot_lv_matrix(x_samps, limits, xtrue=None, symbols=None, save_dir = '.', label = ''):
    #plt.rc('text', usetex=True)
    plt.rc('font', size=12)
    dim = x_samps.shape[1]
    plt.figure(figsize=(9, 9))

    for i in range(dim):
        for j in range(i + 1):
            ax = plt.subplot(dim, dim, (i * dim) + j + 1)
            if i == j:
                plt.hist(x_samps[:, i], bins=40, density=True)
                if xtrue is not None:
                    plt.axvline(xtrue[i], color='r', linewidth=3)
                plt.xlim(limits[i])
            else:
                plt.plot(x_samps[:, j], x_samps[:, i], '.k', markersize=.04, alpha=0.1)
                if xtrue is not None:
                    plt.plot(xtrue[j], xtrue[i], '.r', markersize=8, label='Truth')
                # Peform the kernel density estimate
                xlim = limits[j]
                ylim = limits[i]
                xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = st.gaussian_kde(x_samps[:, [j, i]].T)
                f = np.reshape(kernel(positions), xx.shape)
                ax.contourf(xx, yy, f, cmap='Blues')
                plt.ylim(limits[i])
            plt.xlim(limits[j])
            if symbols is not None:
                if j == 0:
                    plt.ylabel(symbols[i], size=20)
                if i == len(xtrue) - 1:
                    plt.xlabel(symbols[j], size=20)
    plt.savefig(f'{save_dir}/DLV_MCMCposterior{label}.png', bbox_inches='tight')
    clear_plt()
    return True


def lv_exp(N=10000, Yd=18, normal=True, exp_name='lv_exp', n_transports=100,
           N_plot = 0, approx_path = True, nc = 500):
    ref_gen = lambda N: sample_normal(N, 4)
    target_gen = lambda N: get_VL_data(N, normal=False, Yd=Yd)

    if normal:
        normal_target_gen = lambda N: get_VL_data(N, normal=True, Yd=Yd)
        mu, sigma = get_base_stats(target_gen, 10000)
    else:
        normal_target_gen = target_gen
        mu,sigma = 0,1


    idx_dict = {'ref': [[0, 1, 2, 3]],
                'cond': [list(range(4, 4 + Yd))],
                'target': [[0, 1, 2, 3]]}

    save_dir = f'../../data/transport{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    skip_idx = 0
    if not N_plot:
        N_plot = min(10 * N, 4000)
    trained_models, idx_dict = conditional_transport_exp(ref_gen, normal_target_gen, N=N, N_plot=N_plot ,sigma = sigma,
                                                         skip_idx=skip_idx, exp_name=exp_name, process_funcs=[],nc = nc,
                                                         cond_model_trainer=comp_cond_kernel_transport, vmax=None,
                                                         plt_range=None, n_transports=n_transports, idx_dict=idx_dict,
                                                         plot_idx=[], var_eps = 1/3, approx_path = approx_path, mu = mu)
    mu, sigma = get_base_stats(target_gen, 10000)

    slice_val = np.asarray([.8319, .0413, 1.0823, .0399])
    X = np.full((N_plot, 4), slice_val)

    ref_slice_sample = get_VL_data(N_plot, X=X, Yd=Yd, normal=False, T=20)
    ref_sample = ref_gen(N_plot)
    return True
    for j in range(10):
        if j != 0:
            n_ref_slice_sample = np.full(ref_slice_sample.shape, ref_slice_sample[j])
        else:
            n_ref_slice_sample = deepcopy(ref_slice_sample)


        n_ref_slice_sample =  (n_ref_slice_sample - mu)/sigma

        slice_sample = compositional_gen(trained_models, ref_sample, n_ref_slice_sample,
                                         idx_dict, mu = mu, sigma = sigma)[:, :4]

        symbols = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
        limits = [[0.5, 1.3], [0.02, 0.07], [0.7, 1.5], [0.025, 0.065]]
        xtrue = slice_val
        plot_lv_matrix(slice_sample, limits, xtrue, symbols, save_dir, label = j)
    return True


def test_panel(plot_steps = False, approx_path = False, N = 10000, test_name = 'test',  n_transports = 100, k = 1,
               test_keys = ['mgan1','mgan2','swiss','checker','spiral','elden','spheres', 'lv'], N_plot = 1e5, nc = 500):
    test_dir = f'../../data/transport/{test_name}'
    try:
        os.mkdir(test_dir)
    except OSError:
        pass
    for i in range(k):
        i_str = i if k > 1 else ''
        if 'banana' in test_keys:
            done = 0
            while  done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen= sample_banana, N=N,
                              exp_name=f'/{test_name}/banana_{i_str}', n_transports=n_transports, slice_vals=[-1, 0, 1],
                              plt_range=[[-2.5, 2.5], [-1, 3]], slice_range=[-1.5, 1.5], vmax=1.2, skip_idx=1,
                              N_plot=N_plot, plot_steps=plot_steps, normal=True, bins=100, var_eps=1/3,
                              approx_path = approx_path, nc = nc)
                    done += 3
                except torch._C._LinAlgError:
                    done += 1
                    pass
        if 'mgan1' in test_keys:
            done = 0
            while  done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=mgan1, N=N, exp_name=f'/{test_name}/mgan1_{i_str}',
                            n_transports=n_transports, slice_vals=[-1, 0, 1], plt_range=[[-2.5, 2.5], [-1, 3]],
                            slice_range=[-1.5, 1.5], vmax=1.2, skip_idx=1, N_plot=N_plot, plot_steps=plot_steps,
                            normal=True, bins=100, var_eps=1/3, approx_path = approx_path, nc = nc)
                    done += 3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 'mgan2' in test_keys:
            done = 0
            while  done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=mgan2, N=N, exp_name=f'/{test_name}/mgan2_{i_str}',
                              n_transports= n_transports,  slice_vals=[-1, 0, 1], plt_range=[[-2.5, 2.5], [-1.05, 1.05]],
                              slice_range=[-1.5, 1.5], vmax=8,skip_idx=1, N_plot=N_plot, plot_steps=plot_steps, normal=True,
                              bins=100,var_eps=1/2, approx_path = approx_path, nc = nc)
                    done += 3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 'swiss' in test_keys:
            done = 0
            while  done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=sample_swiss_roll, N=N, exp_name=f'/{test_name}/swiss_{i_str}',
                              n_transports= n_transports, slice_vals=[.7], plt_range=[[-3, 3], [-3, 3]], slice_range=[-3, 3],
                              vmax=.35,  skip_idx=1, N_plot=N_plot, plot_steps=plot_steps, normal=True, bins=100, var_eps=1/3,
                              approx_path = approx_path, nc = nc)
                    done += 3
                except torch._C._LinAlgError:
                    done += 1
                    pass


        if 'checker' in test_keys:
            done = 0
            while done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=sample_checkerboard, N=N, n_transports= n_transports,
                              exp_name=f'/{test_name}/checker{i_str}', slice_vals=[-1, 0, 1],skip_idx=1,nc = nc,
                              plt_range=[[-4.4, 4.4], [-4.1, 4.1]], slice_range=[-4.4, 4.4], vmax=.12,N_plot=N_plot,
                              plot_steps=plot_steps, normal=True, bins=100, var_eps=1/3, approx_path = approx_path)
                    done +=3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 'spiral' in test_keys:
            done = 0
            while done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=sample_spirals, N=N, exp_name=f'/{test_name}/spiral_{i_str}',
                              n_transports= n_transports, slice_vals=[0], plt_range=[[-3, 3], [-3, 3]], slice_range=[-3,3],
                              vmax=.33,skip_idx=1, N_plot=N_plot, plot_steps=plot_steps , normal=True, bins=100, var_eps=1/3,
                              approx_path = approx_path,nc = nc)
                    done +=3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 'elden' in test_keys:
            done = 0
            while  done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=sample_elden_ring, N=N, exp_name=f'/{test_name}/elden_{i_str}',
                              n_transports= n_transports, slice_vals=[], plt_range=[[-1, 1], [-1.05, 1.05]],
                              slice_range=[-1.5, 1.5], vmax=8, skip_idx=1, N_plot=N_plot, plot_steps=plot_steps, normal=True,
                              bins=100, var_eps=1/12, approx_path = approx_path, nc = nc)
                    done +=3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 't_fractal' in test_keys:
            done = 0
            while  done <= 2:
                try:
                    two_d_exp(ref_gen=sample_normal, target_gen=sample_t_fractal, N=N,
                              exp_name=f'/{test_name}/t_fractal_{i_str}',  n_transports= n_transports, slice_vals=[],
                              plt_range=[[-1, 1], [-.95, .95]],  slice_range=[-1.5, 1.5], vmax=4.5, skip_idx=1,
                              N_plot=N_plot, plot_steps=plot_steps, normal=True, bins=200, var_eps=1/12,
                              approx_path = approx_path, nc = nc)
                    done +=3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 'lv' in test_keys:
            done = 0
            while done < 2:
                try:
                    lv_exp(min(N,8000), exp_name=f'/{test_name}/lv_{i_str}', normal = True,
                        approx_path = approx_path, n_transports = n_transports, N_plot= 30000)
                    done +=3
                except torch._C._LinAlgError:
                    done += 1
                    pass

        if 'spheres' in test_keys:
            done = 0
            while done < 2:
                try:
                    spheres_exp(min(N,8000), exp_name=f'/{test_name}/spheres_{i_str}', normalize_data=False,
                                approx_path = approx_path, n_transports=n_transports, N_plot = 30000, nc = nc)
                    done +=3
                except torch._C._LinAlgError:
                    done += 1
                    pass


def run():
    test_panel(N=5000, n_transports=70, k=1, approx_path=False, test_name='inducing_test',
               test_keys=['elden'], plot_steps = True, nc = 2000)


if __name__ == '__main__':
    run()