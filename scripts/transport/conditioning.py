import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
import matplotlib.pyplot as plt
import os
from get_data import resample, normal_theta_circle, normal_theta_two_circle, sample_normal, mgan1, mgan2, mgan3,\
    sample_banana, KL, sample_uniform
import pandas as pd

from copy import deepcopy
from fit_kernel import train_kernel,sample_scatter, sample_hmap
import random
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1,1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor


def flip_2tensor(tensor):
    Ttensor = deepcopy(tensor.T)
    Ttensor[0] = deepcopy(tensor.T[1])
    Ttensor[1] = deepcopy(tensor.T[0])
    return Ttensor.T


def mmd(Z, Y, mmd_kernel):
    N = len(Z)
    normalization = N/(N-1)

    k_YY_mean = torch.mean(mmd_kernel(Y,Y))
    k_ZZ = mmd_kernel(Z, Z)
    k_ZZ = k_ZZ - torch.diag(torch.diag(k_ZZ))
    k_ZY =  mmd_kernel(Y, Z)
    return normalization * (torch.mean(k_ZZ)) - 2 * torch.mean(k_ZY) + k_YY_mean


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
        base_params['device'] = self.device
        self.params = base_params

        self.Z_ref= geq_1d(torch.tensor(base_params['Z_ref'], device=self.device, dtype=self.dtype))
        self.Y_ref = geq_1d(torch.tensor(base_params['Y_ref'], device = self.device, dtype = self.dtype))
        self.W_ref = torch.concat([self.Z_ref, self.Y_ref], dim=1)
        self.N = len(self.W_ref)

        self.X_target = geq_1d(torch.tensor(base_params['X_target'], device=self.device, dtype=self.dtype))
        self.Y_target = geq_1d(torch.tensor(base_params['Y_target'], device=self.device, dtype=self.dtype).T)
        self.W_target = torch.concat([self.X_target, self.Y_target], dim=1)

        self.n = len(self.W_target)

        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kref= self.fit_kernel(self.W_ref, self.W_ref)
        self.fit_ktarget = self.fit_kernel(self.W_target, self.W_target)

        self.ref_nugget_matrix = self.params['nugget'] * torch.eye(self.N, device=self.device, dtype=self.dtype)
        self.fit_kref_inv = torch.linalg.inv(self.fit_kref + self.ref_nugget_matrix)

        self.target_nugget_matrix = self.params['nugget'] * torch.eye(self.n, device=self.device, dtype=self.dtype)
        self.fit_ktarget_inv = torch.linalg.inv(self.fit_ktarget + self.target_nugget_matrix)

        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.iters = 0
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.mmd_YY = self.mmd_kernel(self.W_target, self.W_target)

        self.alpha_u = (1/self.N) * torch.ones(self.N, device = self.device, dtype = self.dtype)

        self.alpha_x = self.alpha_u
        self.alpha_y = (1 / self.n) * torch.ones(self.n, device=self.device, dtype=self.dtype)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y


    def init_Z(self):
        return 0 * deepcopy(self.Y_target - self.Y_ref)
        #return torch.zeros(self.Y_ref.shape , device = self.device, dtype = self.dtype)


    def init_alpha_x(self):
        return torch.zeros(self.alpha_u.shape , device=self.device, dtype=self.dtype)


    def get_Lambda(self):
        return self.fit_kref_inv @ (self.Z)


    def map(self, z, y):
        y = geq_1d(torch.tensor(y, device=self.device, dtype=self.dtype))
        z = geq_1d(torch.tensor(z, device=self.device, dtype=self.dtype))
        zy = torch.concat([z, y], dim=1)
        Lambda = self.get_Lambda()
        w = self.fit_kernel(self.W_ref, zy).T @ Lambda
        res_y = w + y
        res = torch.concat([res_y, z], dim = 1)
        return res


    def loss_mmd(self):
        map_vec = torch.concat([self.Z_ref, self.Y_ref + self.Z], dim = 1)
        target = self.W_target

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec,  target)

        alpha_y = self.alpha_y
        alpha_u = self.alpha_u

        Ek_ZZ = alpha_u @ mmd_ZZ @ alpha_u
        Ek_ZY = alpha_u @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        return Ek_ZZ - (2 * Ek_ZY) + Ek_YY


    def loss_reg(self, Z = []):
        if not len(Z):
            Z = self.Z
        try:
            return self.params['reg_lambda'] * torch.trace(Z.T @ self.fit_kref_inv @ Z)
        except BaseException:
            return self.params['reg_lambda'] * Z.T @ self.fit_kref_inv @ Z


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg  = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict



def U_KL(sample, unif_range = [-3,3]):
    try:
        sample = sample.detach().cpu().numpy()
    except BaseException:
        pass

    lb, ub = unif_range
    l= unif_range[1] - unif_range[0]
    sigma = 100
    densitys = []
    for x in sample:
        if lb <= x <= ub:
            densitys.append(1/l)
        else:
            densitys.append(np.exp(-sigma * x**2))
    densitys = np.asarray(densitys)
    KL_div = np.sum(np.log(1 / densitys))
    return KL_div





def param_search(ref_gen, target_gen,  div_f, param_dicts = {}, t_iter = 1000,
                 param_keys = [], N = 1000, exp_name = 'exp', two_part = False):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    ref_sample = torch.tensor(ref_gen(N))
    test_sample_ref = torch.tensor(ref_gen(10 * N))
    target_sample = torch.tensor(target_gen(N)).T
    test_sample_target = torch.tensor(target_gen(10 * N)).T

    if target_sample.shape[0] != max(target_sample.shape):
        target_sample = target_sample.T
        test_sample_target = test_sample_target.T

    Results_dict = {}
    for key in param_keys:
        Results_dict[f'fit_{key}'] = []
        Results_dict[f'mmd_{key}'] = []
    Results_dict['mmd'] = []
    Results_dict['label'] = []


    for i, param_dict in enumerate(param_dicts):
        for key in param_keys:
            Results_dict[f'fit_{key}'].append(param_dict['fit'][key])
            Results_dict[f'mmd_{key}'].append(param_dict['mmd'][key])
        div = light_conditional_transport_exp(ref_sample, target_sample, test_sample_ref, test_sample_target,
                                              div_f= div_f,  t_iter =  t_iter, params = param_dict,
                                              two_part = two_part, save_loc= f'{save_dir}/{i}')

        Results_dict['mmd'].append(div)
        Results_dict['label'].append(i)


    Result_df =  pd.DataFrame.from_dict(Results_dict, orient = 'columns')
    Result_df.to_csv(f'{save_dir}/param_search_res.csv')
    return Result_df



def light_conditional_transport_exp(ref_sample, target_sample, ref_test_sample, target_test_sample, div_f, t_iter = 1000,
                                    params = {'fit': {}, 'mmd': {}}, two_part = False, save_loc =''):

    X_ref = ref_sample[:, 0]
    X_target = target_sample[:, 0]
    X_ref_test = ref_test_sample[:, 0]
    X_target_test = target_test_sample[:, 0]

    Y_ref = ref_sample[:, 1]
    Y_target = target_sample[:, 1]
    Y_ref_test = ref_test_sample[:, 1]
    Y_target_test = target_test_sample[:, 1]

    transport_params = {'X': X_ref, 'Y': X_target, 'fit_kernel_params': params['fit'],
                        'mmd_kernel_params': params['mmd'], 'normalize': False,
                        'reg_lambda': 1e-5, 'print_freq': 100, 'learning_rate': .1,
                        'nugget': 1e-4, 'X_tilde': X_ref, 'alpha_y': [], 'alpha_x': False}

    transport_kernel = TransportKernel(transport_params)


    if not two_part:
        train_kernel(transport_kernel, n_iter=t_iter)

        Z_test = transport_kernel.map(X_ref_test).T
        div = transport_kernel.mmd(Z_test.cuda(), X_target.cuda()).detach().cpu().numpy()
        #div = transport_kernel.mmd(Z_test, X_target).detach().cpu().numpy()

        if save_loc:
            sample_hmap(Z_test.detach().cpu().numpy(), f'{save_loc}slice_sample_map.png', bins=25, d=1, range=[-3.1, 3.1])


    if two_part:
        cond_transport_params = {'Z_ref': X_target, 'Y_ref': Y_ref, 'X_target': X_target, 'Y_target': Y_target,
                                 'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'], 'normalize': False,
                                 'reg_lambda': 1e-5, 'print_freq':100, 'learning_rate': .06,
                                 'nugget': 1e-4, 'X_tilde': X_target, 'alpha_y': [], 'alpha_x': False}

        cond_transport_kernel = CondTransportKernel(cond_transport_params)
        train_kernel(cond_transport_kernel, n_iter= 2 * t_iter)
        sample = cond_transport_kernel.map(X_target_test, Y_ref_test)

        if save_loc:
            sample_scatter(sample.detach().cpu().numpy(), f'{save_loc}slice_sample.png', bins=25, d=2, range=[[-3.1, 3.1], [-1.2, 1.2]])
            sample_hmap(sample.detach().cpu().numpy(), f'{save_loc}slice_sample_map.png', bins=25, d=2, range=[[-3.1, 3.1], [-1.2, 1.2]])

        div = div_f(sample.cuda(), target_sample.cuda()).detach().cpu().numpy()
    return div


def conditional_transport_exp(ref_gen, target_gen, N, t_iter = 801, exp_name= 'exp',  params = {'fit': {}, 'mmd': {}}):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    ref_sample = torch.tensor(ref_gen(N))
    target_sample = torch.tensor(target_gen(N)).T
    if target_sample.shape[0]!= max(target_sample.shape):
        target_sample = target_sample.T

    X_ref = ref_sample[:,0]
    X_target = target_sample[:,0]
    Y_ref = ref_sample[:, 1]
    Y_target = target_sample[:, 1]

    l = l_scale(X_ref)


    if not params['fit']:
        params['fit'] = {'name': 'radial', 'l': l / 5, 'sigma': 1}
    if not params['mmd']:
        params['mmd']= {'name': 'radial', 'l': l / 5, 'sigma': 1}

    transport_params = {'X': X_ref, 'Y':  X_target, 'fit_kernel_params': params['fit'],
                        'mmd_kernel_params':  params['mmd'], 'normalize': False,
                        'reg_lambda': 1e-5, 'print_freq': 100, 'learning_rate': .1,
                         'nugget': 1e-4, 'X_tilde': X_ref, 'alpha_y': [], 'alpha_x': False}

    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=t_iter)

    Z_ref = transport_kernel.map(X_ref).T

    cond_transport_params = {'Z_ref': X_target, 'Y_ref': Y_ref, 'X_target': X_target, 'Y_target': Y_target,
                        'fit_kernel_params': params['mmd'],'mmd_kernel_params': params['fit'], 'normalize': False,
                        'reg_lambda':  1e-5, 'print_freq': 100, 'learning_rate': .06,
                        'nugget': 1e-4, 'X_tilde': X_target, 'alpha_y': [], 'alpha_x': False}

    cond_transport_kernel = CondTransportKernel(cond_transport_params)
    train_kernel(cond_transport_kernel, n_iter= 4 * t_iter)
    sample = cond_transport_kernel.map(X_target, Y_ref)

    slice_samples = []
    N = len(Z_ref)

    slice_vals =  [-1.1, 0, 1.1]
    for z in slice_vals :
        z_slice = torch.full([10000], z)
        idxs = torch.LongTensor(random.choices(list(range(N)), k=10000))

        slice_sample = cond_transport_kernel.map(z_slice,Y_ref[idxs])
        slice_samples.append(slice_sample)

    for i,csample in enumerate(slice_samples):
        csample = csample.T[0].T
        plt.hist(csample.detach().cpu().numpy(), label = f'z = {slice_vals[i]}', bins = 40, range=[-1.5, 1.501])
    plt.legend()
    plt.savefig(f'{save_dir}/cond_hist.png')
    clear_plt()

    target_sample = torch.concat([geq_1d(X_target), geq_1d(Y_target)], dim=1)
    sample = flip_2tensor(sample.detach())

    sample_scatter(sample, f'{save_dir}/cond_sample.png', bins=25, d=2, range = [[-2.5,2.5],[-1,1]])
    sample_hmap(sample, f'{save_dir}/cond_sample_map.png', bins=25, d=2, range = [[-2.5,2.5],[-1,1]])

    sample_scatter(target_sample, f'{save_dir}/target_sample.png', bins=25, d=2, range = [[-2.5,2.5],[-1,1]])
    sample_hmap(target_sample, f'{save_dir}/target_sample_map.png', bins=25, d=2, range = [[-2.5,2.5],[-1,1]])


  #scp -r ald6fd@klone.hyak.uw.edu:/mmfs1/gscratch/dynamicsai/ald6fd/measure_transport/data/kernel_transport/mgan23/ /Users/aloisduston/Desktop/Math/Research/Bambdad/Measure_transport/data/kernel_transport/


def run():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ref_gen = sample_normal
    target_gen = mgan2

    l = l_scale(torch.tensor(ref_gen(5000)[:, 1]))
    #mmd_params = {'name': 'radial', 'l': l/7, 'sigma': 1}

    mmd_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    fit_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    exp_params = {'fit': mmd_params, 'mmd': fit_params}

    conditional_transport_exp(ref_gen, target_gen, N= 5000, t_iter=2001, exp_name='mgan2_exp', params=exp_params)


    '''
    ref_mmd_kernel = get_kernel(mmd_params, device)
    ref_mmd = lambda z,y: mmd(z,y,  ref_mmd_kernel)

    alpha_vals = [1,2]
    #alpha_vals = [1]
    l_log_multipliers = [-1.75,-1.5,-1.25, -1,-.5]

    param_keys = ['l', 'alpha']
    #param_keys = ['l', 'sigma']
    param_dicts = []

    for fit_alpha in alpha_vals:
        for fit_l in l_log_multipliers:
            for mmd_alpha in alpha_vals:
                for mmd_l in l_log_multipliers:
                    fit_dict = {'name': 'r_quadratic', 'l': l*torch.exp(torch.tensor(fit_l)), 'alpha': fit_alpha}
                    mmd_dict = {'name': 'r_quadratic', 'l': l*torch.exp(torch.tensor(mmd_l)), 'alpha': mmd_alpha}
                    #fit_dict = {'name': 'radial', 'l': l*torch.exp(torch.tensor(fit_l)), 'sigma': fit_alpha}
                    #mmd_dict = {'name': 'radial', 'l': l*torch.exp(torch.tensor(mmd_l)), 'sigma': mmd_alpha}

                    param_dict = {'fit': fit_dict, 'mmd': mmd_dict}
                    param_dicts.append(param_dict)

    param_search(ref_gen, target_gen, param_dicts = param_dicts, N = 1000, div_f= ref_mmd,
                 param_keys = param_keys, exp_name='mgan22', two_part = True)
    return True
    '''


if __name__=='__main__':
    run()
