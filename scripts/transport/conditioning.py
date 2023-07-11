import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, normalize,get_kernel, clear_plt
import matplotlib.pyplot as plt
import os
from get_data import resample, normal_theta_circle, normal_theta_two_circle, sample_normal, mgan1, mgan2, mgan3

from copy import deepcopy
from fit_kernel import train_kernel,sample_scatter, sample_hmap
import random

import warnings
warnings.filterwarnings("ignore")


def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1,1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor


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
        res = torch.concat([z, res_y], dim = 1)
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


def conditional_transport_exp(ref_gen, target_gen, N, t_iter = 1001, exp_name= 'exp'):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    ref_sample = torch.tensor(ref_gen(N))
    target_sample = torch.tensor(target_gen(N)).T
    if target_sample.shape[0]!= max(target_sample.shape):
        target_sample = target_sample.T


    #X_ref = ref_sample[:,0]
    #X_target = target_sample[:,0]
    #Y_ref = ref_sample[:, 1]
    #Y_target = target_sample[:, 1]

    X_ref = ref_sample[:,1]
    X_target = target_sample[:,1]
    Y_ref = ref_sample[:, 0]
    Y_target = target_sample[:, 0]

    l = l_scale(X_ref)

    fit_params = {'name': 'radial', 'l': l / 5, 'sigma': 1}
    mmd_params = {'name': 'radial', 'l': l / 5, 'sigma': 1}

    transport_params = {'X': X_ref, 'Y':  X_target, 'fit_kernel_params': fit_params,
                        'mmd_kernel_params': mmd_params, 'normalize': False,
                        'reg_lambda': 1e-5, 'print_freq': 100, 'learning_rate': .1,
                         'nugget': 1e-4, 'X_tilde': X_ref, 'alpha_y': [], 'alpha_x': False}

    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=t_iter)

    Z_ref = transport_kernel.map(X_ref).T
    cond_transport_params = {'Z_ref': Z_ref, 'Y_ref': Y_ref, 'X_target': X_target, 'Y_target': Y_target,
                        'fit_kernel_params': fit_params,'mmd_kernel_params': mmd_params, 'normalize': False,
                        'reg_lambda':  1e-5, 'print_freq': 10, 'learning_rate': .03,
                        'nugget': 1e-4, 'X_tilde': Z_ref, 'alpha_y': [], 'alpha_x': False}

    cond_transport_kernel = CondTransportKernel(cond_transport_params)
    train_kernel(cond_transport_kernel, n_iter= 6 * t_iter)
    sample = cond_transport_kernel.map(Z_ref, Y_ref)

    slice_samples = []
    N = len(Z_ref)

    slice_vals =  [-1.1, 0, 1.1]
    for z in slice_vals :
        z_slice = torch.full([300], z)
        idxs = torch.LongTensor(random.choices(list(range(N)), k= 300))

        slice_sample = cond_transport_kernel.map(z_slice,Y_ref[idxs])
        slice_samples.append(slice_sample)

    for i,csample in enumerate(slice_samples):
        csample = csample.T[1].T
        plt.hist(csample.detach().numpy(), label = f'z = {slice_vals[i]}')
    plt.savefig(f'{save_dir}/cond_hist.png')
    clear_plt()

    #target_sample = torch.concat([geq_1d(X_target),geq_1d(Y_target)], dim = 1)
    target_sample = torch.concat([geq_1d(Y_target), geq_1d(X_target)], dim=1)

    sample = sample.detach()
    sample = sample.T
    Tsample = deepcopy(sample)
    Tsample[0] = deepcopy(sample[1])
    Tsample[1] =  deepcopy(sample[0])
    sample = Tsample.T

    sample_scatter(sample, f'{save_dir}/cond_sample.png', bins=20, d=2, range = [[-3.1,3.1],[-1.1,1.1]])
    sample_hmap(sample, f'{save_dir}/cond_sample_map.png', bins=25, d=2, range = [[-3.1,3.1],[-1.1,1.1]])

    sample_scatter(target_sample, f'{save_dir}/target_sample.png', bins=20, d=2, range = [[-3.1,3.1],[-1.1,1.1]])
    sample_hmap(target_sample, f'{save_dir}/target_sample_map.png', bins=25, d=2, range = [[-3.1,3.1],[-1.1,1.1]])


def run():
    ref_gen = sample_normal
    N = 10000

    target_gen = mgan1
    conditional_transport_exp(ref_gen, target_gen, N, exp_name='mgan1')

    target_gen =  mgan2
    conditional_transport_exp(ref_gen, target_gen, N, exp_name='mgan2')

    target_gen = mgan3
    conditional_transport_exp(ref_gen, target_gen, N, exp_name='mgan2')




if __name__=='__main__':
    run()
