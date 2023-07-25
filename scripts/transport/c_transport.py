import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap
import os
from copy import deepcopy
from get_data import sample_banana, sample_normal


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
        self.X_mu = geq_1d(torch.tensor(base_params['X_mu'], device=self.device, dtype=self.dtype))
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

        self.alpha_z = (1 / self.Nx) * torch.ones(self.Nx, device=self.device, dtype=self.dtype)
        self.alpha_y = (1 / self.Ny) * torch.ones(self.Ny, device=self.device, dtype=self.dtype)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y
        self.iters = 0


    def init_Z(self):
        return torch.zeros(self.Y_mu.shape, device=self.device, dtype=self.dtype)


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def map(self, x_mu, y_eta):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        w = torch.concat([x_mu, y_eta], dim=1)
        Lambda = self.get_Lambda()
        z = self.fit_kernel(self.X, w).T @ Lambda
        return torch.concat([x_mu, z + y_eta], dim = 1)


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


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg()
        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict


def base_kernel_transport(Y_eta, Y_mu, params, n_iter = 1001):
    base_params = {'X': Y_eta, 'Y': Y_mu, 'reg_lambda': 1e-5,'normalize': False,
                   'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                   'print_freq': 100, 'learning_rate': .1, 'nugget': 1e-4}
    transport_kernel = TransportKernel(base_params)
    train_kernel(transport_kernel, n_iter=n_iter)
    return transport_kernel



def cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 10001):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 1e-5,
                        'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                        'print_freq': 100, 'learning_rate': .06, 'nugget': 1e-4}
    ctransport_kernel = CondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel



def train_cond_transport(ref_gen, target_gen, params, N = 1000, n_iter = 1001,process_funcs = [],
                         base_model_trainer = base_kernel_transport, cond_model_trainer = cond_kernel_transport):

    ref_sample = ref_gen(N)
    target_sample = target_gen(N)

    trained_models = []

    if len(process_funcs):
        forward = process_funcs[0]
        ptarget_sample = forward(target_sample)
    else:
        ptarget_sample = target_sample

    Y_eta = ref_sample[:, 0]
    Y_mu = target_sample[:, 0]
    trained_models.append(base_model_trainer(Y_eta, Y_mu, params, n_iter))


    for i in range(1, len(ptarget_sample[0])):
        X_mu = target_sample[:, :i]
        Y_mu = target_sample[:, i]
        Y_eta = ref_sample[:,i]

        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, params, n_iter))
    return trained_models


def compositional_gen(trained_models, ref_sample):
    Y_eta =  ref_sample[:, 0]
    base_model = trained_models[0]
    X = base_model.map(Y_eta)

    for i in range(1, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, i]
        X = model.map(X, Y_eta)
    return X



def conditional_transport_exp(ref_gen, target_gen, N = 1000, n_iter = 1001,
                              exp_name= 'exp', plt_range = None, process_funcs = []):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    l = l_scale(torch.tensor(ref_gen(N)))
    mmd_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    fit_params = {'name': 'r_quadratic', 'l': l * torch.exp(torch.tensor(-1.25)), 'alpha': 1}
    exp_params = {'fit': mmd_params, 'mmd': fit_params}

    trained_models = train_cond_transport(ref_gen, target_gen, exp_params, N, n_iter,process_funcs)
    gen_sample = compositional_gen(trained_models, ref_gen(N))

    if len(process_funcs):
        backward = process_funcs[1]
        gen_sample = backward(gen_sample)

    d = len(gen_sample[0])
    if d <=2:
        sample_scatter(gen_sample, f'{save_dir}/gen_scatter.png', bins=25, d = d, range = plt_range)
        sample_scatter(target_gen(N), f'{save_dir}/target.png', bins=25, d=d, range=plt_range)
    return True


def run():
    ref_gen = sample_normal
    target_gen = sample_banana
    range = [[-2.5,2.5],[-1,5]]
    process_funcs = [flip_2tensor, flip_2tensor]
    #process_funcs = []
    conditional_transport_exp(ref_gen, target_gen, exp_name= 'test', N = 5000, n_iter = 8000,
                              plt_range=range, process_funcs=process_funcs)


if __name__=='__main__':
    run()







