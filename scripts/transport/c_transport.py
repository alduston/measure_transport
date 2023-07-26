import torch
import torch.nn as nn
from transport_kernel import  TransportKernel, l_scale, get_kernel, clear_plt
from fit_kernel import train_kernel, sample_scatter, sample_hmap
import os
from copy import deepcopy
from get_data import sample_banana, sample_normal, mgan2, sample_spirals, sample_pinweel
from K_VAE import VAETransportKernel
import matplotlib.pyplot as plt



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

        self.test = False
        if 'Y_eta_test' in base_params.keys():
            self.test = True
            self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))

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


    def map(self, x_mu, y_eta):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        w = torch.concat([x_mu, y_eta], dim=1)
        Lambda = self.get_Lambda()
        z = self.fit_kernel(self.X, w).T @ Lambda
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


    def loss_test(self):
        x_mu = self.X_mu
        y_eta = self.Y_eta_test
        target = self.Y
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


class VAECondTransportKernel(nn.Module):
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
        self.eps = self.get_eps(self.Y_mu)

        self.test = False
        if 'Y_eta_test' in base_params.keys():
            self.test = True
            self.Y_eta_test = geq_1d(torch.tensor(base_params['Y_eta_test'], device=self.device, dtype=self.dtype))

        self.fit_kernel = get_kernel(self.params['fit_kernel_params'], self.device)
        self.fit_kXX = self.fit_kernel(self.X, self.X)

        self.nugget_matrix = self.params['nugget'] * torch.eye(self.Nx, device=self.device, dtype=self.dtype)
        self.fit_kXX_inv = torch.linalg.inv(self.fit_kXX + self.nugget_matrix)

        self.mmd_kernel = get_kernel(self.params['mmd_kernel_params'], self.device)
        self.Z = nn.Parameter(self.init_Z(), requires_grad=True)
        self.mmd_YY = self.mmd_kernel(self.Y, self.Y)

        n = len(self.Y_mu[0])
        self.t_idx = torch.tril_indices(row=n, col=n, offset=0)

        self.alpha_z = self.p_vec(self.Nx)
        self.alpha_y = self.p_vec(self.Ny)
        self.E_mmd_YY = self.alpha_y.T @ self.mmd_YY @ self.alpha_y
        self.iters = 0


    def p_vec(self, n):
        return torch.full([n], 1 / n, device=self.device, dtype=self.dtype)


    def get_sig_base(self, n):
        sig_base = []
        for i in range(n):
            sig_base += [0.0]*i
            sig_base += [1.0]
        return sig_base


    def init_Z(self):
        n = len(self.Y_mu[0])
        N = self.Nx
        Z_mean = torch.zeros([N,n], device=self.device, dtype=self.dtype)
        sig_base = self.get_sig_base(n)
        sig_base = torch.tensor(sig_base, device=self.device, dtype=self.dtype)

        ly = l_scale(self.Y_mu)
        Z_var = ly * torch.stack([sig_base for i in range(N)])
        Z = torch.concat([Z_mean, Z_var], dim=1)
        return Z


    def v_to_lt(self, V, n = 0, t_idx = []):
        N = len(V)
        if not n:
            n = V.shape[1]-1
        if not len(t_idx):
            t_idx = torch.tril_indices(row=n, col=n, offset=0)
        m = torch.zeros((N, n, n), device = self.device, dtype = self.dtype)
        m[:, t_idx[0], t_idx[1]] = V
        return m


    def get_mu_sig(self, Z = []):
        n = len(self.Y_mu[0])
        if not len(Z):
            Z = self.Z
        mu = Z[:, :n]
        sig_vs = Z[:, n:]

        t_idx = self.t_idx
        sig_ltms = self.v_to_lt(sig_vs,n,t_idx)
        sig_ltms_T = torch.transpose(sig_ltms,1,2)

        sig_ms = torch.matmul(sig_ltms, sig_ltms_T)
        return mu, sig_ms


    def get_sample(self, params = {}):
        if not len(params):
            mu,sig = self.get_mu_sig()
            params = {'mu': mu, 'sig': sig, 'eps': self.eps}

        eps = torch.unsqueeze(self.eps,2)
        #eps = torch.unsqueeze(self.get_eps(self.Y_mu), 2)
        diffs = torch.matmul(params['sig'], eps)
        Z_sample = params['mu'] + diffs.reshape(diffs.shape[:-1])
        return Z_sample

    def get_eps(self, x):
        eps_shape = list(x.shape)
        return torch.randn(eps_shape, device=self.device, dtype=self.dtype)


    def get_Lambda(self):
        return self.fit_kXX_inv @ self.Z


    def map(self, x_mu, y_eta):
        y_eta = geq_1d(torch.tensor(y_eta, device=self.device, dtype=self.dtype))
        x_mu = geq_1d(torch.tensor(x_mu, device=self.device, dtype=self.dtype))
        w = torch.concat([x_mu, y_eta], dim=1)
        Lambda = self.get_Lambda()
        z = self.fit_kernel(self.X, w).T @ Lambda
        mu, sig = self.get_mu_sig(z)
        eps = self.get_eps(y_eta)
        z_sample = self.get_sample({'mu': mu, 'sig': sig, 'eps': eps})
        return torch.concat([x_mu, z_sample + y_eta], dim = 1)


    def loss_mmd(self):
        map_vec = torch.concat([self.X_mu, self.Y_eta + self.get_sample()], dim=1)
        target = self.Y

        mmd_ZZ = self.mmd_kernel(map_vec, map_vec)
        mmd_ZY = self.mmd_kernel(map_vec, target)

        alpha_z = self.alpha_z
        alpha_y = self.alpha_y

        Ek_ZZ = alpha_z @ mmd_ZZ @ alpha_z
        Ek_ZY = alpha_z @ mmd_ZY @ alpha_y
        Ek_YY = self.E_mmd_YY
        return Ek_ZZ - (2 * Ek_ZY) + Ek_YY


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


    def loss_test(self):
        x_mu = self.X_mu
        y_eta = self.Y_eta_test
        target = self.Y
        map_vec = self.map(x_mu, y_eta)
        return self.mmd(map_vec, target)


    def loss_reg(self):
        Z = geq_1d(self.Z)
        return  self.params['reg_lambda'] * torch.trace(Z.T @ self.fit_kXX_inv @ Z)


    def loss(self):
        loss_mmd = self.loss_mmd()
        loss_reg = self.loss_reg()


        loss = loss_mmd + loss_reg
        loss_dict = {'fit': loss_mmd.detach().cpu(),
                     'reg': loss_reg.detach().cpu(),
                     'total': loss.detach().cpu()}
        return loss, loss_dict




def base_kernel_transport(Y_eta, Y_mu, params, n_iter = 1001, Y_eta_test = []):
    transport_params = {'X': Y_eta, 'Y': Y_mu, 'reg_lambda': 5e-6,'normalize': False,
                   'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                   'print_freq': 50, 'learning_rate': .1, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=n_iter)
    return transport_kernel


def base_VAEkernel_transport(Y_eta, Y_mu, params, n_iter = 1001, Y_eta_test = []):
    transport_params = {'X': Y_eta, 'Y': Y_mu, 'reg_lambda': 5e-6,'normalize': False,
                   'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                   'print_freq': 50, 'learning_rate': .1, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    transport_kernel = VAETransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=n_iter)
    return transport_kernel



def cond_kernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 10001, Y_eta_test = []):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 5e-6,
                        'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                        'print_freq': 50, 'learning_rate': .1, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    ctransport_kernel = CondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel


def cond_VAEkernel_transport(X_mu, Y_mu, Y_eta, params, n_iter = 10001, Y_eta_test = []):
    transport_params = {'X_mu': X_mu, 'Y_mu': Y_mu, 'Y_eta': Y_eta, 'reg_lambda': 5e-6,
                        'fit_kernel_params': params['mmd'], 'mmd_kernel_params': params['fit'],
                        'print_freq': 50, 'learning_rate': .06, 'nugget': 1e-4}
    if len(Y_eta_test):
        transport_params['Y_eta_test'] = Y_eta_test
    ctransport_kernel = VAECondTransportKernel(transport_params)
    train_kernel(ctransport_kernel, n_iter)
    return ctransport_kernel



def train_cond_transport(ref_gen, target_gen, params, N = 1000, n_iter = 1001,process_funcs = [],
                         base_model_trainer = base_kernel_transport, cond_model_trainer = cond_kernel_transport):

    ref_sample = ref_gen(N)
    target_sample = target_gen(N)
    test_sample = ref_gen(N)

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
        Y_mu = target_sample[:, i]
        Y_eta = ref_sample[:,i]
        Y_eta_test = test_sample[:, i]

        trained_models.append(cond_model_trainer(X_mu, Y_mu, Y_eta, params, n_iter, Y_eta_test))
    return trained_models


def compositional_gen(trained_models, ref_sample):
    ref = geq_1d(ref_sample)
    Y_eta =  ref_sample[:, 0]
    base_model = trained_models[0]
    X = base_model.map(Y_eta)

    for i in range(1, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, i]
        X = model.map(X, Y_eta)
    return X


def conditional_gen(trained_models, ref_sample, cond_sample):
    ref_sample = geq_1d(ref_sample)
    X = geq_1d(cond_sample)
    Y_eta = ref_sample[:, 0]
    for i in range(0, len(trained_models)):
        model = trained_models[i]
        Y_eta = ref_sample[:, i]
        X = model.map(X, Y_eta)
    return X


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

    trained_models = train_cond_transport(ref_gen, target_gen, exp_params, N, n_iter, process_funcs)
                                          #,base_model_trainer=base_VAEkernel_transport
                                          #,cond_model_trainer=cond_VAEkernel_transport)

    gen_sample = compositional_gen(trained_models, ref_gen(N))

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
        sample_scatter(target_gen(N), f'{save_dir}/target.png', bins=25, d=d, range=plt_range)
    return True

#003641
def run():
    ref_gen = sample_normal
    target_gen = sample_spirals
    range = [[-3,3],[-3,3]]
    slice_range = [-3,3]
    process_funcs = []
    process_funcs = [flip_2tensor, flip_2tensor ]
    conditional_transport_exp(ref_gen, target_gen, exp_name= 'spiral_flip', N = 5000, n_iter = 10000,
                              plt_range=range, slice_range= slice_range, process_funcs=process_funcs, slice_vals=[0])


if __name__=='__main__':
    run()







